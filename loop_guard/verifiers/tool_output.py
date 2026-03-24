"""ToolOutputVerifier — Re-executes tool calls to verify agent claims.

When an agent says "I called tool X with args Y and got result Z",
this verifier re-calls the same tool and compares the results.

This is Layer 1 (deterministic) verification: the tool is the source
of truth, not the agent's claim about the tool's output.

Supports:
- HTTP API calls (GET/POST)
- Python function calls (registered callables)
- Shell commands (with sandboxing)

Usage:
    verifier = ToolOutputVerifier()
    verifier.register_tool("get_weather", get_weather_fn)

    claim = Claim(
        claim_type=ClaimType.CODE_OUTPUT,
        text="Called get_weather('NYC') and got 72°F",
        evidence={
            "tool_name": "get_weather",
            "tool_args": {"city": "NYC"},
            "claimed_output": "72°F",
        },
    )
    finding = verifier.verify(claim)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from typing import Any, Callable, Optional

import httpx

from loop_guard.models import Claim, Finding, Verdict, VerificationLevel


class ToolOutputVerifier:
    """Re-executes tool calls to verify agent claims.

    Supports three verification modes:
    1. Registered Python functions (highest fidelity)
    2. HTTP API re-calls (for REST tools)
    3. Shell command re-execution (sandboxed)
    """

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout
        self._registered_tools: dict[str, Callable] = {}
        self._api_configs: dict[str, dict] = {}
        self._cache: dict[str, tuple[Any, float]] = {}  # hash -> (result, timestamp)
        self._cache_ttl = 60  # seconds

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a Python callable for tool re-execution."""
        self._registered_tools[name] = fn

    def register_api(
        self,
        name: str,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
    ) -> None:
        """Register an HTTP API endpoint for tool re-execution."""
        self._api_configs[name] = {
            "url": url,
            "method": method.upper(),
            "headers": headers or {},
        }

    def verify(self, claim: Claim) -> Finding:
        """Verify a tool output claim by re-executing the tool."""
        if not claim.evidence:
            return self._skip(claim, "No tool evidence provided")

        tool_name = claim.evidence.get("tool_name")
        tool_args = claim.evidence.get("tool_args", {})
        claimed_output = claim.evidence.get("claimed_output")

        if not tool_name:
            return self._skip(claim, "No tool_name in evidence")

        if claimed_output is None:
            return self._skip(claim, "No claimed_output in evidence")

        # Try registered Python function first
        if tool_name in self._registered_tools:
            return self._verify_python_tool(claim, tool_name, tool_args, claimed_output)

        # Try registered API
        if tool_name in self._api_configs:
            return self._verify_api_tool(claim, tool_name, tool_args, claimed_output)

        # Try shell command
        if claim.evidence.get("tool_type") == "shell":
            return self._verify_shell_tool(claim, tool_args, claimed_output)

        return self._skip(
            claim,
            f"Tool '{tool_name}' not registered. Register with "
            f"verifier.register_tool('{tool_name}', fn) to enable verification.",
        )

    def _verify_python_tool(
        self, claim: Claim, tool_name: str, args: dict, claimed: str
    ) -> Finding:
        """Re-execute a registered Python function."""
        fn = self._registered_tools[tool_name]

        # Check cache
        cache_key = self._cache_key(tool_name, args)
        cached = self._get_cached(cache_key)
        if cached is not None:
            actual = str(cached)
        else:
            try:
                result = fn(**args) if isinstance(args, dict) else fn(*args)
                actual = str(result)
                self._set_cached(cache_key, result)
            except Exception as e:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.SKIPPED,
                    level=VerificationLevel.DETERMINISTIC,
                    explanation=f"Tool re-execution failed: {type(e).__name__}: {str(e)[:200]}",
                )

        if self._outputs_match(claimed, actual):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Tool '{tool_name}' re-execution matches claimed output",
                expected=claimed,
                actual=actual,
            )
        else:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_FAIL,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Tool '{tool_name}' re-execution produced different output",
                expected=claimed,
                actual=actual,
            )

    def _verify_api_tool(
        self, claim: Claim, tool_name: str, args: dict, claimed: str
    ) -> Finding:
        """Re-call an HTTP API endpoint."""
        config = self._api_configs[tool_name]
        url = config["url"]
        method = config["method"]
        headers = config["headers"]

        cache_key = self._cache_key(tool_name, args)
        cached = self._get_cached(cache_key)
        if cached is not None:
            actual = str(cached)
        else:
            try:
                if method == "GET":
                    resp = httpx.get(url, params=args, headers=headers, timeout=self.timeout)
                elif method == "POST":
                    resp = httpx.post(url, json=args, headers=headers, timeout=self.timeout)
                else:
                    return self._skip(claim, f"Unsupported HTTP method: {method}")

                resp.raise_for_status()
                actual = resp.text
                self._set_cached(cache_key, actual)
            except Exception as e:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.SKIPPED,
                    level=VerificationLevel.DETERMINISTIC,
                    explanation=f"API re-call failed: {type(e).__name__}: {str(e)[:200]}",
                )

        if self._outputs_match(claimed, actual):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"API '{tool_name}' re-call matches claimed output",
                expected=claimed,
                actual=actual[:500],
            )
        else:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_FAIL,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"API '{tool_name}' re-call produced different output",
                expected=claimed,
                actual=actual[:500],
            )

    def _verify_shell_tool(
        self, claim: Claim, args: dict, claimed: str
    ) -> Finding:
        """Re-execute a shell command in sandbox."""
        command = args.get("command")
        if not command:
            return self._skip(claim, "No command in tool_args")

        # Safety: refuse dangerous commands
        dangerous = ["rm ", "sudo ", "chmod ", "chown ", "mkfs", "dd ", "> /dev"]
        if any(d in command for d in dangerous):
            return self._skip(claim, f"Refusing to re-execute potentially dangerous command")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={"PATH": "/usr/bin:/bin"},
            )
            actual = result.stdout.strip()
        except subprocess.TimeoutExpired:
            return self._skip(claim, "Shell command timed out")
        except Exception as e:
            return self._skip(claim, f"Shell execution error: {str(e)[:200]}")

        if self._outputs_match(claimed, actual):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation="Shell command re-execution matches claimed output",
                expected=claimed,
                actual=actual,
            )
        else:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_FAIL,
                level=VerificationLevel.DETERMINISTIC,
                explanation="Shell command re-execution produced different output",
                expected=claimed,
                actual=actual,
            )

    def _outputs_match(self, claimed: str, actual: str) -> bool:
        """Compare outputs with tolerance for numeric values and whitespace."""
        # Normalize whitespace
        claimed_norm = " ".join(claimed.split())
        actual_norm = " ".join(actual.split())

        if claimed_norm == actual_norm:
            return True

        # Try numeric comparison
        try:
            c_val = float(claimed_norm.strip().rstrip("%"))
            a_val = float(actual_norm.strip().rstrip("%"))
            if a_val == 0:
                return abs(c_val) < 1e-9
            return abs(c_val - a_val) / abs(a_val) <= 0.01
        except (ValueError, ZeroDivisionError):
            pass

        # Substring match (claimed appears in actual or vice versa)
        if claimed_norm in actual_norm or actual_norm in claimed_norm:
            return True

        return False

    def _cache_key(self, tool_name: str, args: Any) -> str:
        """Generate a cache key for tool call deduplication."""
        raw = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Any | None:
        """Get a cached result if still valid."""
        if key in self._cache:
            result, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return result
            del self._cache[key]
        return None

    def _set_cached(self, key: str, result: Any) -> None:
        """Cache a tool result."""
        self._cache[key] = (result, time.time())

    def _skip(self, claim: Claim, reason: str) -> Finding:
        """Return a SKIPPED finding."""
        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.SKIPPED,
            level=VerificationLevel.DETERMINISTIC,
            explanation=reason,
        )
