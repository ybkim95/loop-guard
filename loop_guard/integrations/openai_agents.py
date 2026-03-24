"""OpenAI Agents SDK integration for loop-guard.

Wraps OpenAI agent interactions with deterministic verification.
Works with the openai Python SDK (Responses API, Assistants, or custom loops).

Usage:
    from loop_guard.integrations.openai_agents import OpenAIGuard

    guard = OpenAIGuard(api_key="sk-...")

    # Wrap any OpenAI agent loop
    for step in agent_steps:
        response = client.chat.completions.create(...)
        findings = guard.verify_response(step_id=i, response=response)
        # or for tool calls:
        findings = guard.verify_tool_call(step_id=i, tool_name="...", args={...}, result="...", claimed_output="...")
"""

from __future__ import annotations

import json
import re
from typing import Any

from loop_guard.guard import LoopGuard
from loop_guard.models import (
    Claim,
    ClaimType,
    Finding,
    Verdict,
    VerificationLevel,
)


def _safe_import_openai():
    """Import openai lazily so the module can be loaded without it installed."""
    try:
        import openai
        return openai
    except ImportError:
        return None


class OpenAIGuard:
    """Wraps OpenAI agent interactions with loop-guard verification.

    Intercepts tool calls and agent responses, extracts verifiable claims,
    and runs them through the LoopGuard verification engine.

    Args:
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
        config: Optional LoopGuard config dict.
    """

    def __init__(self, api_key: str | None = None, config: dict | None = None):
        self._config = config or {}
        self._guard = LoopGuard(config=self._config)
        self._api_key = api_key
        self._client = None  # lazy-init OpenAI client

        # Track conversation for cross-step verification
        self._history: list[dict] = []

    # ── Properties ────────────────────────────────────────────────────

    @property
    def findings(self) -> list[Finding]:
        """All findings accumulated across verified steps."""
        return self._guard.findings

    @property
    def summary(self) -> dict:
        """Summary statistics of all verification findings."""
        return self._guard.summary

    def report(self, format: str = "terminal", path: str | None = None) -> str | dict:
        """Generate a verification report.

        Args:
            format: One of 'terminal', 'json', 'html'.
            path: Output file path (for json/html formats).
        """
        return self._guard.report(format=format, path=path)

    # ── Core Verification Methods ─────────────────────────────────────

    def verify_response(
        self,
        step_id: int,
        response: Any,
        code: str | None = None,
        files: list[str] | None = None,
    ) -> list[Finding]:
        """Verify an OpenAI ChatCompletion response.

        Extracts text content and any tool calls from the response,
        then feeds them through LoopGuard for claim verification.

        Args:
            step_id: The step number in the agent loop.
            response: An OpenAI ChatCompletion response object or dict.
            code: Optional code that was executed to produce this response.
            files: Optional list of files referenced or modified.

        Returns:
            List of verification findings.
        """
        text, tool_calls = self._extract_from_response(response)

        # Store in history for cross-step consistency checks
        self._history.append({
            "step_id": step_id,
            "role": "assistant",
            "content": text,
            "tool_calls": tool_calls,
        })

        # Build metadata for the step
        metadata = {}
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # Run through LoopGuard
        findings = self._guard.step(
            output=text,
            step_id=step_id,
            code=code,
            files=files,
            metadata=metadata,
        )

        # Additional: cross-reference numeric claims against prior steps
        cross_findings = self._check_cross_step_consistency(step_id, text)
        findings.extend(cross_findings)

        return findings

    def verify_tool_call(
        self,
        step_id: int,
        tool_name: str,
        args: dict,
        result: str,
        claimed_output: str | None = None,
    ) -> list[Finding]:
        """Verify a tool call made by an OpenAI agent.

        Checks that the tool result is consistent with what the agent
        claims about it. If claimed_output differs from actual result,
        flags a verification failure.

        Args:
            step_id: The step number in the agent loop.
            tool_name: Name of the tool/function called.
            args: Arguments passed to the tool.
            result: The actual result returned by the tool.
            claimed_output: What the agent claims the tool returned (optional).

        Returns:
            List of verification findings.
        """
        findings = []

        # Build a textual representation for LoopGuard
        output_text = (
            f"Tool call: {tool_name}\n"
            f"Arguments: {json.dumps(args, default=str)}\n"
            f"Result: {result}\n"
        )
        if claimed_output:
            output_text += f"Agent claimed: {claimed_output}\n"

        # Run through LoopGuard for metric/claim extraction
        guard_findings = self._guard.step(
            output=output_text,
            step_id=step_id,
            metadata={
                "tool_name": tool_name,
                "tool_args": args,
                "tool_result": result,
            },
        )
        findings.extend(guard_findings)

        # Direct comparison: if claimed_output is given, compare to actual result
        if claimed_output is not None:
            comparison_finding = self._compare_tool_output(
                step_id=step_id,
                tool_name=tool_name,
                actual=result,
                claimed=claimed_output,
            )
            if comparison_finding:
                findings.append(comparison_finding)

        # Check for code execution claims embedded in the result
        code_findings = self._verify_code_claims(step_id, tool_name, args, result)
        findings.extend(code_findings)

        return findings

    def verify_message(
        self,
        step_id: int,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
    ) -> list[Finding]:
        """Verify a single message from the conversation.

        Can be used for user messages, system messages, or assistant
        messages that aren't wrapped in a ChatCompletion response.

        Args:
            step_id: The step number in the agent loop.
            role: Message role ('user', 'assistant', 'system', 'tool').
            content: The message text content.
            tool_calls: Optional list of tool call dicts.

        Returns:
            List of verification findings.
        """
        self._history.append({
            "step_id": step_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
        })

        metadata = {"role": role}
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        findings = self._guard.step(
            output=f"[{role}] {content}",
            step_id=step_id,
            metadata=metadata,
        )

        # Cross-step consistency
        if role == "assistant":
            cross_findings = self._check_cross_step_consistency(step_id, content)
            findings.extend(cross_findings)

        return findings

    # ── Internal Helpers ──────────────────────────────────────────────

    def _extract_from_response(self, response: Any) -> tuple[str, list[dict]]:
        """Extract text content and tool calls from an OpenAI response.

        Handles both response objects and plain dicts.
        """
        text = ""
        tool_calls = []

        # Handle ChatCompletion object (openai v2.x)
        if hasattr(response, "choices"):
            choices = response.choices
            if choices:
                message = choices[0].message
                text = message.content or ""
                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tc in message.tool_calls:
                        tool_calls.append({
                            "id": tc.id if hasattr(tc, "id") else "",
                            "function": {
                                "name": tc.function.name if hasattr(tc, "function") else "",
                                "arguments": tc.function.arguments if hasattr(tc, "function") else "{}",
                            },
                        })
        # Handle dict responses
        elif isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "") or ""
                for tc in message.get("tool_calls") or []:
                    tool_calls.append(tc)

        return text, tool_calls

    def _compare_tool_output(
        self,
        step_id: int,
        tool_name: str,
        actual: str,
        claimed: str,
    ) -> Finding | None:
        """Compare actual tool output to what the agent claimed."""
        actual_stripped = actual.strip()
        claimed_stripped = claimed.strip()

        if actual_stripped == claimed_stripped:
            return None  # exact match, no issue

        # Try numeric comparison for metric-like outputs
        actual_nums = self._extract_numbers(actual_stripped)
        claimed_nums = self._extract_numbers(claimed_stripped)

        if actual_nums and claimed_nums:
            # Check if the key numbers match
            mismatches = []
            for cn in claimed_nums:
                if not any(abs(cn - an) < max(abs(cn) * 0.001, 1e-9) for an in actual_nums):
                    mismatches.append(cn)

            if mismatches:
                return Finding(
                    step_id=step_id,
                    claim=Claim(
                        claim_type=ClaimType.CODE_OUTPUT,
                        source_step=step_id,
                        text=f"Tool '{tool_name}' output claim",
                        verifiable=True,
                        evidence={"tool": tool_name, "actual": actual, "claimed": claimed},
                    ),
                    verdict=Verdict.VERIFIED_FAIL,
                    level=VerificationLevel.DETERMINISTIC,
                    explanation=(
                        f"Tool '{tool_name}' returned different values than claimed. "
                        f"Mismatched numbers: {mismatches}"
                    ),
                    expected=claimed,
                    actual=actual,
                )

        # Fuzzy text comparison: flag if substantially different
        if len(actual_stripped) > 0 and len(claimed_stripped) > 0:
            # Simple overlap check
            actual_words = set(actual_stripped.lower().split())
            claimed_words = set(claimed_stripped.lower().split())
            if actual_words and claimed_words:
                overlap = len(actual_words & claimed_words) / max(len(actual_words), len(claimed_words))
                if overlap < 0.5:
                    return Finding(
                        step_id=step_id,
                        claim=Claim(
                            claim_type=ClaimType.CODE_OUTPUT,
                            source_step=step_id,
                            text=f"Tool '{tool_name}' output claim",
                            verifiable=True,
                            evidence={"tool": tool_name, "actual": actual, "claimed": claimed},
                        ),
                        verdict=Verdict.FLAG_FOR_REVIEW,
                        level=VerificationLevel.RULE_BASED,
                        explanation=(
                            f"Tool '{tool_name}' output diverges significantly from agent's claim "
                            f"(word overlap: {overlap:.0%}). Manual review recommended."
                        ),
                        expected=claimed,
                        actual=actual,
                    )

        return None

    def _verify_code_claims(
        self,
        step_id: int,
        tool_name: str,
        args: dict,
        result: str,
    ) -> list[Finding]:
        """Check for suspicious patterns in code execution tool results."""
        findings = []

        # Detect if this is a code execution tool
        code_tools = {"code_interpreter", "python", "exec", "execute", "run_code", "code"}
        if tool_name.lower() not in code_tools:
            return findings

        code = args.get("code", args.get("input", args.get("script", "")))
        if not code:
            return findings

        # Check: does the result contain numbers that code should produce?
        result_nums = self._extract_numbers(result)
        if not result_nums:
            return findings

        # Flag suspiciously round numbers from computation
        for num in result_nums:
            if num != 0 and num == round(num) and abs(num) > 1:
                # Perfectly round numbers from statistical computations are suspicious
                if any(kw in code.lower() for kw in ["mean", "std", "median", "average", "correlation"]):
                    findings.append(Finding(
                        step_id=step_id,
                        claim=Claim(
                            claim_type=ClaimType.METRIC,
                            source_step=step_id,
                            text=f"Code execution produced perfectly round number: {num}",
                            verifiable=True,
                            evidence={"code": code[:200], "number": num},
                        ),
                        verdict=Verdict.FLAG_FOR_REVIEW,
                        level=VerificationLevel.RULE_BASED,
                        explanation=(
                            f"Statistical computation produced a perfectly round number ({num}). "
                            f"This is unlikely for real data and may indicate a hallucinated result."
                        ),
                    ))

        return findings

    def _check_cross_step_consistency(self, step_id: int, text: str) -> list[Finding]:
        """Check if numeric claims in this step contradict earlier steps."""
        findings = []
        current_nums = self._extract_labeled_numbers(text)

        if not current_nums:
            return findings

        # Look at previous assistant messages for contradictions
        for prev in self._history[:-1]:
            if prev["role"] != "assistant" or not prev.get("content"):
                continue

            prev_nums = self._extract_labeled_numbers(prev["content"])
            for label, cur_val in current_nums.items():
                if label in prev_nums:
                    prev_val = prev_nums[label]
                    # Flag if the same labeled metric changed without explanation
                    if prev_val != 0 and cur_val != 0:
                        rel_diff = abs(cur_val - prev_val) / max(abs(prev_val), 1e-9)
                        if rel_diff > 0.5:
                            findings.append(Finding(
                                step_id=step_id,
                                claim=Claim(
                                    claim_type=ClaimType.METRIC,
                                    source_step=step_id,
                                    text=f"'{label}' changed from {prev_val} to {cur_val}",
                                    verifiable=True,
                                    evidence={
                                        "label": label,
                                        "previous": prev_val,
                                        "current": cur_val,
                                        "previous_step": prev["step_id"],
                                    },
                                ),
                                verdict=Verdict.FLAG_FOR_REVIEW,
                                level=VerificationLevel.RULE_BASED,
                                explanation=(
                                    f"Metric '{label}' changed by {rel_diff:.0%} between step "
                                    f"{prev['step_id']} ({prev_val}) and step {step_id} ({cur_val}). "
                                    f"Verify this is expected."
                                ),
                                expected=str(prev_val),
                                actual=str(cur_val),
                            ))

        return findings

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        """Extract all numeric values from text."""
        pattern = r"-?\d+\.?\d*(?:e[+-]?\d+)?"
        matches = re.findall(pattern, text, re.IGNORECASE)
        nums = []
        for m in matches:
            try:
                nums.append(float(m))
            except ValueError:
                pass
        return nums

    @staticmethod
    def _extract_labeled_numbers(text: str) -> dict[str, float]:
        """Extract labeled numeric values (e.g., 'mean: 4.5', 'accuracy = 0.95')."""
        pattern = r"(\w[\w\s]*?)\s*[:=]\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)"
        matches = re.findall(pattern, text, re.IGNORECASE)
        result = {}
        for label, value in matches:
            label = label.strip().lower()
            try:
                result[label] = float(value)
            except ValueError:
                pass
        return result
