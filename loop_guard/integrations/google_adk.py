"""Google ADK (Agent Development Kit) integration for loop-guard.

Provides two integration modes:
1. ADKGuard: Wraps ADK agent runs with loop-guard verification
2. Callback-based: Inject loop-guard as an ADK before/after_tool callback

Works with google-adk and google-genai SDKs.

Usage (standalone):
    from loop_guard.integrations.google_adk import ADKGuard

    guard = ADKGuard()

    # Your ADK agent loop
    for step_id, event in enumerate(agent.run(task)):
        findings = guard.verify_event(step_id, event)
        for f in findings:
            if f.verdict == Verdict.VERIFIED_FAIL:
                print(f"ALERT: {f.explanation}")

Usage (with Gemini directly):
    from loop_guard.integrations.google_adk import GeminiGuard

    guard = GeminiGuard(api_key="...")
    response = guard.generate(prompt="Analyze this data...")
    # Automatically verified
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from loop_guard.guard import LoopGuard
from loop_guard.models import (
    Claim,
    ClaimType,
    Finding,
    Verdict,
    VerificationLevel,
)
from loop_guard.provenance import ProvenanceChain
from loop_guard.verifiers.tool_output import ToolOutputVerifier


class ADKGuard:
    """Wraps Google ADK agent interactions with loop-guard verification.

    Designed to be framework-agnostic: you feed it events/outputs from
    your ADK agent, and it verifies claims deterministically.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.guard = LoopGuard(config={
            "use_llm_extraction": False,
            "verbosity": self.config.get("verbosity", "findings_only"),
            **self.config,
        })
        self.provenance = ProvenanceChain()
        self.tool_verifier = ToolOutputVerifier(
            timeout=self.config.get("tool_timeout", 30),
        )
        self._step_counter = 0

    def verify_event(
        self,
        event: dict | Any,
        step_id: int | None = None,
    ) -> list[Finding]:
        """Verify an ADK agent event.

        Accepts either a dict with keys like:
            {"type": "tool_call", "tool": "search", "args": {...}, "result": "..."}
        Or raw text output from the agent.
        """
        if step_id is None:
            step_id = self._step_counter
            self._step_counter += 1

        if isinstance(event, str):
            return self._verify_text(step_id, event)
        elif isinstance(event, dict):
            return self._verify_dict_event(step_id, event)
        else:
            # Try to extract text from ADK event objects
            text = self._extract_text_from_event(event)
            return self._verify_text(step_id, text)

    def verify_tool_call(
        self,
        step_id: int,
        tool_name: str,
        args: dict,
        result: str,
        claimed_output: str | None = None,
    ) -> list[Finding]:
        """Verify a specific tool call by re-executing it."""
        claim = Claim(
            claim_type=ClaimType.CODE_OUTPUT,
            source_step=step_id,
            text=f"Called {tool_name}({json.dumps(args)[:100]}) → {str(result)[:100]}",
            verifiable=True,
            evidence={
                "tool_name": tool_name,
                "tool_args": args,
                "claimed_output": claimed_output or str(result),
            },
        )

        finding = self.tool_verifier.verify(claim)
        self.guard.reporter.report_step([finding])
        self.guard.reporter.all_findings.append(finding)

        # Also feed the result through standard verification
        text_findings = self.guard.step(
            output=f"Tool call: {tool_name}\nArgs: {json.dumps(args)[:200]}\nResult: {str(result)[:500]}",
            step_id=step_id,
        )

        return [finding] + text_findings

    def verify_generation(
        self,
        step_id: int,
        prompt: str,
        response_text: str,
        tool_calls: list[dict] | None = None,
    ) -> list[Finding]:
        """Verify a Gemini generation response."""
        all_findings = []

        # Auto-detect dependencies on previous steps
        deps = self.provenance.auto_detect_dependencies(step_id, response_text)

        # Feed through standard loop-guard
        findings = self.guard.step(
            output=response_text,
            step_id=step_id,
            metadata={"prompt": prompt[:200]},
        )
        all_findings.extend(findings)

        # Record in provenance chain
        for f in findings:
            node = self.provenance.record(step_id, f.claim, f, depends_on=deps)
            if node.tainted:
                all_findings.append(Finding(
                    step_id=step_id,
                    claim=f.claim,
                    verdict=Verdict.FLAG_FOR_REVIEW,
                    level=VerificationLevel.RULE_BASED,
                    explanation=f"TAINTED: {node.tainted_reason}",
                ))

        # Verify tool calls if present
        if tool_calls:
            for tc in tool_calls:
                tc_findings = self.verify_tool_call(
                    step_id=step_id,
                    tool_name=tc.get("name", "unknown"),
                    args=tc.get("args", {}),
                    result=tc.get("result", ""),
                )
                all_findings.extend(tc_findings)

        return all_findings

    def register_tool(self, name: str, fn: Callable) -> None:
        """Register a tool for re-execution verification."""
        self.tool_verifier.register_tool(name, fn)

    def _verify_text(self, step_id: int, text: str) -> list[Finding]:
        """Verify plain text output."""
        deps = self.provenance.auto_detect_dependencies(step_id, text)
        findings = self.guard.step(output=text, step_id=step_id)

        for f in findings:
            node = self.provenance.record(step_id, f.claim, f, depends_on=deps)
            if node.tainted:
                findings.append(Finding(
                    step_id=step_id,
                    claim=f.claim,
                    verdict=Verdict.FLAG_FOR_REVIEW,
                    level=VerificationLevel.RULE_BASED,
                    explanation=f"TAINTED: {node.tainted_reason}",
                ))

        return findings

    def _verify_dict_event(self, step_id: int, event: dict) -> list[Finding]:
        """Verify a structured event dict."""
        event_type = event.get("type", "text")

        if event_type == "tool_call":
            return self.verify_tool_call(
                step_id=step_id,
                tool_name=event.get("tool", "unknown"),
                args=event.get("args", {}),
                result=event.get("result", ""),
                claimed_output=event.get("claimed_output"),
            )
        elif event_type == "generation":
            return self.verify_generation(
                step_id=step_id,
                prompt=event.get("prompt", ""),
                response_text=event.get("text", ""),
                tool_calls=event.get("tool_calls"),
            )
        else:
            text = event.get("text", event.get("output", json.dumps(event)))
            return self._verify_text(step_id, text)

    def _extract_text_from_event(self, event: Any) -> str:
        """Extract text from an ADK event object."""
        # Try common ADK event attributes
        for attr in ["text", "content", "output", "message", "result"]:
            if hasattr(event, attr):
                val = getattr(event, attr)
                if isinstance(val, str):
                    return val
        return str(event)

    @property
    def findings(self) -> list[Finding]:
        return self.guard.findings

    @property
    def summary(self) -> dict:
        return {
            "verification": self.guard.summary,
            "provenance": self.provenance.summary(),
        }

    def report(self, format: str = "terminal", path: str | None = None) -> str | dict:
        return self.guard.report(format=format, path=path)


class GeminiGuard:
    """Direct Gemini API wrapper with loop-guard verification.

    For users who call Gemini directly (not through ADK).
    Wraps google.genai with automatic verification.

    Usage:
        guard = GeminiGuard(api_key="...")
        result = guard.generate("Compute the mean of [1,2,3,4,5]")
        # result.text has the response
        # result.findings has loop-guard findings
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        config: dict | None = None,
    ) -> None:
        self.model_name = model
        self.adk_guard = ADKGuard(config=config)
        self._step_counter = 0

        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "google-genai is required: pip install google-genai"
            ) from exc

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        tools: list | None = None,
    ) -> GeminiVerifiedResponse:
        """Generate a response and verify it with loop-guard."""
        from google.genai import types

        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )

        step_id = self._step_counter
        self._step_counter += 1

        # Extract text and tool calls
        response_text = response.text if response.text else ""

        # Verify through ADK guard
        findings = self.adk_guard.verify_generation(
            step_id=step_id,
            prompt=prompt,
            response_text=response_text,
        )

        return GeminiVerifiedResponse(
            text=response_text,
            raw_response=response,
            findings=findings,
            step_id=step_id,
        )

    def multi_step(
        self,
        prompts: list[str],
        system_instruction: str | None = None,
    ) -> list[GeminiVerifiedResponse]:
        """Run multiple prompts sequentially with provenance tracking.

        Each prompt can reference results from previous steps.
        loop-guard tracks dependencies and taints downstream claims.
        """
        responses = []
        context = ""

        for i, prompt in enumerate(prompts):
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            result = self.generate(full_prompt, system_instruction)
            responses.append(result)
            context += f"\n\nStep {i} result: {result.text[:500]}"

        return responses

    @property
    def findings(self) -> list[Finding]:
        return self.adk_guard.findings

    @property
    def summary(self) -> dict:
        return self.adk_guard.summary

    def report(self, format: str = "terminal", path: str | None = None) -> str | dict:
        return self.adk_guard.report(format=format, path=path)


class GeminiVerifiedResponse:
    """A Gemini response with loop-guard verification attached."""

    def __init__(
        self,
        text: str,
        raw_response: Any,
        findings: list[Finding],
        step_id: int,
    ) -> None:
        self.text = text
        self.raw_response = raw_response
        self.findings = findings
        self.step_id = step_id

    @property
    def has_issues(self) -> bool:
        return any(
            f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION)
            for f in self.findings
        )

    @property
    def issues(self) -> list[Finding]:
        return [
            f for f in self.findings
            if f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION, Verdict.FLAG_FOR_REVIEW)
        ]

    def __repr__(self) -> str:
        issues = len(self.issues)
        return f"<GeminiVerifiedResponse step={self.step_id} issues={issues} text={self.text[:50]}...>"
