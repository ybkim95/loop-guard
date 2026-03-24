"""Anthropic Claude SDK integration for loop-guard.

Wraps Claude API tool use interactions with deterministic verification.

Usage:
    from loop_guard.integrations.anthropic_sdk import AnthropicGuard

    guard = AnthropicGuard(api_key="sk-ant-...")

    response = client.messages.create(...)
    findings = guard.verify_response(step_id=0, response=response)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from loop_guard.guard import LoopGuard
from loop_guard.models import Finding

logger = logging.getLogger(__name__)


class AnthropicGuard:
    """Deterministic verification layer for Anthropic Claude API responses.

    Intercepts Claude ``Message`` objects (from ``anthropic`` SDK v0.76+),
    extracts text and ``tool_use`` content blocks, and feeds them through
    the :class:`~loop_guard.guard.LoopGuard` verification pipeline.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Only required if you plan to use the guard's
        own client for auxiliary calls (e.g. LLM-assisted extraction).
        Passed through to :class:`LoopGuard` config as
        ``anthropic_api_key``.
    config:
        Optional :class:`LoopGuard` configuration dict.  Merged with
        defaults; caller-supplied keys take precedence.
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        merged_config: dict[str, Any] = {
            "verbosity": "findings_only",
        }
        if config:
            merged_config.update(config)
        if api_key:
            merged_config["anthropic_api_key"] = api_key

        self._guard = LoopGuard(config=merged_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_response(
        self,
        step_id: int,
        response: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[Finding]:
        """Verify an Anthropic ``Message`` response.

        Iterates over ``response.content`` blocks, concatenates ``text``
        blocks, and captures ``tool_use`` blocks with their inputs.
        Everything is fed to :pymethod:`LoopGuard.step` as a single
        verification step.

        Parameters
        ----------
        step_id:
            Monotonic step index within the agent loop.
        response:
            An ``anthropic.types.Message`` object returned by
            ``client.messages.create()``.
        metadata:
            Optional extra metadata attached to the step.

        Returns
        -------
        list[Finding]
            Verification findings for this step.
        """
        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []

        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_uses.append({
                    "tool_name": block.name,
                    "tool_id": block.id,
                    "input": block.input,
                })

        combined_text = "\n".join(text_parts)

        # Append a structured summary of tool calls so the extractor can
        # find metric / code-output / citation claims inside tool inputs.
        if tool_uses:
            tool_summary_parts: list[str] = []
            for tu in tool_uses:
                input_str = json.dumps(tu["input"], indent=2, default=str)
                tool_summary_parts.append(
                    f"[tool_use: {tu['tool_name']}]\n{input_str}"
                )
            combined_text += "\n\n" + "\n".join(tool_summary_parts)

        step_metadata = {
            "source": "anthropic_sdk",
            "model": getattr(response, "model", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "usage": _extract_usage(response),
            "tool_uses": tool_uses,
        }
        if metadata:
            step_metadata.update(metadata)

        return self._guard.step(
            output=combined_text,
            step_id=step_id,
            metadata=step_metadata,
        )

    def verify_tool_result(
        self,
        step_id: int,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: Any,
        *,
        claimed_output: str | None = None,
    ) -> list[Finding]:
        """Verify a single tool call and its result.

        Use this when you intercept tool results *before* sending them
        back to Claude, allowing loop-guard to flag suspicious outputs
        early.

        Parameters
        ----------
        step_id:
            Step index within the agent loop.
        tool_name:
            Name of the tool that was called.
        tool_input:
            The input dict passed to the tool.
        tool_result:
            The raw result returned by the tool execution.
        claimed_output:
            Optional text the agent *claimed* the tool would return.
            If provided, loop-guard compares it against ``tool_result``.

        Returns
        -------
        list[Finding]
            Verification findings.
        """
        result_str = (
            tool_result
            if isinstance(tool_result, str)
            else json.dumps(tool_result, indent=2, default=str)
        )

        output_parts = [
            f"Tool call: {tool_name}",
            f"Input: {json.dumps(tool_input, default=str)}",
            f"Result: {result_str}",
        ]
        if claimed_output is not None:
            output_parts.append(f"Agent claimed: {claimed_output}")

        combined = "\n".join(output_parts)

        return self._guard.step(
            output=combined,
            step_id=step_id,
            metadata={
                "source": "anthropic_sdk",
                "verification_type": "tool_result",
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
        )

    def verify_text(
        self,
        step_id: int,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[Finding]:
        """Verify a plain text response (no tool use).

        Convenience method for responses that consist only of text
        content — equivalent to calling :pymethod:`LoopGuard.step`
        directly.

        Parameters
        ----------
        step_id:
            Step index within the agent loop.
        text:
            The raw text content of the response.
        metadata:
            Optional extra metadata.

        Returns
        -------
        list[Finding]
            Verification findings.
        """
        step_metadata: dict[str, Any] = {"source": "anthropic_sdk"}
        if metadata:
            step_metadata.update(metadata)

        return self._guard.step(
            output=text,
            step_id=step_id,
            metadata=step_metadata,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def findings(self) -> list[Finding]:
        """All findings accumulated across verified steps."""
        return self._guard.findings

    @property
    def summary(self) -> dict:
        """Summary statistics of all findings."""
        return self._guard.summary

    def report(
        self,
        format: str = "terminal",
        path: str | None = None,
    ) -> str | dict:
        """Generate a verification report.

        Parameters
        ----------
        format:
            One of ``"terminal"``, ``"json"``, ``"html"``.
        path:
            Output file path for ``json`` / ``html`` formats.

        Returns
        -------
        str | dict
            File path (for json/html) or summary dict (for terminal).
        """
        return self._guard.report(format=format, path=path)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _extract_usage(response: Any) -> dict[str, int]:
    """Safely extract token usage from an Anthropic Message."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
    }
