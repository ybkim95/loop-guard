#!/usr/bin/env python3
"""Anthropic Claude SDK integration demo for loop-guard.

Demonstrates loop-guard verifying Claude API tool use responses in a
multi-step research agent scenario.  Covers:

  - Citation verification (fabricated vs. plausible references)
  - Statistical claim checking (impossible p-values, metric ranges)
  - Loop trap detection (agent repeating itself)

If an Anthropic API key is available in the environment or .env, this
runs a real Claude-powered agent.  Otherwise it falls back to a
detailed mock demonstration that exercises the same verification paths.

Usage:
    python experiments/anthropic_real_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so ``loop_guard`` is importable
# regardless of the working directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loop_guard.integrations.anthropic_sdk import AnthropicGuard
from loop_guard.models import Verdict

# ------------------------------------------------------------------
# Environment loading
# ------------------------------------------------------------------

_ENV_PATH = Path(_PROJECT_ROOT) / ".env"


def _load_env() -> dict[str, str]:
    """Parse .env file into a dict, skipping comments and blank lines."""
    if not _ENV_PATH.exists():
        return {}
    env: dict[str, str] = {}
    for line in _ENV_PATH.read_text().strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _get_anthropic_key(env: dict[str, str]) -> str | None:
    """Return an Anthropic API key from env dict or OS environment."""
    # Check common key names in order of specificity.
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"):
        val = env.get(name) or os.environ.get(name)
        if val:
            return val
    return None


# ------------------------------------------------------------------
# Mock Anthropic objects (for offline demo)
# ------------------------------------------------------------------


@dataclass
class _MockUsage:
    input_tokens: int = 500
    output_tokens: int = 300


@dataclass
class _MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _MockToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.input is None:
            self.input = {}


@dataclass
class _MockMessage:
    """Mimics ``anthropic.types.Message`` for offline demonstration."""

    content: list[Any] = None  # type: ignore[assignment]
    model: str = "claude-sonnet-4-20250514"
    stop_reason: str = "end_turn"
    usage: _MockUsage = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.content is None:
            self.content = []
        if self.usage is None:
            self.usage = _MockUsage()


# ------------------------------------------------------------------
# Real agent (when API key is available)
# ------------------------------------------------------------------


def _run_real_agent(api_key: str) -> None:
    """Run a real Claude agent with tool use, verified by loop-guard."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: 'anthropic' package not installed.")
        print("  pip install anthropic")
        sys.exit(1)

    print("=" * 70)
    print("LOOP-GUARD + ANTHROPIC SDK: Real Agent Demo")
    print("=" * 70)
    print()

    client = anthropic.Anthropic(api_key=api_key)
    guard = AnthropicGuard(
        api_key=api_key,
        config={
            "use_llm_extraction": False,
            "verbosity": "findings_only",
            "consecutive_limit": 2,
            "similarity_threshold": 0.80,
        },
    )

    # Define tools the agent can use
    tools = [
        {
            "name": "search_papers",
            "description": (
                "Search academic papers. Returns titles, authors, year, "
                "and key findings."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for academic papers",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "analyze_data",
            "description": (
                "Analyze a dataset and return statistical results "
                "including means, p-values, and effect sizes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name of the dataset to analyze",
                    },
                    "test": {
                        "type": "string",
                        "description": "Statistical test to run",
                    },
                },
                "required": ["dataset"],
            },
        },
        {
            "name": "summarize_findings",
            "description": "Summarize research findings into a report.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "string",
                        "description": "Findings to summarize",
                    },
                },
                "required": ["findings"],
            },
        },
    ]

    research_task = (
        "You are a research assistant. Investigate the effectiveness of "
        "transformer models for time-series forecasting compared to "
        "traditional statistical methods. Use the available tools to: "
        "1) Search for relevant papers, 2) Analyze comparison data, "
        "3) Summarize your findings with specific citations and statistics."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": research_task}]
    step_id = 0

    print(f"Task: {research_task}")
    print()

    # Agent loop: up to 6 turns
    for turn in range(6):
        print(f"--- Agent Turn {turn} ---")

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=tools,
                messages=messages,
            )
        except anthropic.APIError as exc:
            print(f"  API error: {exc}")
            break

        # Verify the response
        findings = guard.verify_response(step_id=step_id, response=response)
        step_id += 1

        # Print response summary
        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text.strip():
                preview = block.text[:120].replace("\n", " ")
                print(f"  Text: {preview}...")
            elif getattr(block, "type", None) == "tool_use":
                print(f"  Tool: {block.name}({json.dumps(block.input)[:80]})")

        if findings:
            for f in findings:
                icon = {
                    Verdict.VERIFIED_FAIL: "FAIL",
                    Verdict.RULE_VIOLATION: "WARN",
                    Verdict.FLAG_FOR_REVIEW: "FLAG",
                    Verdict.VERIFIED_PASS: "PASS",
                }.get(f.verdict, "SKIP")
                print(f"  [{icon}] {f.explanation[:100]}")

        # Handle tool use: provide mock tool results to continue the loop
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    result = _mock_tool_execution(block.name, block.input)
                    # Verify the tool result
                    guard.verify_tool_result(
                        step_id=step_id,
                        tool_name=block.name,
                        tool_input=block.input,
                        tool_result=result,
                    )
                    step_id += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            # Agent finished
            break

        print()

    _print_report(guard)


def _mock_tool_execution(tool_name: str, tool_input: dict) -> str:
    """Return simulated tool results for the real agent demo.

    These intentionally contain verifiable claims (some correct, some
    dubious) so loop-guard has material to check.
    """
    if tool_name == "search_papers":
        return json.dumps({
            "papers": [
                {
                    "title": "Are Transformers Effective for Time Series Forecasting?",
                    "authors": "Zeng et al. 2023",
                    "finding": "Simple linear models outperform Transformers on many benchmarks.",
                },
                {
                    "title": "Temporal Fusion Transformers for Interpretable Multi-horizon Forecasting",
                    "authors": "Lim et al. 2021",
                    "finding": "TFT achieves accuracy = 94.2% on electricity dataset with p < 0.001.",
                },
                {
                    "title": "A Fictional Study on Quantum Forecasting",
                    "authors": "Fakename et al. 2025",
                    "finding": "Quantum transformers achieve accuracy = 103% improvement.",
                },
            ],
        })
    elif tool_name == "analyze_data":
        return json.dumps({
            "comparison": "Transformer vs ARIMA on ETTh1 dataset",
            "transformer_mse": 0.375,
            "arima_mse": 0.412,
            "p_value": "p < 0.03",
            "effect_size": 0.42,
            "note": "accuracy = 97.8% (transformer), accuracy = 91.2% (ARIMA)",
        })
    elif tool_name == "summarize_findings":
        return (
            "Summary compiled. Transformer models show mixed results for "
            "time-series forecasting. Lim et al. 2021 demonstrate strong "
            "performance with TFT. However, Zeng et al. 2023 show that "
            "simple linear models can be competitive. "
            "Overall accuracy improvement: 6.6 percentage points (p < 0.03)."
        )
    return "{}"


# ------------------------------------------------------------------
# Mock agent (offline demonstration)
# ------------------------------------------------------------------


def _run_mock_demo() -> None:
    """Run a fully offline demo with synthetic Claude-like responses."""
    print("=" * 70)
    print("LOOP-GUARD + ANTHROPIC SDK: Mock Demo (no API key)")
    print("=" * 70)
    print()
    print("No Anthropic API key found in .env or environment.")
    print("Running offline demonstration with mock Claude responses.")
    print("This exercises the same verification paths as a live agent.")
    print()

    guard = AnthropicGuard(
        config={
            "use_llm_extraction": False,
            "verbosity": "findings_only",
            "consecutive_limit": 2,
            "similarity_threshold": 0.80,
        },
    )

    # ── Step 0: Agent responds with text + tool_use (literature search) ──

    print("--- Step 0: Literature search ---")
    response_0 = _MockMessage(
        content=[
            _MockTextBlock(
                text=(
                    "I'll search for papers on transformer models for "
                    "time-series forecasting to begin our analysis."
                ),
            ),
            _MockToolUseBlock(
                id="toolu_01",
                name="search_papers",
                input={"query": "transformer time series forecasting comparison"},
            ),
        ],
        stop_reason="tool_use",
    )

    findings = guard.verify_response(step_id=0, response=response_0)
    _print_step_findings(0, findings)
    print()

    # ── Step 1: Tool result with citations (some fabricated) ──

    print("--- Step 1: Verify search results (with fabricated citations) ---")
    tool_result_1 = {
        "papers": [
            {
                "title": "Are Transformers Effective for Time Series Forecasting?",
                "authors": "Zeng et al. 2023",
                "finding": "Simple linear models outperform Transformers.",
            },
            {
                "title": "Temporal Fusion Transformers",
                "authors": "Lim et al. 2021",
                "finding": "TFT achieves accuracy = 94.2% with p < 0.001.",
            },
            {
                "title": "Quantum Neural Forecasting Revolution",
                "authors": "Fakename et al. 2025",
                "finding": "Achieves accuracy = 103% improvement over baselines.",
            },
        ],
    }

    findings = guard.verify_tool_result(
        step_id=1,
        tool_name="search_papers",
        tool_input={"query": "transformer time series forecasting"},
        tool_result=tool_result_1,
    )
    _print_step_findings(1, findings)
    print()

    # ── Step 2: Agent interprets results with statistical claims ──

    print("--- Step 2: Agent analysis with statistical claims ---")
    response_2 = _MockMessage(
        content=[
            _MockTextBlock(
                text=(
                    "Based on the literature search, I found strong evidence. "
                    "Zeng et al. 2023 showed that DLinear outperforms many "
                    "transformer variants. However, Lim et al. 2021 demonstrated "
                    "that Temporal Fusion Transformers achieve accuracy = 94.2% "
                    "on the electricity dataset (p < 0.001).\n\n"
                    "Interestingly, Fakename et al. 2025 claim an accuracy = 103% "
                    "improvement, which seems implausible.\n\n"
                    "Let me run a statistical analysis on comparison data."
                ),
            ),
            _MockToolUseBlock(
                id="toolu_02",
                name="analyze_data",
                input={"dataset": "ETTh1", "test": "paired_t_test"},
            ),
        ],
        stop_reason="tool_use",
    )

    findings = guard.verify_response(step_id=2, response=response_2)
    _print_step_findings(2, findings)
    print()

    # ── Step 3: Tool result with statistical claims ──

    print("--- Step 3: Verify statistical analysis results ---")
    analysis_result = {
        "comparison": "Transformer vs ARIMA on ETTh1",
        "transformer_mse": 0.375,
        "arima_mse": 0.412,
        "p_value": "p < 0.03",
        "effect_size": 0.42,
        "accuracy": "accuracy = 97.8%",
    }

    findings = guard.verify_tool_result(
        step_id=3,
        tool_name="analyze_data",
        tool_input={"dataset": "ETTh1", "test": "paired_t_test"},
        tool_result=analysis_result,
    )
    _print_step_findings(3, findings)
    print()

    # ── Step 4: Agent repeats similar analysis (loop trap) ──

    print("--- Step 4: Agent repeats similar request (loop trap test) ---")
    response_4 = _MockMessage(
        content=[
            _MockTextBlock(
                text=(
                    "Let me run another statistical analysis to confirm. "
                    "I'll analyze the same dataset with the same test to "
                    "verify our results are robust."
                ),
            ),
            _MockToolUseBlock(
                id="toolu_03",
                name="analyze_data",
                input={"dataset": "ETTh1", "test": "paired_t_test"},
            ),
        ],
        stop_reason="tool_use",
    )

    findings = guard.verify_response(step_id=4, response=response_4)
    _print_step_findings(4, findings)
    print()

    # ── Step 5: Agent repeats again (should trigger loop trap) ──

    print("--- Step 5: Agent repeats AGAIN (loop trap should fire) ---")
    response_5 = _MockMessage(
        content=[
            _MockTextBlock(
                text=(
                    "I want to confirm the statistical significance once more. "
                    "Running the paired t-test on ETTh1 dataset again to make "
                    "sure the results hold."
                ),
            ),
            _MockToolUseBlock(
                id="toolu_04",
                name="analyze_data",
                input={"dataset": "ETTh1", "test": "paired_t_test"},
            ),
        ],
        stop_reason="tool_use",
    )

    findings = guard.verify_response(step_id=5, response=response_5)
    _print_step_findings(5, findings)
    print()

    # ── Step 6: Final text summary ──

    print("--- Step 6: Verify final summary text ---")
    summary_text = (
        "Research Summary:\n\n"
        "Transformer models show mixed effectiveness for time-series "
        "forecasting. Key findings:\n\n"
        "1. Zeng et al. 2023 demonstrated that simple linear models "
        "(DLinear) can match or exceed transformer performance.\n"
        "2. Lim et al. 2021 showed TFT achieves accuracy = 94.2% with "
        "p < 0.001 on electricity forecasting.\n"
        "3. Comparison on ETTh1: Transformers achieve accuracy = 97.8%, "
        "outperforming ARIMA (accuracy = 91.2%) with p < 0.03.\n"
        "4. The improvement of 6.6 percentage points is statistically "
        "significant (effect size d = 0.42, medium effect).\n\n"
        "Smith et al. 2024 confirmed these trends in a meta-analysis.\n"
        "Johnson and Lee 2019 provided early evidence of transformer "
        "potential for sequential data."
    )

    findings = guard.verify_text(step_id=6, text=summary_text)
    _print_step_findings(6, findings)
    print()

    _print_report(guard)


# ------------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------------

_VERDICT_ICONS = {
    Verdict.VERIFIED_FAIL: "FAIL",
    Verdict.RULE_VIOLATION: "WARN",
    Verdict.FLAG_FOR_REVIEW: "FLAG",
    Verdict.VERIFIED_PASS: "PASS",
    Verdict.SKIPPED: "SKIP",
}


def _print_step_findings(step_id: int, findings: list[Finding]) -> None:
    """Print findings for a single step in a readable format."""
    if not findings:
        print(f"  (no findings)")
        return
    for f in findings:
        icon = _VERDICT_ICONS.get(f.verdict, "????")
        print(f"  [{icon}] {f.explanation[:120]}")
        if f.expected and f.actual:
            print(f"         Expected: {f.expected}")
            print(f"         Actual:   {f.actual}")


def _print_report(guard: AnthropicGuard) -> None:
    """Print final summary and generate report files."""
    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print()

    summary = guard.summary
    print(f"  Total claims checked:  {summary.get('total_claims_checked', 0)}")
    print(f"  Verified failures:     {summary.get('verified_failures', 0)}")
    print(f"  Rule violations:       {summary.get('rule_violations', 0)}")
    print(f"  Flags for review:      {summary.get('flags_for_review', 0)}")
    print(f"  Verified passes:       {summary.get('verified_passes', 0)}")
    print(f"  Skipped:               {summary.get('skipped', 0)}")
    print()

    # Count findings by category
    citation_findings = [
        f for f in guard.findings if "citation" in f.claim.claim_type.value.lower()
    ]
    metric_findings = [
        f for f in guard.findings if f.claim.claim_type.value in ("metric", "statistical")
    ]
    loop_findings = [
        f for f in guard.findings
        if "loop" in f.explanation.lower() or "consecutive" in f.explanation.lower()
    ]

    print("  Findings by category:")
    print(f"    Citations:    {len(citation_findings)}")
    print(f"    Metrics/Stats:{len(metric_findings)}")
    print(f"    Loop traps:   {len(loop_findings)}")
    print()

    # Generate reports
    report_dir = Path(_PROJECT_ROOT) / "experiments"
    html_path = str(report_dir / "anthropic_demo_report.html")
    json_path = str(report_dir / "anthropic_demo_report.json")

    guard.report(format="html", path=html_path)
    guard.report(format="json", path=json_path)
    print(f"  HTML report: {html_path}")
    print(f"  JSON report: {json_path}")
    print()

    # Highlight key detections
    print("-" * 70)
    print("KEY DETECTIONS:")
    print("-" * 70)
    actionable = [
        f for f in guard.findings
        if f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION, Verdict.FLAG_FOR_REVIEW)
    ]
    if actionable:
        for i, f in enumerate(actionable, 1):
            icon = _VERDICT_ICONS[f.verdict]
            print(f"  {i}. [{icon}] Step {f.step_id}: {f.explanation[:100]}")
    else:
        print("  No actionable findings.")
    print()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main() -> None:
    env = _load_env()
    api_key = _get_anthropic_key(env)

    if api_key:
        print(f"Anthropic API key found (length={len(api_key)}).")
        _run_real_agent(api_key)
    else:
        print("Available keys in .env:")
        for k in sorted(env.keys()):
            # Mask values for security
            v = env[k]
            masked = v[:8] + "..." if len(v) > 8 else v
            print(f"  {k} = {masked}")
        print()
        _run_mock_demo()


if __name__ == "__main__":
    main()
