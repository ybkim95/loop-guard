#!/usr/bin/env python3
"""
REAL OpenAI Agent Demo with loop-guard verification.

Runs a multi-step data analysis agent using GPT-4o-mini, with loop-guard
verifying each step in real-time. Demonstrates:

1. Step 1: GPT computes basic statistics on a dataset
2. Step 2: GPT computes derived statistics based on step 1
3. Step 3: GPT draws conclusions (may hallucinate correlations/trends)

loop-guard catches:
- Metric inconsistencies between steps
- Hallucinated statistics (numbers that don't match re-computation)
- Suspicious claims in conclusions

Requirements:
    - openai Python SDK (v2.x+)
    - OPENAI_API_KEY in /u/ybkim95/loop-guard/.env

Usage:
    python experiments/openai_real_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_guard.integrations.openai_agents import OpenAIGuard
from loop_guard.models import Verdict

# ── Load .env ────────────────────────────────────────────────────────

env_path = Path("/u/ybkim95/loop-guard/.env")
if env_path.exists():
    env = dict(
        line.split("=", 1)
        for line in env_path.read_text().strip().split("\n")
        if "=" in line and not line.startswith("#")
    )
    for k, v in env.items():
        os.environ.setdefault(k.strip(), v.strip())

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env or environment.")
    print(f"  Expected .env at: {env_path}")
    sys.exit(1)

# ── Import OpenAI ────────────────────────────────────────────────────

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

# ── Dataset ──────────────────────────────────────────────────────────

# A small but real dataset for the agent to analyze
DATASET = {
    "description": "Monthly sales data for 3 product lines over 12 months",
    "products": ["Widget A", "Widget B", "Widget C"],
    "months": [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ],
    "sales": {
        "Widget A": [120, 135, 148, 142, 155, 168, 172, 180, 165, 158, 190, 210],
        "Widget B": [85, 78, 92, 88, 95, 102, 98, 105, 110, 115, 108, 125],
        "Widget C": [200, 210, 195, 220, 215, 230, 245, 238, 250, 260, 255, 280],
    },
    "costs": {
        "Widget A": [60, 65, 70, 68, 72, 78, 80, 84, 77, 74, 88, 95],
        "Widget B": [50, 48, 55, 52, 56, 60, 58, 62, 65, 68, 64, 74],
        "Widget C": [120, 125, 118, 130, 128, 135, 142, 138, 145, 150, 148, 160],
    },
}

DATASET_STR = json.dumps(DATASET, indent=2)


# ── Agent Steps ──────────────────────────────────────────────────────

def run_step(client: OpenAI, messages: list[dict], step_desc: str) -> dict:
    """Run one OpenAI chat completion step."""
    print(f"  Calling GPT-4o-mini... ", end="", flush=True)
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )
    elapsed = time.time() - start
    print(f"done ({elapsed:.1f}s)")
    return response


def main():
    print("=" * 70)
    print("LOOP-GUARD + OPENAI REAL DEMO")
    print("Multi-step data analysis with real-time verification")
    print("=" * 70)
    print()

    # ── Initialize ────────────────────────────────────────────────────

    client = OpenAI(api_key=api_key)
    guard = OpenAIGuard(
        api_key=api_key,
        config={
            "use_llm_extraction": False,
            "verbosity": "findings_only",
        },
    )

    conversation: list[dict] = []
    all_findings = []

    # ── STEP 1: Basic statistics ──────────────────────────────────────

    print("--- Step 1: Compute basic statistics ---")
    print()

    conversation.append({
        "role": "system",
        "content": (
            "You are a data analyst. Compute statistics precisely. "
            "Always show your numbers clearly with labels like 'mean: X', 'total: Y'. "
            "Do NOT round aggressively - show at least 2 decimal places."
        ),
    })
    conversation.append({
        "role": "user",
        "content": (
            f"Here is a sales dataset:\n\n{DATASET_STR}\n\n"
            "For each product, compute:\n"
            "1. Total annual sales\n"
            "2. Mean monthly sales\n"
            "3. Min and max monthly sales\n"
            "4. Total annual cost\n"
            "5. Total annual profit (sales - cost)\n"
            "Be precise with all numbers."
        ),
    })

    response1 = run_step(client, conversation, "basic stats")
    text1 = response1.choices[0].message.content
    conversation.append({"role": "assistant", "content": text1})

    print(f"  Response length: {len(text1)} chars")
    print()

    findings1 = guard.verify_response(step_id=0, response=response1)
    all_findings.extend(findings1)
    _print_findings(0, findings1)

    # ── STEP 2: Derived statistics ────────────────────────────────────

    print("--- Step 2: Compute derived statistics ---")
    print()

    conversation.append({
        "role": "user",
        "content": (
            "Now using the statistics you just computed, calculate:\n"
            "1. Profit margin (%) for each product = (profit / sales) * 100\n"
            "2. Month-over-month growth rate for each product (average % change)\n"
            "3. Which product has the highest profit margin?\n"
            "4. Which product is growing fastest?\n"
            "5. The correlation between Widget A and Widget C monthly sales "
            "(compute Pearson correlation coefficient)\n"
            "Show all calculations clearly."
        ),
    })

    response2 = run_step(client, conversation, "derived stats")
    text2 = response2.choices[0].message.content
    conversation.append({"role": "assistant", "content": text2})

    print(f"  Response length: {len(text2)} chars")
    print()

    findings2 = guard.verify_response(step_id=1, response=response2)
    all_findings.extend(findings2)
    _print_findings(1, findings2)

    # ── STEP 3: Conclusions (hallucination-prone) ─────────────────────

    print("--- Step 3: Draw conclusions and predictions ---")
    print()

    conversation.append({
        "role": "user",
        "content": (
            "Based on your analysis, provide:\n"
            "1. A summary of key findings with specific numbers\n"
            "2. Predict next month's (January) sales for each product\n"
            "3. Identify any seasonal patterns\n"
            "4. Recommend which product to invest more in and why\n"
            "5. State the statistical confidence of your predictions\n"
            "Be specific - cite the exact numbers from your earlier analysis."
        ),
    })

    response3 = run_step(client, conversation, "conclusions")
    text3 = response3.choices[0].message.content
    conversation.append({"role": "assistant", "content": text3})

    print(f"  Response length: {len(text3)} chars")
    print()

    findings3 = guard.verify_response(step_id=2, response=response3)
    all_findings.extend(findings3)
    _print_findings(2, findings3)

    # ── Verify tool call example (simulated) ──────────────────────────

    print("--- Bonus: Verify a simulated tool call ---")
    print()

    # Simulate what would happen if the agent used a calculator tool
    # but claimed wrong output
    tool_findings = guard.verify_tool_call(
        step_id=3,
        tool_name="calculator",
        args={"expression": "sum([120,135,148,142,155,168,172,180,165,158,190,210])"},
        result="1943",
        claimed_output="The total annual sales for Widget A is 1,950 units.",
    )
    all_findings.extend(tool_findings)
    _print_findings(3, tool_findings)

    # ── Final Summary ─────────────────────────────────────────────────

    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print()

    summary = guard.summary
    print(json.dumps(summary, indent=2, default=str))
    print()

    # Count by verdict
    verdict_counts: dict[str, int] = {}
    for f in all_findings:
        key = f.verdict.value
        verdict_counts[key] = verdict_counts.get(key, 0) + 1

    print(f"Total findings: {len(all_findings)}")
    for verdict, count in sorted(verdict_counts.items()):
        print(f"  {verdict}: {count}")
    print()

    # Generate reports
    report_dir = Path(__file__).resolve().parent.parent
    guard.report(format="html", path=str(report_dir / "openai_demo_report.html"))
    guard.report(format="json", path=str(report_dir / "openai_demo_report.json"))
    print(f"Reports saved:")
    print(f"  HTML: {report_dir / 'openai_demo_report.html'}")
    print(f"  JSON: {report_dir / 'openai_demo_report.json'}")

    # ── Print the actual GPT responses for inspection ─────────────────

    print()
    print("=" * 70)
    print("GPT RESPONSES (for manual inspection)")
    print("=" * 70)

    for i, (label, text) in enumerate([
        ("Step 1: Basic Statistics", text1),
        ("Step 2: Derived Statistics", text2),
        ("Step 3: Conclusions", text3),
    ]):
        print(f"\n--- {label} ---")
        # Truncate very long responses for terminal readability
        if len(text) > 2000:
            print(text[:2000])
            print(f"\n  ... (truncated, {len(text)} chars total)")
        else:
            print(text)

    print()
    print("Done.")


def _print_findings(step_id: int, findings: list) -> None:
    """Print findings for a step in a readable format."""
    non_pass = [
        f for f in findings
        if f.verdict not in (Verdict.VERIFIED_PASS, Verdict.SKIPPED)
    ]
    if non_pass:
        for f in non_pass:
            icon = {
                Verdict.VERIFIED_FAIL: "FAIL",
                Verdict.RULE_VIOLATION: "WARN",
                Verdict.FLAG_FOR_REVIEW: "FLAG",
            }.get(f.verdict, "????")
            print(f"  [{icon}] {f.explanation}")
            if f.expected and f.actual:
                print(f"         Expected: {f.expected}")
                print(f"         Actual:   {f.actual}")
    else:
        print(f"  [OK] No issues detected in step {step_id}")
    print()


if __name__ == "__main__":
    main()
