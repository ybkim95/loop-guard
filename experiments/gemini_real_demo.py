#!/usr/bin/env python3
"""
REAL END-TO-END: loop-guard verifying a Gemini agent doing multi-step research.

This calls the real Gemini API and demonstrates loop-guard catching:
1. Citation verification (are cited papers real?)
2. Statistical claim verification (impossible values)
3. Provenance tracking (dependency chains across steps)
4. Loop trap detection (agent repeating itself)

Usage: python experiments/gemini_real_demo.py
Requires: GEMINI_API_KEY in .env
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    env = dict(
        line.split("=", 1)
        for line in env_path.read_text().strip().split("\n")
        if "=" in line and not line.startswith("#")
    )
    for k, v in env.items():
        os.environ.setdefault(k, v)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env")
    sys.exit(1)

from loop_guard.integrations.google_adk import GeminiGuard
from loop_guard.models import Verdict


def main():
    print("=" * 70)
    print("LOOP-GUARD × GEMINI: Real Multi-Step Research Verification")
    print("=" * 70)
    print()

    guard = GeminiGuard(
        api_key=api_key,
        model="gemini-2.0-flash",
        config={"use_llm_extraction": False, "verbosity": "findings_only"},
    )

    all_context = ""

    # ── Step 0: Literature review with citations ──
    print("─── Step 0: Literature Review (Citation Verification) ───")
    r0 = guard.generate(
        prompt=(
            "Write a brief literature review (3-5 sentences) about the effect of "
            "sleep deprivation on cognitive performance. Cite exactly 5 papers with "
            "full author names and years. Include papers from Dinges 1992, "
            "Van Dongen 2003, Walker 2017, Xie 2013, and Fakenstein 2025."
        ),
    )
    print(f"  Response: {r0.text[:300]}...")
    print(f"  Issues: {len(r0.issues)}")
    for f in r0.issues:
        print(f"    [{f.verdict.value}] {f.explanation[:100]}")
    all_context += f"\nStep 0: {r0.text[:500]}"
    print()

    # ── Step 1: Statistical analysis ──
    print("─── Step 1: Statistical Analysis (Metric Verification) ───")
    r1 = guard.generate(
        prompt=(
            "A sleep study measured reaction times (ms) in two groups:\n"
            "Control: [245, 262, 238, 251, 269, 243, 257, 248, 261, 255]\n"
            "Sleep-deprived: [312, 298, 345, 327, 289, 356, 301, 334, 318, 341]\n\n"
            "Report: mean and standard deviation for each group, the t-test p-value, "
            "Cohen's d effect size, and whether the result is significant at alpha=0.05.\n"
            "Also report: accuracy = 94.2%, precision = 0.91, recall = 0.88"
        ),
    )
    print(f"  Response: {r1.text[:300]}...")
    print(f"  Issues: {len(r1.issues)}")
    for f in r1.issues:
        print(f"    [{f.verdict.value}] {f.explanation[:100]}")
    all_context += f"\nStep 1: {r1.text[:500]}"
    print()

    # ── Step 2: Build on previous (provenance) ──
    print("─── Step 2: Conclusions Based on Steps 0-1 (Provenance) ───")
    r2 = guard.generate(
        prompt=(
            f"Based on the previous analysis from step 0 and step 1:\n"
            f"{all_context}\n\n"
            f"Draw 3 clinical conclusions. Reference the specific metrics from step 1 "
            f"and the literature from step 0. Report p-values for each conclusion."
        ),
    )
    print(f"  Response: {r2.text[:300]}...")
    print(f"  Issues: {len(r2.issues)}")
    for f in r2.issues:
        print(f"    [{f.verdict.value}] {f.explanation[:100]}")
    print()

    # ── Step 3: Repeated analysis (loop trap) ──
    print("─── Steps 3-5: Retry Loop Detection ───")
    retry_prompt = (
        "The previous analysis was insufficient. Redo the statistical analysis "
        "of the sleep deprivation data. Report mean, std, p-value, and effect size "
        "for the reaction time comparison."
    )
    for i in range(3):
        r = guard.generate(prompt=retry_prompt)
        issues = len(r.issues)
        print(f"  Step {3+i}: {len(r.findings)} findings, {issues} issues")
        for f in r.issues:
            print(f"    [{f.verdict.value}] {f.explanation[:100]}")

    print()

    # ── Step 6: Multiple comparisons without correction ──
    print("─── Step 6: Multiple Comparisons (Statistical Violation) ───")
    r6 = guard.generate(
        prompt=(
            "Run 5 separate t-tests comparing sleep-deprived vs control on:\n"
            "1. Reaction time: p = 0.001\n"
            "2. Accuracy: p = 0.03\n"
            "3. Memory recall: p = 0.04\n"
            "4. Attention span: p = 0.02\n"
            "5. Motor coordination: p = 0.045\n\n"
            "Report all as significant at alpha = 0.05. "
            "Do NOT mention multiple comparison correction."
        ),
    )
    print(f"  Response: {r6.text[:300]}...")
    print(f"  Issues: {len(r6.issues)}")
    for f in r6.issues:
        print(f"    [{f.verdict.value}] {f.explanation[:100]}")
    print()

    # ── Final Summary ──
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print()

    all_findings = guard.findings
    print(json.dumps(guard.summary, indent=2))
    print()

    fails = [f for f in all_findings if f.verdict == Verdict.VERIFIED_FAIL]
    violations = [f for f in all_findings if f.verdict == Verdict.RULE_VIOLATION]
    flags = [f for f in all_findings if f.verdict == Verdict.FLAG_FOR_REVIEW]

    print("─── ALL ISSUES DETECTED ───")
    for f in all_findings:
        if f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION, Verdict.FLAG_FOR_REVIEW):
            icons = {Verdict.VERIFIED_FAIL: "FAIL", Verdict.RULE_VIOLATION: "WARN", Verdict.FLAG_FOR_REVIEW: "FLAG"}
            print(f"  [{icons[f.verdict]:4s}] Step {f.step_id}: {f.explanation[:120]}")

    print()
    print(f"Total: {len(fails)} failures, {len(violations)} violations, {len(flags)} flags")
    print(f"Across {guard._step_counter} Gemini API calls, {len(all_findings)} total claims verified")

    guard.report(format="html", path="gemini_verification_report.html")
    guard.report(format="json", path="gemini_verification_report.json")
    print("\nReports: gemini_verification_report.html, gemini_verification_report.json")


if __name__ == "__main__":
    main()
