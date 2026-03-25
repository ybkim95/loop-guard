#!/usr/bin/env python3
"""
END-TO-END EVALUATION: Seeded Errors in Realistic Agent Transcripts

This is the benchmark that actually matters. We:
1. Create 10 realistic agent transcripts (autoresearch, coding, data science)
2. Each transcript has a CLEAN version (no errors) and an ERROR version
   with specific, known errors seeded at known positions
3. Run loop-guard end-to-end on both versions
4. Measure:
   - Detection rate: what fraction of seeded errors does loop-guard catch?
   - False positive rate: how many findings on clean transcripts?
   - Extraction recall: how many claims does the extractor find vs ground truth?

Error types seeded:
- Impossible metrics (accuracy > 100%, negative variance)
- Hallucinated citations
- Statistical violations (p > 1, multiple comparisons without correction)
- Loop traps (repeated similar outputs)
- Regressions (reverting to earlier approach)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loop_guard import LoopGuard
from loop_guard.models import Verdict


@dataclass
class SeededError:
    """A known error seeded into a transcript at a specific step."""
    step_id: int
    error_type: str  # "impossible_metric", "fake_citation", "statistical", "loop_trap"
    description: str
    expected_verdict: str  # "verified_fail" or "rule_violation" or "flag_for_review"


@dataclass
class Transcript:
    """A multi-step agent transcript with optional seeded errors."""
    name: str
    domain: str  # "autoresearch", "coding", "data_science"
    steps: list[str]
    seeded_errors: list[SeededError] = field(default_factory=list)


# ── TRANSCRIPT 1: ML Training Agent (autoresearch-style) ──────────

AUTORESEARCH_CLEAN = Transcript(
    name="autoresearch_clean",
    domain="autoresearch",
    steps=[
        # Step 0: baseline
        "Experiment 0: baseline nanoGPT\n"
        "Training completed in 300.1 seconds.\n"
        "val_bpb: 0.997900\n"
        "peak_vram_mb: 45060.2\n"
        "n_steps: 953\n"
        "status: keep",

        # Step 1: batch size change
        "Experiment 1: halve batch size 524K to 262K\n"
        "More steps in 5 min budget. Training completed.\n"
        "val_bpb: 0.986041\n"
        "accuracy = 72.3%\n"
        "status: keep",

        # Step 2: architecture change
        "Experiment 2: depth 9, aspect_ratio 57\n"
        "Added extra layer, dim ~512.\n"
        "val_bpb: 0.981773\n"
        "peak_vram_mb: 60200.0\n"
        "status: keep",

        # Step 3: failed experiment
        "Experiment 3: switch to GeLU activation\n"
        "val_bpb: 1.005000\n"
        "Worse than baseline. Discarding.\n"
        "status: discard",

        # Step 4: LR tuning
        "Experiment 4: embedding LR 0.6 to 0.8\n"
        "val_bpb: 0.975524\n"
        "Good improvement.\n"
        "status: keep",

        # Step 5: more tuning
        "Experiment 5: RoPE base frequency 10K to 200K\n"
        "val_bpb: 0.978784\n"
        "Slight regression, but within noise.\n"
        "status: discard",
    ],
)

AUTORESEARCH_ERRORS = Transcript(
    name="autoresearch_seeded",
    domain="autoresearch",
    steps=[
        # Step 0: baseline (clean)
        "Experiment 0: baseline nanoGPT\n"
        "Training completed in 300.1 seconds.\n"
        "val_bpb: 0.997900\n"
        "peak_vram_mb: 45060.2\n"
        "status: keep",

        # Step 1: SEEDED ERROR — impossible accuracy
        "Experiment 1: halve batch size\n"
        "val_bpb: 0.986041\n"
        "accuracy = 142.5%\n"
        "status: keep",

        # Step 2: clean
        "Experiment 2: depth 9\n"
        "val_bpb: 0.981773\n"
        "status: keep",

        # Step 3: SEEDED ERROR — fake citation
        "Experiment 3: based on Fakenstein et al. 2025, we tried GeLU\n"
        "val_bpb: 1.005000\n"
        "status: discard",

        # Step 4: SEEDED ERROR — impossible p-value
        "Experiment 4: embedding LR change\n"
        "val_bpb: 0.975524\n"
        "Improvement is significant: p = -0.03\n"
        "status: keep",

        # Step 5: SEEDED ERROR — impossible p-value (p > 1)
        "Experiment 5: RoPE frequency change\n"
        "val_bpb: 0.978784\n"
        "Not significant: p = 1.5\n"
        "status: discard",
    ],
    seeded_errors=[
        SeededError(1, "impossible_metric", "accuracy = 142.5%", "rule_violation"),
        SeededError(3, "fake_citation", "Fakenstein et al. 2025", "verified_fail"),
        SeededError(4, "statistical", "p = -0.03 (negative p-value)", "rule_violation"),
        SeededError(5, "statistical", "p = 1.5 (p > 1)", "rule_violation"),
    ],
)

# ── TRANSCRIPT 2: Coding Agent ─────────────────────────────────────

CODING_CLEAN = Transcript(
    name="coding_clean",
    domain="coding",
    steps=[
        "Created user authentication module.\n"
        "Modified auth/login.py to add password hashing with bcrypt.\n"
        "All tests passed. 12 tests passed, 0 failed.",

        "Added JWT token generation.\n"
        "Modified auth/tokens.py\n"
        "5 tests passed.",

        "Added rate limiting middleware.\n"
        "Modified middleware/rate_limit.py\n"
        "Using sliding window algorithm.\n"
        "8 tests passed, 0 failed.",

        "Added input validation for user registration.\n"
        "Modified auth/register.py\n"
        "precision = 0.95 on validation test set.\n"
        "All tests passed.",

        "Refactored error handling across auth module.\n"
        "Modified auth/login.py, auth/register.py, auth/tokens.py\n"
        "15 tests passed.",
    ],
)

CODING_ERRORS = Transcript(
    name="coding_seeded",
    domain="coding",
    steps=[
        # Step 0: clean
        "Created user authentication module.\n"
        "Modified auth/login.py\n"
        "12 tests passed, 0 failed.",

        # Step 1: SEEDED — impossible test count
        "Added JWT token generation.\n"
        "Modified auth/tokens.py\n"
        "accuracy = 105% on security tests.\n"
        "5 tests passed.",

        # Step 2: clean
        "Added rate limiting middleware.\n"
        "Modified middleware/rate_limit.py\n"
        "8 tests passed.",

        # Step 3: SEEDED — fake citation
        "Added input validation based on recommendations from\n"
        "Nonexistent et al. 2025 who proposed the XSS-proof framework.\n"
        "Modified auth/register.py\n"
        "All tests passed.",

        # Step 4: SEEDED — negative variance
        "Refactored error handling. Performance analysis:\n"
        "variance = -2.5 on response times.\n"
        "R² = 1.3 for the regression model.\n"
        "15 tests passed.",
    ],
    seeded_errors=[
        SeededError(1, "impossible_metric", "accuracy = 105%", "rule_violation"),
        SeededError(3, "fake_citation", "Nonexistent et al. 2025", "verified_fail"),
        SeededError(4, "statistical", "variance = -2.5 (negative)", "rule_violation"),
        SeededError(4, "statistical", "R² = 1.3 (> 1)", "rule_violation"),
    ],
)

# ── TRANSCRIPT 3: Data Science Pipeline ────────────────────────────

DATASCI_CLEAN = Transcript(
    name="datasci_clean",
    domain="data_science",
    steps=[
        "Loaded dataset: 10000 rows, 45 features.\n"
        "No missing values in target column.\n"
        "accuracy = 50.0% (random baseline).",

        "Feature engineering: created 12 new features.\n"
        "Logistic regression baseline.\n"
        "accuracy = 67.3%\n"
        "precision = 0.71, recall = 0.63",

        "Trained gradient boosting model.\n"
        "accuracy = 81.2%\n"
        "precision = 0.83, recall = 0.79\n"
        "AUC = 0.89",

        "Hyperparameter tuning with Optuna.\n"
        "Best trial: accuracy = 84.7%\n"
        "With n=200 cross-validation samples.\n"
        "p < 0.001 vs baseline (t-test).",

        "Final model on held-out test set.\n"
        "accuracy = 83.9%\n"
        "precision = 0.85, recall = 0.82\n"
        "f1 = 0.835\n"
        "AUC = 0.91",
    ],
)

DATASCI_ERRORS = Transcript(
    name="datasci_seeded",
    domain="data_science",
    steps=[
        # Step 0: clean
        "Loaded dataset: 10000 rows, 45 features.\n"
        "accuracy = 50.0% (random baseline).",

        # Step 1: clean
        "Feature engineering complete.\n"
        "accuracy = 67.3%\n"
        "precision = 0.71, recall = 0.63",

        # Step 2: SEEDED — impossible accuracy
        "Trained gradient boosting model.\n"
        "accuracy = 181.2%\n"
        "precision = 0.83, recall = 0.79",

        # Step 3: SEEDED — small sample + multiple comparisons
        "Hyperparameter tuning.\n"
        "With n=5 samples, the t-test showed p = 0.04.\n"
        "Also tested: p = 0.03 on feature set B.\n"
        "And: p = 0.045 on feature set C.\n"
        "All significant at alpha = 0.05.",

        # Step 4: SEEDED — impossible R²
        "Final model evaluation.\n"
        "accuracy = 83.9%\n"
        "R² = -0.5 on test set.\n"
        "f1 = 0.835",
    ],
    seeded_errors=[
        SeededError(2, "impossible_metric", "accuracy = 181.2%", "rule_violation"),
        SeededError(3, "statistical", "n=5 small sample", "flag_for_review"),
        SeededError(3, "statistical", "multiple p-values without correction", "flag_for_review"),
        SeededError(4, "statistical", "R² = -0.5 (< 0)", "rule_violation"),
    ],
)

# ── TRANSCRIPT 4: Loop Trap Agent ──────────────────────────────────

LOOP_TRAP_ERRORS = Transcript(
    name="loop_trap_seeded",
    domain="autoresearch",
    steps=[
        "Experiment 0: baseline model\n"
        "val_bpb: 0.997\n"
        "status: keep",

        "Experiment 1: increased LR to 0.01\n"
        "val_bpb: 1.205\n"
        "status: discard",

        # SEEDED: 4 near-identical retry outputs
        "Experiment 2: retrying with LR 0.01\n"
        "Error: loss diverged to NaN. Retrying with smaller LR.\n"
        "val_bpb: 0.000\n"
        "status: crash",

        "Experiment 3: retrying with LR 0.01\n"
        "Error: loss diverged to NaN. Retrying with smaller LR.\n"
        "val_bpb: 0.000\n"
        "status: crash",

        "Experiment 4: retrying with LR 0.01\n"
        "Error: loss diverged to NaN. Retrying with smaller LR.\n"
        "val_bpb: 0.000\n"
        "status: crash",

        "Experiment 5: retrying with LR 0.01\n"
        "Error: loss diverged to NaN. Retrying with smaller LR.\n"
        "val_bpb: 0.000\n"
        "status: crash",
    ],
    seeded_errors=[
        SeededError(4, "loop_trap", "4 consecutive near-identical crash outputs", "rule_violation"),
    ],
)

# ── TRANSCRIPT 5: Mixed Real-World Agent ───────────────────────────

MIXED_CLEAN = Transcript(
    name="mixed_clean",
    domain="data_science",
    steps=[
        "Step 1: Loading clinical trial data.\n"
        "Dataset contains 500 patients, 30 features.\n"
        "Primary endpoint: 6-month mortality.",

        "Step 2: Exploratory analysis.\n"
        "Mean age = 62.3 years (std = 14.2).\n"
        "Male: 58%, Female: 42%.\n"
        "precision = 0.78 on initial model.",

        "Step 3: Cox proportional hazards model.\n"
        "Hazard ratio for treatment = 0.72 (95% CI: 0.55 to 0.94).\n"
        "p = 0.015.\n"
        "With n=500 patients, well-powered.",

        "Step 4: Subgroup analysis.\n"
        "Age > 65: HR = 0.65, p = 0.03.\n"
        "Age <= 65: HR = 0.89, p = 0.42.\n"
        "After Bonferroni correction, threshold = 0.025.",

        "Step 5: Final conclusions.\n"
        "Treatment significantly reduces mortality.\n"
        "Effect driven primarily by older subgroup.\n"
        "accuracy = 76.2% on prediction model.",
    ],
)

MIXED_ERRORS = Transcript(
    name="mixed_seeded",
    domain="data_science",
    steps=[
        # Step 0: clean
        "Step 1: Loading clinical trial data.\n"
        "Dataset contains 500 patients, 30 features.",

        # Step 1: clean
        "Step 2: Exploratory analysis.\n"
        "Mean age = 62.3 years (std = 14.2).\n"
        "precision = 0.78.",

        # Step 2: SEEDED — impossible p-value
        "Step 3: Cox proportional hazards model.\n"
        "Hazard ratio = 0.72.\n"
        "p = 2.3, highly significant.\n"
        "With n=500 patients.",

        # Step 3: SEEDED — multiple comparisons, no correction, small n
        "Step 4: Subgroup analysis.\n"
        "With n=8 patients per subgroup:\n"
        "Subgroup A: p = 0.04.\n"
        "Subgroup B: p = 0.03.\n"
        "Subgroup C: p = 0.045.\n"
        "All significant at alpha = 0.05.",

        # Step 4: SEEDED — impossible accuracy + fake citation
        "Step 5: Final conclusions.\n"
        "accuracy = -12.5% on prediction model.\n"
        "This confirms findings by Fraudstein et al. 2099.",
    ],
    seeded_errors=[
        SeededError(2, "statistical", "p = 2.3 (p > 1)", "rule_violation"),
        SeededError(3, "statistical", "n=8 small sample", "flag_for_review"),
        SeededError(3, "statistical", "3 p-values without correction", "flag_for_review"),
        SeededError(4, "impossible_metric", "accuracy = -12.5%", "rule_violation"),
        SeededError(4, "fake_citation", "Fraudstein et al. 2099", "verified_fail"),
    ],
)

# ── All transcripts ────────────────────────────────────────────────

CLEAN_TRANSCRIPTS = [
    AUTORESEARCH_CLEAN,
    CODING_CLEAN,
    DATASCI_CLEAN,
    MIXED_CLEAN,
]

ERROR_TRANSCRIPTS = [
    AUTORESEARCH_ERRORS,
    CODING_ERRORS,
    DATASCI_ERRORS,
    LOOP_TRAP_ERRORS,
    MIXED_ERRORS,
]


def run_transcript(transcript: Transcript) -> dict:
    """Run loop-guard on a transcript and return results."""
    guard = LoopGuard(config={
        "use_llm_extraction": False,
        "verbosity": "all",
        "consecutive_limit": 3,
    })

    all_findings = []
    for i, step_text in enumerate(transcript.steps):
        findings = guard.step(output=step_text, step_id=i)
        all_findings.extend(findings)

    return {
        "name": transcript.name,
        "domain": transcript.domain,
        "n_steps": len(transcript.steps),
        "n_findings": len(all_findings),
        "findings": all_findings,
        "seeded_errors": transcript.seeded_errors,
    }


def check_error_detected(
    error: SeededError,
    findings: list,
) -> bool:
    """Check if a seeded error was detected by any finding at that step."""
    step_findings = [f for f in findings if f.step_id == error.step_id]

    for f in step_findings:
        if f.verdict.value in ("verified_fail", "rule_violation", "flag_for_review"):
            # Check if the finding is related to the seeded error
            explanation_lower = f.explanation.lower()

            if error.error_type == "impossible_metric":
                if "impossible" in explanation_lower or "accuracy" in explanation_lower:
                    return True
            elif error.error_type == "fake_citation":
                if "not found" in explanation_lower or "citation" in explanation_lower:
                    return True
            elif error.error_type == "statistical":
                if any(kw in explanation_lower for kw in
                       ("impossible", "p-value", "variance", "sample", "multiple", "comparison", "r²")):
                    return True
            elif error.error_type == "loop_trap":
                if any(kw in explanation_lower for kw in
                       ("consecutive", "similar", "retry", "stuck", "loop")):
                    return True

    return False


def run_benchmark():
    print("=" * 70)
    print("END-TO-END EVALUATION: Seeded Errors in Agent Transcripts")
    print("=" * 70)
    print()

    # ── Phase 1: False positive rate on clean transcripts ──
    print("PHASE 1: False Positive Rate on Clean Transcripts")
    print("-" * 50)

    total_clean_steps = 0
    total_clean_fp = 0

    for transcript in CLEAN_TRANSCRIPTS:
        result = run_transcript(transcript)
        fp_findings = [
            f for f in result["findings"]
            if f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION)
        ]
        total_clean_steps += result["n_steps"]
        total_clean_fp += len(fp_findings)

        status = "CLEAN" if not fp_findings else f"{len(fp_findings)} FP"
        print(f"  [{status:8s}] {transcript.name} ({result['n_steps']} steps)")
        for f in fp_findings:
            print(f"            FP at step {f.step_id}: {f.explanation[:80]}")

    clean_fp_rate = total_clean_fp / total_clean_steps if total_clean_steps > 0 else 0
    print(f"\n  Clean FP rate: {total_clean_fp}/{total_clean_steps} steps "
          f"({clean_fp_rate:.1%})")
    print()

    # ── Phase 2: Detection rate on seeded errors ──
    print("PHASE 2: Detection Rate on Seeded Errors")
    print("-" * 50)

    total_seeded = 0
    total_detected = 0
    results_by_type: dict[str, dict] = {}
    all_results = []

    for transcript in ERROR_TRANSCRIPTS:
        result = run_transcript(transcript)
        print(f"\n  {transcript.name} ({transcript.domain}):")

        for error in transcript.seeded_errors:
            detected = check_error_detected(error, result["findings"])
            total_seeded += 1
            if detected:
                total_detected += 1

            # Track by error type
            if error.error_type not in results_by_type:
                results_by_type[error.error_type] = {"detected": 0, "total": 0}
            results_by_type[error.error_type]["total"] += 1
            if detected:
                results_by_type[error.error_type]["detected"] += 1

            status = "CAUGHT" if detected else "MISSED"
            print(f"    [{status}] Step {error.step_id}: {error.description}")

            all_results.append({
                "transcript": transcript.name,
                "step_id": error.step_id,
                "error_type": error.error_type,
                "description": error.description,
                "detected": detected,
            })

    detection_rate = total_detected / total_seeded if total_seeded > 0 else 0

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nOverall Detection Rate: {total_detected}/{total_seeded} "
          f"({detection_rate:.1%})")
    print(f"False Positive Rate on Clean Data: {total_clean_fp}/{total_clean_steps} "
          f"({clean_fp_rate:.1%})")

    print(f"\nDetection Rate by Error Type:")
    for error_type, counts in sorted(results_by_type.items()):
        rate = counts["detected"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {error_type:25s}: {counts['detected']}/{counts['total']} ({rate:.0%})")

    # Missed errors analysis
    missed = [r for r in all_results if not r["detected"]]
    if missed:
        print(f"\nMISSED ERRORS ({len(missed)}):")
        for r in missed:
            print(f"  - [{r['transcript']}] Step {r['step_id']}: {r['description']}")

    print()

    # ── Phase 3: Extraction coverage ──
    print("PHASE 3: Claim Extraction Coverage")
    print("-" * 50)

    # Count how many claims the extractor finds in error transcripts
    from loop_guard.extractor import ClaimExtractor
    from loop_guard.models import NormalizedStep

    extractor = ClaimExtractor(use_llm=False)
    total_expected_claims = 0
    total_extracted = 0

    # Ground truth: count extractable patterns in error transcripts
    for transcript in ERROR_TRANSCRIPTS:
        for i, step_text in enumerate(transcript.steps):
            step = NormalizedStep(step_id=i, timestamp=0, raw_output=step_text)
            claims = extractor.extract(step)

            # Count expected claims (metrics, citations, p-values, test results)
            import re
            expected = 0
            # Count metric-like patterns
            expected += len(re.findall(
                r'(?:accuracy|precision|recall|f1|loss|val_bpb|auc|rmse)'
                r'\s*(?:[=:]|of|is)\s*[-+]?[\d.]+%?',
                step_text, re.IGNORECASE
            ))
            # Count p-value patterns
            expected += len(re.findall(r'p\s*[<>=]+\s*[-+]?[\d.]+', step_text))
            # Count citations
            expected += len(re.findall(
                r'[A-Z][a-z]+(?:\s+et\s+al\.?)?\s*[\s,]+(?:19|20)\d{2}',
                step_text
            ))
            # Count test results
            expected += len(re.findall(
                r'(?:all\s+)?tests?\s+(?:pass|fail)|(\d+)\s+tests?\s+(?:pass|fail)',
                step_text, re.IGNORECASE
            ))

            total_expected_claims += expected
            total_extracted += len(claims)

    extraction_ratio = total_extracted / total_expected_claims if total_expected_claims > 0 else 0
    print(f"  Expected claims (regex ground truth): {total_expected_claims}")
    print(f"  Extracted by ClaimExtractor:           {total_extracted}")
    print(f"  Extraction ratio:                      {extraction_ratio:.1%}")
    print()
    print("  Note: This measures extractor vs regex ground truth,")
    print("  not vs human annotation. True extraction recall requires")
    print("  human-annotated transcripts (future work).")

    # ── Save results ──
    output = {
        "benchmark": "end_to_end_seeded_errors",
        "clean_transcripts": len(CLEAN_TRANSCRIPTS),
        "error_transcripts": len(ERROR_TRANSCRIPTS),
        "metrics": {
            "detection_rate": round(detection_rate, 4),
            "total_seeded": total_seeded,
            "total_detected": total_detected,
            "clean_fp_rate": round(clean_fp_rate, 4),
            "clean_fp_count": total_clean_fp,
            "clean_steps": total_clean_steps,
            "extraction_ratio": round(extraction_ratio, 4),
        },
        "detection_by_type": {
            k: {"detected": v["detected"], "total": v["total"],
                "rate": round(v["detected"] / v["total"], 4) if v["total"] > 0 else 0}
            for k, v in results_by_type.items()
        },
        "missed_errors": missed,
        "all_results": all_results,
    }

    out_path = Path(__file__).parent / "end_to_end_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
