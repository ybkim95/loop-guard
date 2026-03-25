#!/usr/bin/env python3
"""
BENCHMARK: Statistical Verifier Precision & Recall

Tests against:
- 50 CORRECT statistical claims (should pass)
- 50 INCORRECT statistical claims (should be flagged)

Reports precision, recall, F1 for detecting statistical errors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loop_guard.models import Claim, ClaimType, NormalizedStep, Verdict
from loop_guard.verifiers.statistical import StatisticalVerifier

# ── 50 CORRECT statistical claims ──────────────────────────────────

CORRECT_CLAIMS = [
    "p = 0.03, statistically significant at alpha = 0.05",
    "p = 0.001, highly significant",
    "p < 0.05",
    "p = 0.5, not significant",
    "p = 1.0, no effect",
    "p = 0.0001",
    "accuracy = 95.2%",
    "accuracy = 50.0% (random baseline)",
    "accuracy = 0.1% (worst case)",
    "accuracy = 100.0% (perfect on training set)",

    "R² = 0.85",
    "R² = 0.0 (no correlation)",
    "R² = 1.0 (perfect fit)",
    "R² = 0.42",
    "variance = 2.5",
    "variance = 0.001",
    "variance = 100.0",
    "precision = 0.91",
    "recall = 0.88",
    "f1 = 0.895",

    "The sample size was n=500, well above the minimum for parametric tests",
    "With n=1000 participants, the study was well-powered",
    "Using n=50 observations per group",
    "Sample of n=200 patients",
    "With n=30 measurements per condition",

    "The effect size (Cohen's d) was 0.8, considered large",
    "Mean difference = 5.2 (95% CI: 3.1 to 7.3)",
    "Standard deviation = 12.4",
    "The median was 45.0, IQR = 38.0 to 52.0",
    "Skewness = 0.3, approximately normal",

    "ANOVA F(2, 147) = 4.52, p = 0.012",
    "Chi-square test: X² = 8.3, df = 3, p = 0.04",
    "Pearson correlation r = 0.65, p < 0.001",
    "Spearman rho = 0.72, p = 0.003",
    "Mann-Whitney U = 1234, p = 0.02",

    "After Bonferroni correction, the threshold was 0.01",
    "Using FDR correction (Benjamini-Hochberg), q = 0.05",
    "The Holm-Bonferroni method was applied for multiple comparisons",
    "AUC = 0.92 (95% CI: 0.88 to 0.96)",
    "Sensitivity = 0.85, Specificity = 0.90",

    "Log-rank test: chi-square = 6.2, p = 0.013",
    "Hazard ratio = 1.5 (95% CI: 1.1 to 2.0)",
    "Odds ratio = 2.3 (p = 0.008)",
    "Number needed to treat = 12",
    "Inter-rater reliability: kappa = 0.78",

    "Mean absolute error = 3.2",
    "Root mean square error = 4.1",
    "The model explained 72% of the variance",
    "Confidence interval width = 2.4",
    "Standard error of the mean = 0.8",
]

# ── 50 INCORRECT statistical claims ────────────────────────────────

INCORRECT_CLAIMS = [
    # Impossible p-values
    "p = 1.5, highly significant",
    "p = -0.03, statistically significant",
    "p = 2.0",
    "p = -1.0, very significant",
    "p = 5.0, marginally significant",

    # Impossible accuracy
    "accuracy = 105%",
    "accuracy = -3.2%",
    "accuracy = 200%",
    "accuracy = 150.5%",
    "accuracy = -10%",

    # Impossible R²
    "R² = 1.5",
    "R² = -0.3",
    "R² = 2.0",
    "R² = -1.0",
    "R² = 1.1",

    # Negative variance
    "variance = -2.5",
    "variance = -0.001",
    "variance = -100",
    "variance = -50.3",
    "variance = -0.1",

    # Small sample sizes (for parametric claims)
    "With n=3, we found a significant effect (p < 0.05)",
    "Using n=5 samples, the t-test showed p = 0.04",
    "n=2 participants showed significant improvement",
    "With n=7, the ANOVA was significant",
    "n=10 measurements: p = 0.03, significant after correction",

    # Additional impossible values
    "p = 3.14159",
    "accuracy = 999%",
    "R² = 10.0",
    "variance = -999",
    "p = -0.5, significant",

    # More impossible accuracy
    "accuracy = 101%",
    "accuracy = -0.5%",
    "accuracy = 110.3%",
    "accuracy = 250%",
    "accuracy = -50%",

    # More impossible R²
    "R² = 1.001",
    "R² = -0.01",
    "R² = 5.0",
    "R² = -2.0",
    "R² = 1.5 indicates strong fit",

    # More negative variance
    "The variance was -3.7",
    "Sample variance = -12.5",
    "variance = -0.5 for this group",
    "Pooled variance = -8.0",
    "variance = -1.0 after correction",

    # More impossible p-values
    "p = 1.2, not significant",
    "p = -0.001, highly significant",
    "p = 10.0",
    "p = -2.5",
    "p = 1.5 after correction",
]


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Statistical Verifier Precision & Recall")
    print(f"50 correct claims + 50 incorrect claims")
    print("=" * 70)
    print()

    verifier = StatisticalVerifier()
    results = []

    # Test CORRECT claims
    print("Testing 50 CORRECT statistical claims...")
    for i, text in enumerate(CORRECT_CLAIMS):
        claim = Claim(
            claim_type=ClaimType.STATISTICAL,
            source_step=i,
            text=text,
            verifiable=True,
        )
        finding = verifier.verify(claim, [])

        is_correct = finding.verdict in (Verdict.VERIFIED_PASS, Verdict.SKIPPED)
        results.append({
            "text": text[:80],
            "ground_truth": "correct",
            "verdict": finding.verdict.value,
            "correct": is_correct,
            "explanation": finding.explanation[:100],
        })

        if not is_correct:
            print(f"  [FP] {text[:60]}... → {finding.verdict.value}")

    print(f"  Correct claims: {sum(1 for r in results if r['ground_truth']=='correct' and r['correct'])}/50 passed")
    print()

    # Test INCORRECT claims
    print("Testing 50 INCORRECT statistical claims...")
    for i, text in enumerate(INCORRECT_CLAIMS):
        claim = Claim(
            claim_type=ClaimType.STATISTICAL,
            source_step=100 + i,
            text=text,
            verifiable=True,
        )
        finding = verifier.verify(claim, [])

        is_correct = finding.verdict in (Verdict.RULE_VIOLATION, Verdict.FLAG_FOR_REVIEW)
        results.append({
            "text": text[:80],
            "ground_truth": "incorrect",
            "verdict": finding.verdict.value,
            "correct": is_correct,
            "explanation": finding.explanation[:100],
        })

        if not is_correct:
            print(f"  [FN] {text[:60]}... → {finding.verdict.value}")

    print(f"  Incorrect claims caught: {sum(1 for r in results if r['ground_truth']=='incorrect' and r['correct'])}/50")
    print()

    # ── Compute metrics ──
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    correct_results = [r for r in results if r["ground_truth"] == "correct"]
    incorrect_results = [r for r in results if r["ground_truth"] == "incorrect"]

    # TP = incorrect claim correctly flagged
    # TN = correct claim correctly passed
    # FP = correct claim incorrectly flagged
    # FN = incorrect claim incorrectly passed
    tp = sum(1 for r in incorrect_results if r["correct"])
    fn = sum(1 for r in incorrect_results if not r["correct"])
    tn = sum(1 for r in correct_results if r["correct"])
    fp = sum(1 for r in correct_results if not r["correct"])

    print()
    print("Confusion Matrix:")
    print(f"                      Predicted OK    Predicted ERROR")
    print(f"  Actually CORRECT    TN = {tn:3d}        FP = {fp:3d}")
    print(f"  Actually INCORRECT  FN = {fn:3d}        TP = {tp:3d}")
    print()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print(f"Precision (error detection): {precision:.1%}  ({tp}/{tp+fp})")
    print(f"Recall (error detection):    {recall:.1%}  ({tp}/{tp+fn})")
    print(f"F1 Score:                    {f1:.1%}")
    print(f"Overall accuracy:            {accuracy:.1%}  ({tp+tn}/{tp+tn+fp+fn})")
    print()

    # Error analysis
    fps = [r for r in correct_results if not r["correct"]]
    fns = [r for r in incorrect_results if not r["correct"]]

    if fps:
        print("FALSE POSITIVES (correct claims incorrectly flagged):")
        for r in fps:
            print(f"  - \"{r['text']}\" → {r['verdict']}: {r['explanation']}")
        print()

    if fns:
        print("FALSE NEGATIVES (incorrect claims missed):")
        for r in fns:
            print(f"  - \"{r['text']}\" → {r['verdict']}")
        print()

    # Save
    output = {
        "benchmark": "statistical_verifier",
        "n_correct": len(CORRECT_CLAIMS),
        "n_incorrect": len(INCORRECT_CLAIMS),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        },
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "results": results,
    }

    out_path = Path(__file__).parent / "statistical_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
