"""Rule-based checks for statistical claims."""

from __future__ import annotations

import re
import time

from loop_guard.models import (
    Claim,
    Finding,
    NormalizedStep,
    VerificationLevel,
    Verdict,
)


class StatisticalVerifier:
    """Applies rule-based sanity checks to statistical claims."""

    def verify(
        self, claim: Claim, step_history: list[NormalizedStep]
    ) -> Finding:
        """Run all statistical rules against the claim."""
        # Check impossible values first (most concrete)
        finding = self._check_impossible_values(claim)
        if finding:
            return finding

        # Check multiple comparisons
        finding = self._check_multiple_comparisons(claim, step_history)
        if finding:
            return finding

        # Check sample size
        finding = self._check_sample_size(claim)
        if finding:
            return finding

        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.VERIFIED_PASS,
            level=VerificationLevel.RULE_BASED,
            explanation="No statistical rule violations detected.",
            timestamp=time.time(),
        )

    def _check_impossible_values(self, claim: Claim) -> Finding | None:
        """Detect impossible statistical values in claim text."""
        text = claim.text
        evidence_text = ""
        if claim.evidence:
            evidence_text = " ".join(str(v) for v in claim.evidence.values())
        combined = f"{text} {evidence_text}"

        # p-value checks
        for match in re.finditer(r"p\s*[=<>≤≥]\s*([-+]?\d*\.?\d+)", combined):
            val = float(match.group(1))
            if val > 1 or val < 0:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.RULE_VIOLATION,
                    level=VerificationLevel.RULE_BASED,
                    explanation=f"Impossible p-value: {val} (must be in [0, 1]).",
                    expected="0 <= p <= 1",
                    actual=str(val),
                    timestamp=time.time(),
                )

        # R² checks
        for match in re.finditer(r"R[²2]\s*[=<>≤≥]\s*([-+]?\d*\.?\d+)", combined):
            val = float(match.group(1))
            if val > 1 or val < 0:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.RULE_VIOLATION,
                    level=VerificationLevel.RULE_BASED,
                    explanation=f"Impossible R² value: {val} (must be in [0, 1]).",
                    expected="0 <= R² <= 1",
                    actual=str(val),
                    timestamp=time.time(),
                )

        # Negative variance
        for match in re.finditer(
            r"variance\s*[=:]\s*([-+]?\d*\.?\d+)", combined, re.IGNORECASE
        ):
            val = float(match.group(1))
            if val < 0:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.RULE_VIOLATION,
                    level=VerificationLevel.RULE_BASED,
                    explanation=f"Negative variance: {val} (must be >= 0).",
                    expected="variance >= 0",
                    actual=str(val),
                    timestamp=time.time(),
                )

        # Accuracy checks
        for match in re.finditer(
            r"accuracy\s*(?:[=:]|of|is|was|at)\s*([-+]?\d*\.?\d+)\s*%?", combined, re.IGNORECASE
        ):
            val = float(match.group(1))
            if val > 100 or val < 0:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.RULE_VIOLATION,
                    level=VerificationLevel.RULE_BASED,
                    explanation=f"Impossible accuracy: {val}% (must be in [0, 100]).",
                    expected="0% <= accuracy <= 100%",
                    actual=f"{val}%",
                    timestamp=time.time(),
                )

        return None

    def _check_multiple_comparisons(
        self, claim: Claim, step_history: list[NormalizedStep]
    ) -> Finding | None:
        """Flag multiple p-values without correction methods."""
        recent = step_history[-20:]
        all_text = " ".join(s.raw_output for s in recent) + " " + claim.text

        p_values = re.findall(r"p\s*[=<>≤≥]\s*\d*\.?\d+", all_text)
        if len(p_values) <= 1:
            return None

        corrections = re.findall(
            r"bonferroni|benjamini|hochberg|\bBH\b|\bFDR\b|\bholm\b",
            all_text,
            re.IGNORECASE,
        )
        if corrections:
            return None

        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.FLAG_FOR_REVIEW,
            level=VerificationLevel.RULE_BASED,
            explanation=(
                f"Found {len(p_values)} p-value comparisons in recent outputs "
                f"without mention of multiple-comparison correction "
                f"(Bonferroni, BH, FDR, Holm)."
            ),
            timestamp=time.time(),
        )

    def _check_sample_size(self, claim: Claim) -> Finding | None:
        """Flag small sample sizes for parametric claims."""
        text = claim.text
        evidence_text = ""
        if claim.evidence:
            evidence_text = " ".join(str(v) for v in claim.evidence.values())
        combined = f"{text} {evidence_text}"

        # Look for sample size indicators: n=, N=, sample size, etc.
        for match in re.finditer(
            r"(?:[nN]\s*=\s*|sample\s+(?:size\s+)?(?:of\s+)?|n\s*:\s*)(\d+)",
            combined,
        ):
            n = int(match.group(1))
            if n < 30:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.FLAG_FOR_REVIEW,
                    level=VerificationLevel.RULE_BASED,
                    explanation=(
                        f"Small sample size (n={n}) for a parametric claim. "
                        f"Results may not be reliable with n < 30."
                    ),
                    expected="n >= 30 for parametric claims",
                    actual=f"n = {n}",
                    timestamp=time.time(),
                )

        return None
