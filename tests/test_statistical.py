"""Tests for StatisticalVerifier."""

from loop_guard.models import Claim, ClaimType, NormalizedStep, Verdict
from loop_guard.verifiers.statistical import StatisticalVerifier


def _make_stat_claim(text: str, step: int = 0, evidence: dict | None = None) -> Claim:
    return Claim(
        claim_type=ClaimType.STATISTICAL,
        source_step=step,
        text=text,
        verifiable=True,
        evidence=evidence,
    )


def _make_step(step_id: int, output: str) -> NormalizedStep:
    return NormalizedStep(step_id=step_id, timestamp=0.0, raw_output=output)


class TestStatisticalVerifier:
    def test_impossible_p_value_greater_than_1(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim("The test yielded p = 1.5")
        history = [_make_step(0, "p = 1.5")]
        finding = v.verify(claim, history)
        assert finding.verdict == Verdict.RULE_VIOLATION
        assert "impossible" in finding.explanation.lower() or "p" in finding.explanation.lower()

    def test_impossible_negative_p_value(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim("p = -0.03")
        finding = v.verify(claim, [])
        assert finding.verdict == Verdict.RULE_VIOLATION

    def test_impossible_accuracy_over_100(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim("accuracy = 105%")
        finding = v.verify(claim, [])
        assert finding.verdict == Verdict.RULE_VIOLATION

    def test_impossible_r2_over_1(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim("R² = 1.3")
        finding = v.verify(claim, [])
        assert finding.verdict == Verdict.RULE_VIOLATION

    def test_valid_p_value_passes(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim("p = 0.03, statistically significant")
        finding = v.verify(claim, [])
        assert finding.verdict == Verdict.VERIFIED_PASS

    def test_multiple_comparisons_without_correction(self):
        v = StatisticalVerifier()
        # Build history with multiple p-values
        history = [
            _make_step(0, "Comparison 1: p = 0.04"),
            _make_step(1, "Comparison 2: p = 0.03"),
            _make_step(2, "Comparison 3: p = 0.02"),
        ]
        claim = _make_stat_claim("Comparison 4: p = 0.01", step=3)
        finding = v.verify(claim, history)
        # Should flag multiple comparisons without correction
        assert finding.verdict in (Verdict.RULE_VIOLATION, Verdict.FLAG_FOR_REVIEW)
        assert "multiple" in finding.explanation.lower() or "comparison" in finding.explanation.lower()

    def test_multiple_comparisons_with_bonferroni(self):
        v = StatisticalVerifier()
        history = [
            _make_step(0, "p = 0.04"),
            _make_step(1, "p = 0.03"),
        ]
        claim = _make_stat_claim(
            "After Bonferroni correction, p = 0.01", step=2
        )
        finding = v.verify(claim, history)
        # Should pass because correction is mentioned
        assert finding.verdict == Verdict.VERIFIED_PASS

    def test_small_sample_size(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim(
            "With n=5, we found a significant effect (p < 0.05)",
            evidence={"sample_size": 5},
        )
        finding = v.verify(claim, [])
        assert finding.verdict in (Verdict.RULE_VIOLATION, Verdict.FLAG_FOR_REVIEW)
        assert "sample" in finding.explanation.lower()

    def test_valid_statistics_pass(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim("With n=500, accuracy = 94.2%")
        finding = v.verify(claim, [])
        assert finding.verdict == Verdict.VERIFIED_PASS

    def test_negative_variance(self):
        v = StatisticalVerifier()
        claim = _make_stat_claim("variance = -2.5")
        finding = v.verify(claim, [])
        assert finding.verdict == Verdict.RULE_VIOLATION
