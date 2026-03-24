"""Tests for ClaimExtractor."""

from loop_guard.extractor import ClaimExtractor
from loop_guard.models import ClaimType, NormalizedStep


def _make_step(output: str, step_id: int = 0) -> NormalizedStep:
    return NormalizedStep(step_id=step_id, timestamp=0.0, raw_output=output)


class TestRegexExtraction:
    def setup_method(self):
        self.extractor = ClaimExtractor(use_llm=False)

    def test_extract_citation(self):
        step = _make_step("As shown by Smith et al. 2024, transformers scale well.")
        claims = self.extractor.extract(step)
        citation_claims = [c for c in claims if c.claim_type == ClaimType.CITATION]
        assert len(citation_claims) >= 1
        assert "Smith" in citation_claims[0].text

    def test_extract_multiple_citations(self):
        step = _make_step(
            "Smith et al. 2024 and Jones 2023 both found similar results. "
            "Brown and Lee 2022 disagreed."
        )
        claims = self.extractor.extract(step)
        citation_claims = [c for c in claims if c.claim_type == ClaimType.CITATION]
        assert len(citation_claims) >= 2

    def test_extract_metric(self):
        step = _make_step("The model achieved accuracy = 94.2% on the test set.")
        claims = self.extractor.extract(step)
        metric_claims = [c for c in claims if c.claim_type == ClaimType.METRIC]
        assert len(metric_claims) >= 1
        assert "94.2" in metric_claims[0].text

    def test_extract_various_metrics(self):
        step = _make_step("loss: 0.342, precision = 0.89, recall: 0.91, f1 = 0.90")
        claims = self.extractor.extract(step)
        metric_claims = [c for c in claims if c.claim_type == ClaimType.METRIC]
        assert len(metric_claims) >= 3

    def test_extract_p_value(self):
        step = _make_step("The difference was significant (p < 0.05).")
        claims = self.extractor.extract(step)
        stat_claims = [c for c in claims if c.claim_type == ClaimType.STATISTICAL]
        assert len(stat_claims) >= 1

    def test_extract_p_value_scientific(self):
        step = _make_step("We found p = 3.2e-5, highly significant.")
        claims = self.extractor.extract(step)
        stat_claims = [c for c in claims if c.claim_type == ClaimType.STATISTICAL]
        assert len(stat_claims) >= 1

    def test_extract_test_result_pass(self):
        step = _make_step("All tests passed successfully.")
        claims = self.extractor.extract(step)
        test_claims = [c for c in claims if c.claim_type == ClaimType.TEST_RESULT]
        assert len(test_claims) >= 1

    def test_extract_test_result_count(self):
        step = _make_step("42 tests passed, 3 failed")
        claims = self.extractor.extract(step)
        test_claims = [c for c in claims if c.claim_type == ClaimType.TEST_RESULT]
        assert len(test_claims) >= 1

    def test_extract_file_state(self):
        step = _make_step("Modified `train.py` to add dropout layer.")
        claims = self.extractor.extract(step)
        file_claims = [c for c in claims if c.claim_type == ClaimType.FILE_STATE]
        assert len(file_claims) >= 1
        assert "train.py" in file_claims[0].text

    def test_no_claims_from_plain_text(self):
        step = _make_step("Thinking about the problem. Let me consider the options.")
        claims = self.extractor.extract(step)
        # Should produce no typed claims (only general if any)
        typed_claims = [c for c in claims if c.claim_type != ClaimType.GENERAL]
        assert len(typed_claims) == 0

    def test_empty_output(self):
        step = _make_step("")
        claims = self.extractor.extract(step)
        assert len(claims) == 0

    def test_mixed_claims(self):
        step = _make_step(
            "Based on Smith et al. 2024, we trained the model. "
            "accuracy = 94.2%, loss: 0.15. "
            "All tests passed. p < 0.01."
        )
        claims = self.extractor.extract(step)
        types = {c.claim_type for c in claims}
        assert ClaimType.CITATION in types
        assert ClaimType.METRIC in types
        assert ClaimType.TEST_RESULT in types
        assert ClaimType.STATISTICAL in types

    def test_claim_source_step(self):
        step = _make_step("accuracy = 99%", step_id=42)
        claims = self.extractor.extract(step)
        for claim in claims:
            assert claim.source_step == 42
