"""Tests for CitationVerifier."""

from unittest.mock import patch

import httpx

from loop_guard.models import Claim, ClaimType, Verdict
from loop_guard.verifiers.citation import CitationVerifier, _title_similarity


def _make_citation_claim(text: str, title: str | None = None) -> Claim:
    evidence = {"title": title} if title else None
    return Claim(
        claim_type=ClaimType.CITATION,
        source_step=0,
        text=text,
        verifiable=True,
        evidence=evidence,
    )


class TestTitleSimilarity:
    def test_identical_titles(self):
        assert _title_similarity("Attention Is All You Need", "Attention Is All You Need") > 0.9

    def test_case_insensitive(self):
        assert _title_similarity("Attention Is All You Need", "attention is all you need") > 0.9

    def test_completely_different(self):
        assert _title_similarity("Attention Is All You Need", "Cold Fusion Energy Prediction") < 0.1

    def test_partial_overlap(self):
        sim = _title_similarity("Deep Learning", "Deep Learning for Computer Vision")
        assert 0.2 < sim < 0.8

    def test_empty_strings(self):
        assert _title_similarity("", "something") == 0.0
        assert _title_similarity("something", "") == 0.0


class TestCitationParsing:
    def test_parse_author_et_al_year(self):
        v = CitationVerifier()
        author, year = v._parse_citation("Smith et al. 2024")
        assert author == "Smith"
        assert year == 2024

    def test_parse_author_year(self):
        v = CitationVerifier()
        author, year = v._parse_citation("Jones 2023")
        assert author == "Jones"
        assert year == 2023

    def test_parse_author_and_coauthor(self):
        v = CitationVerifier()
        author, year = v._parse_citation("Brown and Lee 2022")
        assert author == "Brown"
        assert year == 2022

    def test_parse_with_comma(self):
        v = CitationVerifier()
        author, year = v._parse_citation("Smith, 2024")
        assert author == "Smith"
        assert year == 2024


class TestCitationVerification:
    def test_title_match_passes(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Vaswani et al. 2017", title="Attention Is All You Need")

        with patch.object(v, "_search_crossref", return_value=[
            {"title": "Attention is all you need", "source": "CrossRef"},
        ]), patch.object(v, "_search_semantic_scholar", return_value=[]):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_PASS
            assert "similarity" in finding.explanation.lower()

    def test_title_mismatch_fails(self):
        """Author published in that year, but title doesn't match."""
        v = CitationVerifier()
        claim = _make_citation_claim(
            "Smith et al. 2024",
            title="Neural Scaling Laws for Quantum Computing",
        )

        with patch.object(v, "_search_crossref", return_value=[
            {"title": "Protein structure prediction using deep networks", "source": "CrossRef"},
        ]), patch.object(v, "_search_semantic_scholar", return_value=[]):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_FAIL
            assert "no title matches" in finding.explanation.lower()

    def test_no_results_fails(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Fakerson et al. 2099", title="Nonexistent Paper")

        with patch.object(v, "_search_crossref", return_value=[]), \
             patch.object(v, "_search_semantic_scholar", return_value=[]):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_FAIL
            assert "not found" in finding.explanation.lower()

    def test_no_title_flags_for_review(self):
        """Without a title, we can't confirm the specific claim."""
        v = CitationVerifier()
        claim = _make_citation_claim("Smith 2024")  # no title

        with patch.object(v, "_search_crossref", return_value=[
            {"title": "Some paper by Smith", "source": "CrossRef"},
        ]), patch.object(v, "_search_semantic_scholar", return_value=[]):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.FLAG_FOR_REVIEW
            assert "no title was provided" in finding.explanation.lower()

    def test_api_failure_returns_skipped(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Smith 2024", title="Some Paper")

        with patch.object(v, "_search_crossref", side_effect=httpx.HTTPError("timeout")), \
             patch.object(v, "_search_semantic_scholar", side_effect=httpx.HTTPError("timeout")):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.SKIPPED

    def test_unparseable_citation(self):
        v = CitationVerifier()
        claim = _make_citation_claim("some random text without a citation")
        finding = v.verify(claim)
        assert finding.verdict == Verdict.SKIPPED

    def test_semantic_scholar_match(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Devlin et al. 2019", title="BERT")

        with patch.object(v, "_search_crossref", return_value=[]), \
             patch.object(v, "_search_semantic_scholar", return_value=[
                 {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "source": "Semantic Scholar"},
             ]):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_PASS

    def test_best_match_selected(self):
        """When multiple results returned, the best title match wins."""
        v = CitationVerifier()
        claim = _make_citation_claim("Brown et al. 2020", title="Language Models are Few-Shot Learners")

        with patch.object(v, "_search_crossref", return_value=[
            {"title": "Unrelated paper about brown bears", "source": "CrossRef"},
            {"title": "Language Models are Few-Shot Learners", "source": "CrossRef"},
        ]), patch.object(v, "_search_semantic_scholar", return_value=[]):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_PASS
