"""Tests for CitationVerifier."""

from unittest.mock import MagicMock, patch

import httpx

from loop_guard.models import Claim, ClaimType, Verdict
from loop_guard.verifiers.citation import CitationVerifier


def _make_citation_claim(text: str, title: str | None = None) -> Claim:
    evidence = {"title": title} if title else None
    return Claim(
        claim_type=ClaimType.CITATION,
        source_step=0,
        text=text,
        verifiable=True,
        evidence=evidence,
    )


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
    def test_found_in_crossref(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Vaswani et al. 2017", title="Attention Is All You Need")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "items": [
                    {
                        "title": ["Attention is all you need"],
                        "author": [{"family": "Vaswani"}],
                        "published-print": {"date-parts": [[2017]]},
                        "DOI": "10.5555/3295222.3295349",
                    }
                ]
            }
        }

        with patch.object(v, "_search_crossref", return_value={"title": "Attention is all you need", "doi": "10.5555/3295222.3295349"}):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_PASS

    def test_not_found_anywhere(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Fakerson et al. 2099", title="A Paper That Does Not Exist")

        with patch.object(v, "_search_crossref", return_value=None), \
             patch.object(v, "_search_semantic_scholar", return_value=None):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_FAIL
            assert "not found" in finding.explanation.lower()

    def test_found_in_semantic_scholar(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Devlin et al. 2019", title="BERT")

        with patch.object(v, "_search_crossref", return_value=None), \
             patch.object(v, "_search_semantic_scholar", return_value={"title": "BERT: Pre-training of Deep Bidirectional Transformers"}):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.VERIFIED_PASS

    def test_api_failure_returns_skipped(self):
        v = CitationVerifier()
        claim = _make_citation_claim("Smith 2024")

        with patch.object(v, "_search_crossref", side_effect=httpx.HTTPError("timeout")), \
             patch.object(v, "_search_semantic_scholar", side_effect=httpx.HTTPError("timeout")):
            finding = v.verify(claim)
            assert finding.verdict == Verdict.SKIPPED

    def test_unparseable_citation(self):
        v = CitationVerifier()
        claim = _make_citation_claim("some random text without a citation")
        finding = v.verify(claim)
        assert finding.verdict in (Verdict.SKIPPED, Verdict.VERIFIED_FAIL)
