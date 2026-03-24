"""Checks citations against CrossRef and Semantic Scholar APIs."""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx

from loop_guard.models import (
    Claim,
    Finding,
    VerificationLevel,
    Verdict,
)


class CitationVerifier:
    """Verifies academic citations via CrossRef and Semantic Scholar."""

    def __init__(self) -> None:
        self._last_request_time: float = 0.0

    def verify(self, claim: Claim) -> Finding:
        """Verify a citation claim against external APIs."""
        try:
            author, year = self._parse_citation(claim.text)
        except ValueError:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation="Could not parse author/year from citation text.",
                timestamp=time.time(),
            )

        title = claim.evidence.get("title", "") if claim.evidence else ""

        crossref_error = False
        semantic_error = False

        # Try CrossRef first
        try:
            cr_result = self._search_crossref(author, year, title)
            if cr_result:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.VERIFIED_PASS,
                    level=VerificationLevel.DETERMINISTIC,
                    explanation=f"Citation confirmed via CrossRef: {cr_result.get('title', 'N/A')}",
                    expected=f"{author} ({year})",
                    actual=cr_result.get("title", ""),
                    timestamp=time.time(),
                )
        except Exception:
            crossref_error = True

        # Try Semantic Scholar
        try:
            ss_result = self._search_semantic_scholar(author, year, title)
            if ss_result:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.VERIFIED_PASS,
                    level=VerificationLevel.DETERMINISTIC,
                    explanation=f"Citation confirmed via Semantic Scholar: {ss_result.get('title', 'N/A')}",
                    expected=f"{author} ({year})",
                    actual=ss_result.get("title", ""),
                    timestamp=time.time(),
                )
        except Exception:
            semantic_error = True

        # If both APIs errored, we can't verify — skip
        if crossref_error and semantic_error:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Could not verify citation (API errors): {author} ({year})",
                expected=f"{author} ({year})",
                actual="API unreachable",
                timestamp=time.time(),
            )

        # At least one API responded but neither found the citation
        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.VERIFIED_FAIL,
            level=VerificationLevel.DETERMINISTIC,
            explanation=f"Citation not found in CrossRef or Semantic Scholar: {author} ({year})",
            expected=f"{author} ({year})",
            actual="No matching record found",
            timestamp=time.time(),
        )

    def _parse_citation(self, text: str) -> tuple[str, int]:
        """Extract author surname and year from citation text.

        Handles patterns like:
        - "Smith et al. 2024"
        - "Smith (2024)"
        - "Smith et al., 2024"
        - "Smith & Jones 2024"
        """
        match = re.search(
            r"([A-Z][a-z]+)(?:\s+(?:et\s+al\.?|(?:and|&)\s+[A-Z][a-z]+))?"
            r"[\s,.(]+(\d{4})",
            text,
        )
        if not match:
            raise ValueError(f"Cannot parse citation from: {text}")
        author = match.group(1).strip()
        year = int(match.group(2))
        return author, year

    def _rate_limit(self) -> None:
        """Enforce polite 1-second spacing between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _search_crossref(
        self, author: str, year: int, title: str
    ) -> Optional[dict]:
        """Search CrossRef for a matching work."""
        self._rate_limit()
        params = {
            "query.author": author,
            "query.bibliographic": title,
            "filter": f"from-pub-date:{year},until-pub-date:{year}",
            "rows": "3",
        }
        try:
            resp = httpx.get(
                "https://api.crossref.org/works",
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException):
            return None

        data = resp.json()
        items = data.get("message", {}).get("items", [])
        if not items:
            return None

        # Return first item as confirmation
        item = items[0]
        return {
            "title": (item.get("title") or [""])[0],
            "doi": item.get("DOI", ""),
        }

    def _search_semantic_scholar(
        self, author: str, year: int, title: str
    ) -> Optional[dict]:
        """Search Semantic Scholar for a matching paper."""
        self._rate_limit()
        query = f"{author} {year} {title}".strip()
        params = {"query": query, "limit": "3"}
        try:
            resp = httpx.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException):
            return None

        data = resp.json()
        papers = data.get("data", [])
        if not papers:
            return None

        paper = papers[0]
        return {
            "title": paper.get("title", ""),
            "paperId": paper.get("paperId", ""),
        }
