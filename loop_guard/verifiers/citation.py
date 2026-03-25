"""Checks citations against CrossRef and Semantic Scholar APIs.

The verifier searches for papers by author+year, then compares the
returned titles against the claimed title using token-level similarity.
A match requires sufficient overlap between claimed and returned titles,
not just that *some* paper by that author in that year exists.

Without a claimed title, the verifier can only confirm that the author
published in the given year — it cannot verify the specific claim.
This is reported honestly as FLAG_FOR_REVIEW, not VERIFIED_PASS.
"""

from __future__ import annotations

import re
import time

import httpx

from loop_guard.models import (
    Claim,
    Finding,
    Verdict,
    VerificationLevel,
)


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in text.split() if len(w) > 2}


def _title_similarity(claimed: str, returned: str) -> float:
    """Compute similarity between two titles.

    Uses max of:
    - Token-level Jaccard similarity (handles rewordings)
    - Substring containment (handles short titles like "BERT")

    Returns a value in [0, 1].
    """
    if not claimed or not returned:
        return 0.0

    claimed_lower = claimed.lower().strip()
    returned_lower = returned.lower().strip()

    # Exact match
    if claimed_lower == returned_lower:
        return 1.0

    # Substring containment: if one title fully contains the other
    if claimed_lower in returned_lower or returned_lower in claimed_lower:
        shorter = min(len(claimed_lower), len(returned_lower))
        longer = max(len(claimed_lower), len(returned_lower))
        # Scale by length ratio to avoid single-character matches
        if shorter >= 3:
            return max(0.5, shorter / longer)

    # Token-level Jaccard
    tokens_a = _tokenize(claimed)
    tokens_b = _tokenize(returned)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# Minimum title similarity to consider a match.
# 0.3 is deliberately conservative — even partial title overlap
# (e.g., "Attention Is All You Need" matching "Attention is all you need")
# should clear this threshold, while completely unrelated papers won't.
_TITLE_MATCH_THRESHOLD = 0.3


class CitationVerifier:
    """Verifies academic citations via CrossRef and Semantic Scholar.

    Design decisions:
    - With title: search APIs, compare returned titles. PASS only if
      a returned title matches the claimed title above threshold.
    - Without title: can only verify author+year existence. Returns
      FLAG_FOR_REVIEW (not PASS) because we cannot confirm the specific claim.
    - API errors: SKIPPED (honest about inability to verify).
    """

    def __init__(self, title_threshold: float = _TITLE_MATCH_THRESHOLD) -> None:
        self._last_request_time: float = 0.0
        self.title_threshold = title_threshold

    def verify(self, claim: Claim) -> Finding:
        """Verify a citation claim against external APIs."""
        try:
            author, year = self._parse_citation(claim.text)
        except ValueError:
            return self._finding(
                claim, Verdict.SKIPPED,
                "Could not parse author/year from citation text.",
            )

        title = claim.evidence.get("title", "") if claim.evidence else ""

        # Collect results from both APIs
        crossref_error = False
        semantic_error = False
        all_returned: list[dict] = []  # {"title": str, "source": str}

        try:
            cr_results = self._search_crossref(author, year, title)
            all_returned.extend(cr_results)
        except Exception:
            crossref_error = True

        try:
            ss_results = self._search_semantic_scholar(author, year, title)
            all_returned.extend(ss_results)
        except Exception:
            semantic_error = True

        # Both APIs failed — we cannot verify
        if crossref_error and semantic_error:
            return self._finding(
                claim, Verdict.SKIPPED,
                f"Could not verify citation (API errors): {author} ({year})",
            )

        # No results from either API
        if not all_returned:
            return self._finding(
                claim, Verdict.VERIFIED_FAIL,
                f"Citation not found: no papers by {author} in {year} "
                f"in CrossRef or Semantic Scholar.",
                expected=f"{author} ({year})" + (f": {title}" if title else ""),
                actual="No results returned",
            )

        # If we have a claimed title, require title match
        if title:
            best_match = None
            best_sim = 0.0
            for result in all_returned:
                sim = _title_similarity(title, result["title"])
                if sim > best_sim:
                    best_sim = sim
                    best_match = result

            if best_match and best_sim >= self.title_threshold:
                return self._finding(
                    claim, Verdict.VERIFIED_PASS,
                    f"Citation confirmed via {best_match['source']}: "
                    f"\"{best_match['title']}\" "
                    f"(title similarity: {best_sim:.0%})",
                    expected=f"{author} ({year}): {title}",
                    actual=best_match["title"],
                )
            else:
                # API returned papers by this author in this year,
                # but none match the claimed title
                returned_titles = [r["title"] for r in all_returned[:3]]
                best_title = best_match["title"][:80] if best_match else "N/A"
                return self._finding(
                    claim, Verdict.VERIFIED_FAIL,
                    f"Author {author} published in {year}, but no title "
                    f"matches \"{title[:80]}\". "
                    f"Best match: \"{best_title}\" "
                    f"(similarity: {best_sim:.0%})",
                    expected=title,
                    actual="; ".join(returned_titles)[:200],
                )

        # No title provided — we found papers by this author in this year,
        # but we cannot confirm the specific claim
        return self._finding(
            claim, Verdict.FLAG_FOR_REVIEW,
            f"Papers by {author} in {year} exist, but no title was provided "
            f"to verify the specific claim. Cannot confirm this is the "
            f"correct paper.",
            expected=f"{author} ({year})",
            actual=f"Found {len(all_returned)} paper(s), cannot match without title",
        )

    def _parse_citation(self, text: str) -> tuple[str, int]:
        """Extract author surname and year from citation text."""
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
    ) -> list[dict]:
        """Search CrossRef. Returns list of {title, source} dicts.

        Uses ±1 year range because publication dates often differ from
        the conference/arxiv year a paper is commonly cited by.
        """
        self._rate_limit()
        params: dict = {
            "query.author": author,
            "filter": f"from-pub-date:{year - 1},until-pub-date:{year + 1}",
            "rows": "5",
        }
        if title:
            params["query.bibliographic"] = title

        try:
            resp = httpx.get(
                "https://api.crossref.org/works",
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException):
            return []

        data = resp.json()
        items = data.get("message", {}).get("items", [])
        results = []
        for item in items[:5]:
            item_title = (item.get("title") or [""])[0]
            if item_title:
                results.append({"title": item_title, "source": "CrossRef"})
        return results

    def _search_semantic_scholar(
        self, author: str, year: int, title: str
    ) -> list[dict]:
        """Search Semantic Scholar. Returns list of {title, source} dicts.

        Tries year-filtered search first, then falls back to unfiltered
        if no results (many papers are indexed under different years).
        """
        results = []

        # Try with year filter first (±1 year)
        self._rate_limit()
        query = f"{author} {title}" if title else f"{author} {year}"
        year_range = f"{year - 1}-{year + 1}"
        params: dict = {"query": query.strip(), "limit": "5", "year": year_range}
        try:
            resp = httpx.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            for paper in data.get("data", [])[:5]:
                paper_title = paper.get("title", "")
                if paper_title:
                    results.append({"title": paper_title, "source": "Semantic Scholar"})
        except (httpx.HTTPError, httpx.TimeoutException):
            pass

        # If no results with year filter, try without
        if not results and title:
            self._rate_limit()
            params_no_year: dict = {"query": f"{author} {title}".strip(), "limit": "5"}
            try:
                resp = httpx.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params=params_no_year,
                    timeout=15.0,
                )
                resp.raise_for_status()
                data = resp.json()
                for paper in data.get("data", [])[:5]:
                    paper_title = paper.get("title", "")
                    if paper_title:
                        results.append({"title": paper_title, "source": "Semantic Scholar"})
            except (httpx.HTTPError, httpx.TimeoutException):
                pass

        return results

    def _finding(
        self,
        claim: Claim,
        verdict: Verdict,
        explanation: str,
        expected: str | None = None,
        actual: str | None = None,
    ) -> Finding:
        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=verdict,
            level=VerificationLevel.DETERMINISTIC,
            explanation=explanation,
            expected=expected,
            actual=actual,
            timestamp=time.time(),
        )
