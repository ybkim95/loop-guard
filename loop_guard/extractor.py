"""Claim extraction from agent output using regex and optional LLM."""

from __future__ import annotations

import json
import re
from typing import Optional

from loop_guard.models import Claim, ClaimType, NormalizedStep

# ---------------------------------------------------------------------------
# Regex patterns for deterministic extraction
# ---------------------------------------------------------------------------

CITATION_PATTERN = re.compile(
    r'([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?\s*[\s,]+(?:19|20)\d{2})'
)

METRIC_PATTERN = re.compile(
    r'((?:accuracy|precision|recall|f1|loss|val_bpb|auc|rmse|mae|mse|r2|bleu|rouge)'
    r'[\s:=]+[\d.]+%?'
    r'|(?:accuracy|precision|recall|f1|loss|auc|rmse|mae)\s+(?:of|is|was|at)\s+[\d.]+%?)',
    re.IGNORECASE,
)

PVALUE_PATTERN = re.compile(
    r'(p[\s]*[<>=]+[\s]*[-+]?[\d.]+(?:e-?\d+)?)'
)

TEST_PATTERN = re.compile(
    r'((?:all\s+)?tests?\s+(?:pass|fail|passed|failed|passing|failing)'
    r'|(\d+)\s+(?:passed|failed|tests?\s+passed|tests?\s+failed))',
    re.IGNORECASE,
)

FILE_STATE_PATTERN = re.compile(
    r'(?:modified|created|deleted|updated|wrote|changed|edited)'
    r'\s+[`"\']?([^\s`"\']+\.\w+)[`"\']?',
    re.IGNORECASE,
)

# Map patterns to claim types and whether the claim is deterministically verifiable.
_REGEX_RULES: list[tuple[re.Pattern, ClaimType, bool]] = [
    (CITATION_PATTERN, ClaimType.CITATION, False),
    (METRIC_PATTERN, ClaimType.METRIC, True),
    (PVALUE_PATTERN, ClaimType.STATISTICAL, True),
    (TEST_PATTERN, ClaimType.TEST_RESULT, True),
    (FILE_STATE_PATTERN, ClaimType.FILE_STATE, True),
]

# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "You extract verifiable claims from AI-agent output. "
    "Return a JSON array of objects with keys: "
    '"claim_type" (one of: code_output, metric, statistical, citation, '
    'test_result, file_state, general), "text" (the verbatim claim), '
    '"verifiable" (boolean). '
    "Only include claims not already covered by the provided existing_claims list. "
    "If there are no new claims, return an empty array []."
)


class ClaimExtractor:
    """Extract verifiable claims from a :class:`NormalizedStep`.

    Regex-based extraction runs first and is always available.  An optional
    LLM pass (using the ``anthropic`` SDK) can pick up remaining claims that
    the regex rules miss.
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.use_llm = use_llm
        self.llm_model = llm_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown formatting that interferes with regex extraction."""
        # Remove bold/italic markers
        text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
        text = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', text)
        # Remove inline code backticks
        text = re.sub(r'`([^`]+)`', r'\1', text)
        return text

    def extract(self, step: NormalizedStep) -> list[Claim]:
        """Extract claims from *step* output using regex first, then optional LLM."""
        if not step.raw_output or not step.raw_output.strip():
            return []

        # Strip markdown formatting before extraction
        clean_step = NormalizedStep(
            step_id=step.step_id,
            timestamp=step.timestamp,
            raw_output=self._strip_markdown(step.raw_output),
            code_executed=step.code_executed,
            files_modified=step.files_modified,
            metadata=step.metadata,
        )

        claims: list[Claim] = []
        claims.extend(self._extract_regex(clean_step))

        if self.use_llm and self._has_unmatched_content(step.raw_output, claims):
            claims.extend(self._extract_llm(step))

        return claims

    # ------------------------------------------------------------------
    # Regex extraction
    # ------------------------------------------------------------------

    def _extract_regex(self, step: NormalizedStep) -> list[Claim]:
        claims: list[Claim] = []
        text = step.raw_output

        for pattern, claim_type, verifiable in _REGEX_RULES:
            for match in pattern.finditer(text):
                # group(1) holds the main capture; fall back to group(0)
                matched_text = match.group(1) if match.lastindex and match.group(1) else match.group(0)
                claims.append(
                    Claim(
                        claim_type=claim_type,
                        source_step=step.step_id,
                        text=matched_text.strip(),
                        verifiable=verifiable,
                    )
                )

        return claims

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _extract_llm(self, step: NormalizedStep) -> list[Claim]:
        """Use the Anthropic API to extract claims the regex missed."""
        try:
            import anthropic  # noqa: F811
        except ImportError:
            return []

        existing_texts = [
            c.text for c in self._extract_regex(step)
        ]

        user_message = (
            f"Existing claims already extracted (do not duplicate):\n"
            f"{json.dumps(existing_texts)}\n\n"
            f"Agent output to analyze:\n{step.raw_output}"
        )

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.llm_model,
                max_tokens=1024,
                system=_LLM_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            # Parse the text block(s) from the response.
            raw = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return self._parse_llm_response(raw, step.step_id)
        except Exception:
            # Network errors, auth issues, malformed response, etc.
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_unmatched_content(text: str, claims: list[Claim]) -> bool:
        """Return True if *text* likely contains claims not yet captured."""
        if not claims:
            # No regex hits at all -- worth trying LLM if text is non-trivial.
            return len(text.strip()) > 20

        # Remove all matched spans from the text and check what remains.
        remaining = text
        for claim in claims:
            remaining = remaining.replace(claim.text, "", 1)

        # Heuristic: if the remaining text (after stripping whitespace/punctuation)
        # still has a meaningful amount of content, there may be more claims.
        remaining_stripped = re.sub(r'[\s\W]+', ' ', remaining).strip()
        return len(remaining_stripped) > 40

    @staticmethod
    def _parse_llm_response(raw: str, source_step: int) -> list[Claim]:
        """Parse LLM JSON response into :class:`Claim` objects."""
        # Try to find a JSON array in the response text.
        # The LLM may wrap the array in markdown code fences.
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$', '', cleaned.strip(), flags=re.MULTILINE)

        try:
            items = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return []

        if not isinstance(items, list):
            return []

        _type_map = {t.value: t for t in ClaimType}
        claims: list[Claim] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            claim_type_str = item.get("claim_type", "general")
            claim_type = _type_map.get(claim_type_str, ClaimType.GENERAL)
            text = item.get("text", "")
            if not text:
                continue
            claims.append(
                Claim(
                    claim_type=claim_type,
                    source_step=source_step,
                    text=text,
                    verifiable=bool(item.get("verifiable", False)),
                )
            )

        return claims
