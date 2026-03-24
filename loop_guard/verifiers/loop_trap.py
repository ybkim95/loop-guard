"""Detects agent stuck in retry loops."""

from __future__ import annotations

import difflib
import time
from collections import deque
from typing import Optional

from loop_guard.models import (
    Claim,
    ClaimType,
    Finding,
    NormalizedStep,
    VerificationLevel,
    Verdict,
)


class LoopTrapVerifier:
    """Detects when an agent is stuck repeating similar outputs."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        consecutive_limit: int = 3,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.consecutive_limit = consecutive_limit
        self._recent_outputs: deque[str] = deque(maxlen=20)

    def verify(self, step: NormalizedStep) -> Optional[Finding]:
        """Check whether the agent is stuck in a retry loop.

        Returns a Finding with RULE_VIOLATION if the last
        *consecutive_limit* outputs are all similar to one another.
        """
        self._recent_outputs.append(step.raw_output)

        if len(self._recent_outputs) < self.consecutive_limit:
            return None

        # Check the tail of the window for consecutive similar outputs
        tail = list(self._recent_outputs)[-self.consecutive_limit :]
        all_similar = True
        for i in range(1, len(tail)):
            ratio = difflib.SequenceMatcher(None, tail[0], tail[i]).ratio()
            if ratio < self.similarity_threshold:
                all_similar = False
                break

        if not all_similar:
            return None

        claim = Claim(
            claim_type=ClaimType.GENERAL,
            source_step=step.step_id,
            text="Agent produced repeated similar outputs",
        )

        return Finding(
            step_id=step.step_id,
            claim=claim,
            verdict=Verdict.RULE_VIOLATION,
            level=VerificationLevel.RULE_BASED,
            explanation=(
                f"Detected {self.consecutive_limit} consecutive outputs with "
                f">={self.similarity_threshold:.0%} similarity — possible retry loop."
            ),
            timestamp=time.time(),
        )
