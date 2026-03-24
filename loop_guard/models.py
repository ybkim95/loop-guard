"""Core data models for loop-guard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClaimType(Enum):
    """Types of verifiable claims an agent can make."""

    CODE_OUTPUT = "code_output"  # "I ran X, got Y"
    METRIC = "metric"  # "accuracy is 94%"
    STATISTICAL = "statistical"  # "p < 0.05, significant"
    CITATION = "citation"  # "Smith et al. 2024 showed..."
    TEST_RESULT = "test_result"  # "all tests pass"
    FILE_STATE = "file_state"  # "I modified file X"
    GENERAL = "general"  # untyped claim (Layer 3 only)


class VerificationLevel(Enum):
    """Reliability tier of a verification check."""

    DETERMINISTIC = 1  # re-execution, API lookup: cannot be wrong
    RULE_BASED = 2  # pattern matching, sanity check: rarely wrong
    LLM_ASSISTED = 3  # soft flag: may be wrong, needs human review


class Verdict(Enum):
    """Outcome of a verification check."""

    VERIFIED_PASS = "verified_pass"  # deterministic check passed
    VERIFIED_FAIL = "verified_fail"  # deterministic check failed
    RULE_VIOLATION = "rule_violation"  # rule-based check flagged
    FLAG_FOR_REVIEW = "flag_for_review"  # LLM flag, uncertain
    SKIPPED = "skipped"  # could not verify (no verifier)


@dataclass
class NormalizedStep:
    """A single agent loop iteration, normalized across adapters."""

    step_id: int
    timestamp: float
    raw_output: str
    code_executed: Optional[str] = None
    files_modified: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Claim:
    """A single verifiable claim extracted from agent output."""

    claim_type: ClaimType
    source_step: int
    text: str
    verifiable: bool = False
    evidence: Optional[dict] = None


@dataclass
class Finding:
    """Result of verifying a single claim."""

    step_id: int
    claim: Claim
    verdict: Verdict
    level: VerificationLevel
    explanation: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        return {
            "step_id": self.step_id,
            "claim": {
                "type": self.claim.claim_type.value,
                "source_step": self.claim.source_step,
                "text": self.claim.text,
                "verifiable": self.claim.verifiable,
                "evidence": self.claim.evidence,
            },
            "verdict": self.verdict.value,
            "level": self.level.value,
            "explanation": self.explanation,
            "expected": self.expected,
            "actual": self.actual,
            "timestamp": self.timestamp,
        }
