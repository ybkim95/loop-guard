"""loop-guard: Deterministic verification for autonomous agent loops."""

from loop_guard.guard import LoopGuard
from loop_guard.models import (
    Claim,
    ClaimType,
    Finding,
    NormalizedStep,
    Verdict,
    VerificationLevel,
)

__version__ = "0.1.0"

__all__ = [
    "LoopGuard",
    "Claim",
    "ClaimType",
    "Finding",
    "NormalizedStep",
    "Verdict",
    "VerificationLevel",
]
