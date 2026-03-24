from __future__ import annotations

from loop_guard.models import Claim, ClaimType, Finding, NormalizedStep, Verdict, VerificationLevel
from loop_guard.verifiers.code_output import CodeOutputVerifier
from loop_guard.verifiers.citation import CitationVerifier
from loop_guard.verifiers.statistical import StatisticalVerifier
from loop_guard.verifiers.regression import RegressionVerifier
from loop_guard.verifiers.loop_trap import LoopTrapVerifier
from loop_guard.verifiers.metric import MetricVerifier


class VerificationEngine:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        sandbox_dir = self.config.get("sandbox_dir", "/tmp/loopguard_sandbox")
        timeout = self.config.get("timeout", 60)
        self.code_verifier = CodeOutputVerifier(sandbox_dir=sandbox_dir, timeout=timeout)
        self.citation_verifier = CitationVerifier()
        self.statistical_verifier = StatisticalVerifier()
        self.regression_verifier = RegressionVerifier()
        self.loop_trap_verifier = LoopTrapVerifier(
            similarity_threshold=self.config.get("similarity_threshold", 0.8),
            consecutive_limit=self.config.get("consecutive_limit", 3),
        )
        self.metric_verifier = MetricVerifier(sandbox_dir=sandbox_dir, timeout=timeout)
        self.step_history: list[NormalizedStep] = []

    def verify_step(self, step: NormalizedStep, claims: list[Claim]) -> list[Finding]:
        self.step_history.append(step)
        findings = []

        # Always run structural verifiers
        regression_findings = self.regression_verifier.verify(step)
        findings.extend(regression_findings)

        trap_finding = self.loop_trap_verifier.verify(step)
        if trap_finding:
            findings.append(trap_finding)

        # Route each claim to its verifier
        for claim in claims:
            finding = self._route_claim(claim)
            findings.append(finding)

        return findings

    def _route_claim(self, claim: Claim) -> Finding:
        match claim.claim_type:
            case ClaimType.CODE_OUTPUT:
                return self.code_verifier.verify(claim)
            case ClaimType.CITATION:
                return self.citation_verifier.verify(claim)
            case ClaimType.STATISTICAL:
                return self.statistical_verifier.verify(claim, self.step_history)
            case ClaimType.METRIC:
                # Check for impossible values first (e.g. accuracy > 100%)
                impossible = self.statistical_verifier._check_impossible_values(claim)
                if impossible:
                    return impossible
                return self.metric_verifier.verify(claim)
            case ClaimType.TEST_RESULT:
                return self.code_verifier.verify(claim)
            case ClaimType.FILE_STATE:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.SKIPPED,
                    level=VerificationLevel.RULE_BASED,
                    explanation="File state verified by RegressionVerifier at step level",
                )
            case _:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.SKIPPED,
                    level=VerificationLevel.LLM_ASSISTED,
                    explanation="No deterministic verifier available for this claim type",
                )
