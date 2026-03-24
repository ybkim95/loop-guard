"""Tests for Claim Provenance Chain."""

from loop_guard.models import Claim, ClaimType, Finding, Verdict, VerificationLevel
from loop_guard.provenance import ProvenanceChain


def _make_claim(step: int, text: str, ctype: ClaimType = ClaimType.METRIC) -> Claim:
    return Claim(claim_type=ctype, source_step=step, text=text, verifiable=True)


def _make_finding(step: int, claim: Claim, verdict: Verdict = Verdict.VERIFIED_PASS) -> Finding:
    return Finding(
        step_id=step, claim=claim, verdict=verdict,
        level=VerificationLevel.DETERMINISTIC, explanation="test",
    )


class TestProvenanceChain:
    def test_record_simple(self):
        chain = ProvenanceChain()
        c = _make_claim(0, "accuracy = 94%")
        f = _make_finding(0, c)
        node = chain.record(0, c, f)
        assert node.step_id == 0
        assert len(chain.nodes) == 1

    def test_dependency_tracking(self):
        chain = ProvenanceChain()
        c0 = _make_claim(0, "accuracy = 94%")
        f0 = _make_finding(0, c0)
        chain.record(0, c0, f0)

        c1 = _make_claim(1, "Based on 94% accuracy")
        f1 = _make_finding(1, c1)
        node1 = chain.record(1, c1, f1, depends_on=[0])
        assert node1.depends_on == [0]

    def test_invalidation_cascades(self):
        chain = ProvenanceChain()

        # Step 0: base claim
        c0 = _make_claim(0, "accuracy = 94%")
        f0 = _make_finding(0, c0)
        chain.record(0, c0, f0)

        # Step 1: depends on step 0
        c1 = _make_claim(1, "Based on 94% accuracy, we conclude...")
        f1 = _make_finding(1, c1)
        chain.record(1, c1, f1, depends_on=[0])

        # Step 2: depends on step 1
        c2 = _make_claim(2, "Final recommendation based on conclusions")
        f2 = _make_finding(2, c2)
        chain.record(2, c2, f2, depends_on=[1])

        # Invalidate step 0
        tainted = chain.invalidate(0, "accuracy was actually 74%")
        assert len(tainted) >= 1
        assert any(f.step_id == 1 for f in tainted)

    def test_tainted_on_failed_dependency(self):
        chain = ProvenanceChain()

        c0 = _make_claim(0, "fake claim")
        f0 = _make_finding(0, c0, Verdict.VERIFIED_FAIL)
        chain.record(0, c0, f0)

        c1 = _make_claim(1, "depends on fake")
        f1 = _make_finding(1, c1)
        node1 = chain.record(1, c1, f1, depends_on=[0])
        assert node1.tainted is True

    def test_auto_detect_step_references(self):
        chain = ProvenanceChain()
        c0 = _make_claim(0, "accuracy = 94%")
        f0 = _make_finding(0, c0)
        chain.record(0, c0, f0)

        deps = chain.auto_detect_dependencies(1, "As shown in step 0, the accuracy was good")
        assert 0 in deps

    def test_auto_detect_previous_reference(self):
        chain = ProvenanceChain()
        c0 = _make_claim(0, "result A")
        f0 = _make_finding(0, c0)
        chain.record(0, c0, f0)

        deps = chain.auto_detect_dependencies(1, "Based on the previous result")
        assert 0 in deps

    def test_get_tainted_steps(self):
        chain = ProvenanceChain()
        c0 = _make_claim(0, "wrong")
        f0 = _make_finding(0, c0, Verdict.VERIFIED_FAIL)
        chain.record(0, c0, f0)

        c1 = _make_claim(1, "depends on wrong")
        f1 = _make_finding(1, c1)
        chain.record(1, c1, f1, depends_on=[0])

        chain.invalidate(0)
        tainted = chain.get_tainted_steps()
        assert 0 in tainted
        assert 1 in tainted

    def test_summary(self):
        chain = ProvenanceChain()
        c0 = _make_claim(0, "base")
        f0 = _make_finding(0, c0)
        chain.record(0, c0, f0)
        c1 = _make_claim(1, "derived")
        f1 = _make_finding(1, c1)
        chain.record(1, c1, f1, depends_on=[0])

        s = chain.summary()
        assert s["total_claims_tracked"] == 2
        assert s["total_dependency_edges"] == 1
        assert s["steps_with_dependencies"] == 1

    def test_no_dependencies(self):
        chain = ProvenanceChain()
        for i in range(5):
            c = _make_claim(i, f"independent claim {i}")
            f = _make_finding(i, c)
            chain.record(i, c, f)
        s = chain.summary()
        assert s["total_dependency_edges"] == 0
        assert s["tainted_steps"] == 0
