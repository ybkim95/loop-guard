"""Tests for LoopTrapVerifier."""

from loop_guard.models import NormalizedStep, Verdict, VerificationLevel
from loop_guard.verifiers.loop_trap import LoopTrapVerifier


def _make_step(step_id: int, output: str) -> NormalizedStep:
    return NormalizedStep(step_id=step_id, timestamp=0.0, raw_output=output)


class TestLoopTrapVerifier:
    def test_no_trap_different_outputs(self):
        v = LoopTrapVerifier()
        diverse_outputs = [
            "Loading the dataset from disk and preprocessing all columns into numeric features",
            "The gradient descent optimizer converged after 200 epochs with final loss 0.0032",
            "Switching to a completely different architecture: random forest with 500 trees and max depth 12",
            "Evaluating on the holdout test set produced accuracy=91.5%, precision=0.88, recall=0.94",
            "Generating final PDF report with all charts, tables, and statistical summaries included",
        ]
        for i, output in enumerate(diverse_outputs):
            finding = v.verify(_make_step(i, output))
            assert finding is None

    def test_detects_stuck_loop(self):
        v = LoopTrapVerifier(similarity_threshold=0.8, consecutive_limit=3)
        repeated = "Error: connection timed out. Retrying request to API endpoint."

        findings = []
        for i in range(5):
            f = v.verify(_make_step(i, repeated))
            if f is not None:
                findings.append(f)

        assert len(findings) > 0
        assert findings[0].verdict == Verdict.RULE_VIOLATION
        assert findings[0].level == VerificationLevel.RULE_BASED
        assert any(w in findings[0].explanation.lower() for w in ("stuck", "retry", "consecutive", "similar"))

    def test_similar_but_not_identical(self):
        v = LoopTrapVerifier(similarity_threshold=0.8, consecutive_limit=3)
        base = "Running experiment with learning rate 0.001. Loss: {:.4f}. Accuracy: {:.2f}%"

        for i in range(4):
            # Slightly different numbers each time, but very similar structure
            f = v.verify(_make_step(i, base.format(0.5 + i * 0.001, 85.0 + i * 0.01)))

        # With very similar outputs, should eventually trigger
        # The exact behavior depends on similarity threshold

    def test_resets_after_different_output(self):
        v = LoopTrapVerifier(similarity_threshold=0.8, consecutive_limit=3)
        repeated = "Same error message repeated"

        v.verify(_make_step(0, repeated))
        v.verify(_make_step(1, repeated))
        # Break the chain
        v.verify(_make_step(2, "Completely different output with new information"))
        # Start again
        f = v.verify(_make_step(3, repeated))
        assert f is None  # chain was broken

    def test_custom_threshold(self):
        v = LoopTrapVerifier(similarity_threshold=0.95, consecutive_limit=2)
        # These are similar but not 95% similar
        v.verify(_make_step(0, "Error in module A: timeout after 30s"))
        v.verify(_make_step(1, "Error in module B: timeout after 45s"))
        f = v.verify(_make_step(2, "Error in module C: timeout after 60s"))
        # Should not trigger because similarity < 0.95
        assert f is None

    def test_sliding_window_limit(self):
        v = LoopTrapVerifier()
        # Feed 25 unique outputs to exceed the 20-item window
        for i in range(25):
            v.verify(_make_step(i, f"Completely unique output {i} with random data {i**2}"))
        assert len(v._recent_outputs) <= 20

    def test_empty_output(self):
        v = LoopTrapVerifier()
        f = v.verify(_make_step(0, ""))
        assert f is None
