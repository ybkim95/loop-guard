"""Integration tests for the full LoopGuard pipeline."""

import json
import os
import tempfile

from loop_guard.guard import LoopGuard
from loop_guard.models import Verdict


class TestLoopGuardIntegration:
    def test_basic_step_processing(self):
        guard = LoopGuard(config={"use_llm_extraction": False, "verbosity": "all"})
        findings = guard.step(output="Model accuracy = 95.2% on validation set.")
        assert isinstance(findings, list)

    def test_loop_trap_detection(self):
        guard = LoopGuard(config={
            "use_llm_extraction": False,
            "verbosity": "all",
            "consecutive_limit": 3,
        })
        repeated = "Error: API rate limit exceeded. Retrying in 5 seconds..."

        all_findings = []
        for i in range(5):
            findings = guard.step(output=repeated, step_id=i)
            all_findings.extend(findings)

        violations = [f for f in all_findings if f.verdict == Verdict.RULE_VIOLATION]
        assert len(violations) > 0
        assert any("retry" in f.explanation.lower() or "stuck" in f.explanation.lower() or "consecutive" in f.explanation.lower() for f in violations)

    def test_statistical_impossible_value(self):
        guard = LoopGuard(config={"use_llm_extraction": False, "verbosity": "all"})
        findings = guard.step(output="The test showed p = 2.5, which is significant.")
        violations = [f for f in findings if f.verdict == Verdict.RULE_VIOLATION]
        assert len(violations) > 0

    def test_multiple_steps_with_metrics(self):
        guard = LoopGuard(config={"use_llm_extraction": False})

        guard.step(output="Epoch 1: loss = 2.5, accuracy = 45.2%", step_id=0)
        guard.step(output="Epoch 2: loss = 1.8, accuracy = 62.1%", step_id=1)
        guard.step(output="Epoch 3: loss = 1.2, accuracy = 78.5%", step_id=2)

        summary = guard.summary
        assert summary["total_claims_checked"] > 0

    def test_json_report(self):
        guard = LoopGuard(config={"use_llm_extraction": False})
        guard.step(output="accuracy = 90%", step_id=0)
        guard.step(output="p = 1.5 is significant", step_id=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            guard.report(format="json", path=path)
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)
            assert "summary" in data
            assert "findings" in data
            assert len(data["findings"]) > 0

    def test_html_report(self):
        guard = LoopGuard(config={"use_llm_extraction": False})
        guard.step(output="accuracy = 90%", step_id=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            guard.report(format="html", path=path)
            assert os.path.exists(path)

            with open(path) as f:
                html = f.read()
            assert "LoopGuard" in html
            assert "<!DOCTYPE html>" in html

    def test_summary_statistics(self):
        guard = LoopGuard(config={"use_llm_extraction": False})
        guard.step(output="accuracy = 95%", step_id=0)
        guard.step(output="p = -0.5", step_id=1)

        summary = guard.summary
        assert "total_claims_checked" in summary
        assert "verified_failures" in summary
        assert "rule_violations" in summary
        assert "verified_passes" in summary
        assert "skipped" in summary

    def test_step_auto_increment(self):
        guard = LoopGuard(config={"use_llm_extraction": False})
        guard.step(output="Step one accuracy = 90%")
        guard.step(output="Step two accuracy = 92%")
        guard.step(output="Step three accuracy = 94%")

        # Check that step IDs were auto-assigned 0, 1, 2
        step_ids = {f.step_id for f in guard.findings}
        assert 0 in step_ids
        assert 1 in step_ids or 2 in step_ids

    def test_regression_detection_integration(self):
        guard = LoopGuard(config={"use_llm_extraction": False, "verbosity": "all"})

        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "model.py")

            # Step 0: initial version
            v1 = "class Model:\n    def __init__(self):\n        self.lr = 0.001\n"
            with open(filepath, "w") as f:
                f.write(v1)
            guard.step(output="Created model.py", step_id=0, files=[filepath])

            # Step 1: improved version
            with open(filepath, "w") as f:
                f.write("class Model:\n    def __init__(self):\n        self.lr = 0.0005\n    def train(self):\n        pass\n")
            guard.step(output="Updated model.py", step_id=1, files=[filepath])

            # Step 2: revert to v1
            with open(filepath, "w") as f:
                f.write(v1)
            findings = guard.step(output="Fixed model.py", step_id=2, files=[filepath])

            violations = [f for f in findings if f.verdict == Verdict.RULE_VIOLATION]
            assert len(violations) > 0

    def test_mixed_claim_types(self):
        guard = LoopGuard(config={"use_llm_extraction": False})
        output = (
            "Experiment results:\n"
            "- Based on Smith et al. 2024, we used a transformer architecture\n"
            "- accuracy = 94.2%\n"
            "- loss: 0.15\n"
            "- All tests passed\n"
            "- p < 0.001\n"
        )
        findings = guard.step(output=output, step_id=0)
        # Should extract multiple claim types
        assert len(findings) > 0

    def test_findings_property(self):
        guard = LoopGuard(config={"use_llm_extraction": False})
        guard.step(output="accuracy = 90%")
        assert len(guard.findings) > 0
        assert guard.findings is guard.reporter.all_findings

    def test_config_passthrough(self):
        guard = LoopGuard(config={
            "use_llm_extraction": False,
            "verbosity": "failures_only",
            "sandbox_dir": "/tmp/custom_sandbox",
            "timeout": 30,
        })
        assert guard.reporter.verbosity == "failures_only"
        assert guard.engine.config.get("sandbox_dir") == "/tmp/custom_sandbox"
