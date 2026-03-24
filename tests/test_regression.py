"""Tests for RegressionVerifier."""

import os
import tempfile

from loop_guard.models import NormalizedStep, Verdict
from loop_guard.verifiers.regression import RegressionVerifier


class TestRegressionVerifier:
    def test_no_regression_forward_progress(self):
        v = RegressionVerifier()
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "test.py")

            # Step 0: create file v1
            with open(filepath, "w") as f:
                f.write("def hello():\n    return 'v1'\n")
            step0 = NormalizedStep(
                step_id=0, timestamp=0.0, raw_output="Created test.py",
                files_modified=[filepath],
            )
            findings = v.verify(step0)
            assert len(findings) == 0

            # Step 1: update to v2
            with open(filepath, "w") as f:
                f.write("def hello():\n    return 'v2'\n\ndef goodbye():\n    return 'bye'\n")
            step1 = NormalizedStep(
                step_id=1, timestamp=1.0, raw_output="Updated test.py",
                files_modified=[filepath],
            )
            findings = v.verify(step1)
            assert len(findings) == 0

            # Step 2: update to v3
            with open(filepath, "w") as f:
                f.write("def hello():\n    return 'v3'\n\ndef goodbye():\n    return 'bye'\n\ndef extra():\n    pass\n")
            step2 = NormalizedStep(
                step_id=2, timestamp=2.0, raw_output="Updated test.py",
                files_modified=[filepath],
            )
            findings = v.verify(step2)
            assert len(findings) == 0

    def test_detects_regression(self):
        v = RegressionVerifier()
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "test.py")

            # Step 0: v1
            v1_content = "def hello():\n    return 'version_1'\n"
            with open(filepath, "w") as f:
                f.write(v1_content)
            v.verify(NormalizedStep(
                step_id=0, timestamp=0.0, raw_output="v1",
                files_modified=[filepath],
            ))

            # Step 1: v2 (different)
            with open(filepath, "w") as f:
                f.write("def hello():\n    return 'version_2'\n\ndef new_func():\n    pass\n")
            v.verify(NormalizedStep(
                step_id=1, timestamp=1.0, raw_output="v2",
                files_modified=[filepath],
            ))

            # Step 2: revert to v1
            with open(filepath, "w") as f:
                f.write(v1_content)
            findings = v.verify(NormalizedStep(
                step_id=2, timestamp=2.0, raw_output="v1 again",
                files_modified=[filepath],
            ))

            assert len(findings) > 0
            assert findings[0].verdict == Verdict.RULE_VIOLATION
            assert "regression" in findings[0].explanation.lower() or "revert" in findings[0].explanation.lower()

    def test_no_findings_for_nonexistent_file(self):
        v = RegressionVerifier()
        step = NormalizedStep(
            step_id=0, timestamp=0.0, raw_output="Modified something",
            files_modified=["/nonexistent/path/to/file.py"],
        )
        findings = v.verify(step)
        assert len(findings) == 0

    def test_multiple_files(self):
        v = RegressionVerifier()
        with tempfile.TemporaryDirectory() as tmp:
            file_a = os.path.join(tmp, "a.py")
            file_b = os.path.join(tmp, "b.py")

            with open(file_a, "w") as f:
                f.write("a_v1")
            with open(file_b, "w") as f:
                f.write("b_v1")

            v.verify(NormalizedStep(
                step_id=0, timestamp=0.0, raw_output="init",
                files_modified=[file_a, file_b],
            ))

            # Only change file_a
            with open(file_a, "w") as f:
                f.write("a_v2 with completely different content that is long enough")
            v.verify(NormalizedStep(
                step_id=1, timestamp=1.0, raw_output="changed a",
                files_modified=[file_a],
            ))

            # file_b unchanged, should not trigger
            findings = v.verify(NormalizedStep(
                step_id=2, timestamp=2.0, raw_output="no change to b",
                files_modified=[file_b],
            ))
            assert all(f.verdict != Verdict.RULE_VIOLATION for f in findings)
