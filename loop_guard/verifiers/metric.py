"""Re-computes claimed metrics in a subprocess sandbox."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

from loop_guard.models import (
    Claim,
    Finding,
    Verdict,
    VerificationLevel,
)


class MetricVerifier:
    """Verifies metric claims by re-computing them in a sandbox."""

    def __init__(
        self,
        sandbox_dir: str = "/tmp/loopguard_sandbox",
        timeout: int = 60,
    ) -> None:
        self.sandbox_dir = sandbox_dir
        self.timeout = timeout
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def verify(self, claim: Claim) -> Finding:
        """Verify a metric claim by re-executing the computation."""
        if not claim.evidence:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation="No evidence provided (need 'metric_name', 'claimed_value', 'code').",
                timestamp=time.time(),
            )

        metric_name = claim.evidence.get("metric_name")
        claimed_value = claim.evidence.get("claimed_value")
        code = claim.evidence.get("code")

        if not all([metric_name, claimed_value is not None, code]):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation=(
                    f"Missing evidence fields. Need 'metric_name', 'claimed_value', "
                    f"and 'code'. Got: {list(claim.evidence.keys())}"
                ),
                timestamp=time.time(),
            )

        try:
            claimed_float = float(claimed_value)
        except (TypeError, ValueError):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Cannot parse claimed_value as float: {claimed_value}",
                timestamp=time.time(),
            )

        actual = self._recompute(code, self.sandbox_dir)

        if actual is None:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_FAIL,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Re-computation of '{metric_name}' failed to produce a numeric result.",
                expected=str(claimed_float),
                actual="<execution error>",
                timestamp=time.time(),
            )

        if self._values_match(claimed_float, actual):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Metric '{metric_name}' re-computed successfully and matches.",
                expected=str(claimed_float),
                actual=str(actual),
                timestamp=time.time(),
            )

        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.VERIFIED_FAIL,
            level=VerificationLevel.DETERMINISTIC,
            explanation=(
                f"Metric '{metric_name}' mismatch: claimed {claimed_float}, "
                f"re-computed {actual}."
            ),
            expected=str(claimed_float),
            actual=str(actual),
            timestamp=time.time(),
        )

    def _recompute(self, code: str, sandbox_dir: str) -> float | None:
        """Execute code in sandbox and parse the last line as a float."""
        os.makedirs(sandbox_dir, exist_ok=True)

        try:
            fd, script_path = tempfile.mkstemp(
                suffix=".py", dir=sandbox_dir
            )
            with os.fdopen(fd, "w") as f:
                f.write(code)

            env = os.environ.copy()
            env["HOME"] = sandbox_dir
            env["TMPDIR"] = sandbox_dir

            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=sandbox_dir,
                env=env,
            )

            if result.returncode != 0:
                return None

            output = result.stdout.strip()
            if not output:
                return None

            # Parse the last line as a float
            last_line = output.splitlines()[-1].strip()
            return float(last_line)

        except (subprocess.TimeoutExpired, ValueError, Exception):
            return None
        finally:
            try:
                os.unlink(script_path)
            except (OSError, UnboundLocalError):
                pass

    @staticmethod
    def _values_match(
        claimed: float, actual: float, rtol: float = 0.01
    ) -> bool:
        """Compare two floats with relative tolerance."""
        if actual == 0 and claimed == 0:
            return True
        if actual == 0:
            return abs(claimed) < rtol
        return abs((claimed - actual) / actual) <= rtol
