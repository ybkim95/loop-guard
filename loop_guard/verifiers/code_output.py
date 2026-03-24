"""Re-executes code in a subprocess sandbox to verify claimed output."""

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


class CodeOutputVerifier:
    """Verifies code-output claims by re-executing code in a sandbox."""

    def __init__(
        self,
        sandbox_dir: str = "/tmp/loopguard_sandbox",
        timeout: int = 60,
    ) -> None:
        self.sandbox_dir = sandbox_dir
        self.timeout = timeout
        os.makedirs(self.sandbox_dir, exist_ok=True)

    def verify(self, claim: Claim) -> Finding:
        """Verify a code-output claim by re-running the code."""
        if not claim.evidence:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation="No evidence provided (need 'code' and 'claimed_output').",
                timestamp=time.time(),
            )

        code = claim.evidence.get("code", "")
        claimed_output = claim.evidence.get("claimed_output", "")

        if not code:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation="No code provided in evidence.",
                timestamp=time.time(),
            )

        actual_output, success = self._execute_in_sandbox(code)

        if not success:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_FAIL,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Code execution failed: {actual_output}",
                expected=claimed_output,
                actual=actual_output,
                timestamp=time.time(),
            )

        if self._outputs_match(claimed_output, actual_output):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation="Re-execution output matches claimed output.",
                expected=claimed_output,
                actual=actual_output,
                timestamp=time.time(),
            )

        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.VERIFIED_FAIL,
            level=VerificationLevel.DETERMINISTIC,
            explanation="Re-execution output does not match claimed output.",
            expected=claimed_output,
            actual=actual_output,
            timestamp=time.time(),
        )

    def _execute_in_sandbox(self, code: str) -> tuple[str, bool]:
        """Execute code in a sandboxed subprocess.

        Returns (output, success) where output is stdout on success or
        stderr on failure.
        """
        os.makedirs(self.sandbox_dir, exist_ok=True)

        try:
            fd, script_path = tempfile.mkstemp(
                suffix=".py", dir=self.sandbox_dir
            )
            with os.fdopen(fd, "w") as f:
                f.write(code)

            env = os.environ.copy()
            env["HOME"] = self.sandbox_dir
            env["TMPDIR"] = self.sandbox_dir

            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.sandbox_dir,
                env=env,
            )

            if result.returncode == 0:
                return result.stdout, True
            else:
                return result.stderr or result.stdout, False

        except subprocess.TimeoutExpired:
            return f"Execution timed out after {self.timeout}s", False
        except Exception as exc:
            return str(exc), False
        finally:
            try:
                os.unlink(script_path)
            except (OSError, UnboundLocalError):
                pass

    @staticmethod
    def _outputs_match(
        claimed: str, actual: str, tolerance: float = 0.01
    ) -> bool:
        """Compare claimed and actual outputs.

        Tries numeric comparison first, then falls back to normalized
        string comparison.
        """
        claimed_stripped = claimed.strip()
        actual_stripped = actual.strip()

        # Try numeric comparison
        try:
            claimed_val = float(claimed_stripped)
            actual_val = float(actual_stripped)
            if actual_val == 0 and claimed_val == 0:
                return True
            if actual_val == 0:
                return abs(claimed_val) < tolerance
            return abs((claimed_val - actual_val) / actual_val) <= tolerance
        except (ValueError, ZeroDivisionError):
            pass

        # Fall back to normalized string comparison
        def normalize(s: str) -> str:
            return " ".join(s.split())

        return normalize(claimed_stripped) == normalize(actual_stripped)
