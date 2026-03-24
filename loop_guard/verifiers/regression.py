"""Detects when an agent reverts a file to an earlier state."""

from __future__ import annotations

import difflib
import os
import time

from loop_guard.models import (
    Claim,
    ClaimType,
    Finding,
    NormalizedStep,
    Verdict,
    VerificationLevel,
)


class RegressionVerifier:
    """Detects file regressions — when a file reverts to a previous version."""

    def __init__(self) -> None:
        # file path -> list of (step_id, content) tuples
        self.file_snapshots: dict[str, list[tuple[int, str]]] = {}

    def verify(self, step: NormalizedStep) -> list[Finding]:
        """Check whether any modified files regressed to an older state."""
        findings: list[Finding] = []

        for file_path in step.files_modified:
            abs_path = os.path.abspath(file_path)

            try:
                with open(abs_path) as f:
                    current_content = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            if abs_path not in self.file_snapshots:
                self.file_snapshots[abs_path] = []

            history = self.file_snapshots[abs_path]

            # Compare against older versions (skip immediate predecessor)
            if len(history) >= 2:
                for old_step_id, old_content in history[:-1]:
                    ratio = difflib.SequenceMatcher(
                        None, current_content, old_content
                    ).ratio()
                    if ratio >= 0.95:
                        claim = Claim(
                            claim_type=ClaimType.FILE_STATE,
                            source_step=step.step_id,
                            text=f"File {file_path} may have regressed to step {old_step_id}",
                        )
                        findings.append(
                            Finding(
                                step_id=step.step_id,
                                claim=claim,
                                verdict=Verdict.RULE_VIOLATION,
                                level=VerificationLevel.RULE_BASED,
                                explanation=(
                                    f"File '{file_path}' at step {step.step_id} is "
                                    f"{ratio:.0%} similar to its content at step "
                                    f"{old_step_id}, suggesting a regression."
                                ),
                                expected="Forward progress on file content",
                                actual=f"Content reverted to step {old_step_id}",
                                timestamp=time.time(),
                            )
                        )
                        break  # one finding per file is enough

            # Record current version (keep last 50)
            history.append((step.step_id, current_content))
            if len(history) > 50:
                history[:] = history[-50:]

        return findings
