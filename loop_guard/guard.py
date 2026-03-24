"""Main entry point for loop-guard."""

from __future__ import annotations

import time

from loop_guard.engine import VerificationEngine
from loop_guard.extractor import ClaimExtractor
from loop_guard.models import Finding, NormalizedStep
from loop_guard.reporter import Reporter


class LoopGuard:
    """Deterministic verification for autonomous agent loops.

    Usage:
        guard = LoopGuard()
        for i in range(num_steps):
            result = agent.run(task)
            findings = guard.step(output=result.text, code=result.code)
            for f in findings:
                print(f)
        guard.report(format="html")
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.extractor = ClaimExtractor(
            use_llm=self.config.get("use_llm_extraction", True),
            llm_model=self.config.get("extraction_model", "claude-haiku-4-5-20251001"),
        )
        self.engine = VerificationEngine(config=self.config)
        self.reporter = Reporter(
            verbosity=self.config.get("verbosity", "findings_only"),
        )
        self._step_counter = 0

    def step(
        self,
        output: str,
        step_id: int | None = None,
        code: str | None = None,
        files: list[str] | None = None,
        metadata: dict | None = None,
    ) -> list[Finding]:
        if step_id is None:
            step_id = self._step_counter
            self._step_counter += 1

        normalized = NormalizedStep(
            step_id=step_id,
            timestamp=time.time(),
            raw_output=output,
            code_executed=code,
            files_modified=files or [],
            metadata=metadata or {},
        )

        claims = self.extractor.extract(normalized)
        findings = self.engine.verify_step(normalized, claims)
        self.reporter.report_step(findings)
        return findings

    def report(self, format: str = "terminal", path: str | None = None) -> str | dict:
        if format == "json":
            output_path = path or "loopguard_report.json"
            return self.reporter.generate_json_report(output_path)
        elif format == "html":
            output_path = path or "loopguard_report.html"
            return self.reporter.generate_html_report(output_path)
        else:
            return self.reporter.summary()

    @property
    def findings(self) -> list[Finding]:
        return self.reporter.all_findings

    @property
    def summary(self) -> dict:
        return self.reporter.summary()
