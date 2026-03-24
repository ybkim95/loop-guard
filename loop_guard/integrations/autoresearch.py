"""Autoresearch integration for loop-guard.

Monitors Karpathy's autoresearch experiments by watching results.tsv
and git commits. Detects:
- Plateau: last N experiments show no improvement in val_bpb
- Crash loops: repeated OOM or convergence failures
- Convergence stall: agent retrying same approach
- Metric anomalies: impossible val_bpb values, NaN, suspiciously good results
- Success rate degradation: keep/discard ratio dropping

Usage:
    from loop_guard.integrations.autoresearch import AutoresearchGuard

    guard = AutoresearchGuard("./autoresearch/")
    guard.watch()  # blocks, monitoring results.tsv
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from loop_guard.guard import LoopGuard
from loop_guard.models import (
    Claim,
    ClaimType,
    Finding,
    Verdict,
    VerificationLevel,
)


@dataclass
class ExperimentRecord:
    """A single row from results.tsv."""

    commit: str = ""
    val_bpb: float = 0.0
    memory_gb: float = 0.0
    status: str = ""  # keep, discard, crash
    description: str = ""
    step_index: int = 0


class AutoresearchGuard:
    """Specialized LoopGuard for Karpathy's autoresearch.

    Monitors results.tsv and applies autoresearch-specific verification:
    plateau detection, crash loop detection, convergence monitoring.
    """

    def __init__(
        self,
        project_dir: str,
        plateau_window: int = 10,
        plateau_threshold: float = 0.0001,
        crash_limit: int = 5,
        similarity_threshold: float = 0.85,
        on_finding: Callable[[Finding], None] | None = None,
    ):
        self.project_dir = Path(project_dir)
        self.results_path = self.project_dir / "results.tsv"
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.crash_limit = crash_limit
        self.similarity_threshold = similarity_threshold
        self.on_finding = on_finding or self._default_on_finding

        self.guard = LoopGuard(config={
            "use_llm_extraction": False,
            "verbosity": "findings_only",
        })
        self.experiments: list[ExperimentRecord] = []
        self._last_line_count = 0

    def watch(self, poll_interval: float = 30.0) -> None:
        """Watch results.tsv for new experiments. Blocks until Ctrl+C."""
        print(f"[loop-guard:autoresearch] Watching {self.results_path}")
        print(f"[loop-guard:autoresearch] Plateau window: {self.plateau_window} experiments")
        print(f"[loop-guard:autoresearch] Poll interval: {poll_interval}s")
        print()

        try:
            while True:
                new_experiments = self._read_new_experiments()
                for exp in new_experiments:
                    findings = self._verify_experiment(exp)
                    for f in findings:
                        self.on_finding(f)
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[loop-guard:autoresearch] Stopped.")
            self._print_summary()

    def check(self) -> list[Finding]:
        """One-shot check of the entire results.tsv. Returns all findings."""
        all_findings = []
        experiments = self._read_all_experiments()
        for exp in experiments:
            findings = self._verify_experiment(exp)
            all_findings.extend(findings)
        return all_findings

    def _verify_experiment(self, exp: ExperimentRecord) -> list[Finding]:
        """Run all autoresearch-specific checks on one experiment."""
        self.experiments.append(exp)
        findings = []

        # 1. Metric sanity check
        f = self._check_metric_sanity(exp)
        if f:
            findings.append(f)

        # 2. Crash loop detection
        f = self._check_crash_loop(exp)
        if f:
            findings.append(f)

        # 3. Plateau detection
        f = self._check_plateau(exp)
        if f:
            findings.append(f)

        # 4. Convergence stall (same description repeated)
        f = self._check_convergence_stall(exp)
        if f:
            findings.append(f)

        # 5. Success rate degradation
        f = self._check_success_rate(exp)
        if f:
            findings.append(f)

        # Also run through the standard LoopGuard pipeline
        step_output = (
            f"Experiment {exp.step_index}: {exp.description}\n"
            f"val_bpb = {exp.val_bpb}\n"
            f"status: {exp.status}\n"
            f"memory: {exp.memory_gb} GB"
        )
        standard_findings = self.guard.step(
            output=step_output,
            step_id=exp.step_index,
        )
        # Filter out loop trap findings (we have our own convergence detector)
        # and PASS/SKIP findings
        findings.extend([
            f for f in standard_findings
            if f.verdict not in (Verdict.VERIFIED_PASS, Verdict.SKIPPED)
            and "consecutive" not in f.explanation.lower()
            and "retry loop" not in f.explanation.lower()
        ])

        return findings

    def _check_metric_sanity(self, exp: ExperimentRecord) -> Finding | None:
        """Check for impossible or suspicious val_bpb values."""
        if exp.status == "crash":
            return None  # crashed experiments have val_bpb = 0

        if exp.val_bpb <= 0:
            return self._make_finding(
                exp, Verdict.VERIFIED_FAIL, VerificationLevel.DETERMINISTIC,
                f"val_bpb = {exp.val_bpb} (must be > 0 for non-crash experiments)",
                expected="val_bpb > 0", actual=str(exp.val_bpb),
            )

        if exp.val_bpb > 10.0:
            return self._make_finding(
                exp, Verdict.RULE_VIOLATION, VerificationLevel.RULE_BASED,
                f"Suspiciously high val_bpb = {exp.val_bpb} (typical range: 0.8-2.0)",
                expected="val_bpb in [0.5, 5.0]", actual=str(exp.val_bpb),
            )

        # Check for suspiciously good improvement (>10% in one step)
        if len(self.experiments) >= 2:
            prev = self._last_kept()
            if prev and prev.val_bpb > 0:
                improvement = (prev.val_bpb - exp.val_bpb) / prev.val_bpb
                if improvement > 0.10 and exp.status == "keep":
                    return self._make_finding(
                        exp, Verdict.FLAG_FOR_REVIEW, VerificationLevel.RULE_BASED,
                        f"Unusually large improvement: {improvement:.1%} in one experiment "
                        f"(val_bpb {prev.val_bpb:.6f} → {exp.val_bpb:.6f}). Verify this is real.",
                    )

        return None

    def _check_crash_loop(self, exp: ExperimentRecord) -> Finding | None:
        """Detect repeated crashes."""
        if exp.status != "crash":
            return None

        recent = self.experiments[-self.crash_limit:]
        crash_count = sum(1 for e in recent if e.status == "crash")

        if crash_count >= self.crash_limit:
            return self._make_finding(
                exp, Verdict.RULE_VIOLATION, VerificationLevel.RULE_BASED,
                f"Crash loop: {crash_count}/{len(recent)} recent experiments crashed. "
                f"Agent may be stuck trying approaches that exceed memory.",
            )

        return None

    def _check_plateau(self, exp: ExperimentRecord) -> Finding | None:
        """Detect when agent is no longer improving."""
        kept = [e for e in self.experiments if e.status == "keep"]
        if len(kept) < self.plateau_window:
            return None

        recent_kept = kept[-self.plateau_window:]
        best_old = min(e.val_bpb for e in recent_kept[:len(recent_kept)//2])
        best_new = min(e.val_bpb for e in recent_kept[len(recent_kept)//2:])
        improvement = best_old - best_new

        if improvement < self.plateau_threshold:
            return self._make_finding(
                exp, Verdict.FLAG_FOR_REVIEW, VerificationLevel.RULE_BASED,
                f"Plateau detected: last {self.plateau_window} kept experiments show "
                f"only {improvement:.6f} improvement in val_bpb "
                f"(threshold: {self.plateau_threshold}). Consider changing strategy.",
            )

        return None

    def _check_convergence_stall(self, exp: ExperimentRecord) -> Finding | None:
        """Detect when agent keeps trying similar approaches."""
        if len(self.experiments) < 4:
            return None

        recent = self.experiments[-4:]
        descriptions = [e.description for e in recent]
        similarities = []
        for i in range(len(descriptions)):
            for j in range(i + 1, len(descriptions)):
                sim = SequenceMatcher(None, descriptions[i], descriptions[j]).ratio()
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        if avg_similarity > self.similarity_threshold:
            return self._make_finding(
                exp, Verdict.RULE_VIOLATION, VerificationLevel.RULE_BASED,
                f"Convergence stall: last {len(recent)} experiment descriptions are "
                f"{avg_similarity:.0%} similar. Agent is repeating the same approach.",
            )

        return None

    def _check_success_rate(self, exp: ExperimentRecord) -> Finding | None:
        """Detect degrading success rate."""
        if len(self.experiments) < 20:
            return None

        recent = self.experiments[-20:]
        kept = sum(1 for e in recent if e.status == "keep")
        rate = kept / len(recent)

        # Normal rate for autoresearch is ~30-50% keep
        if rate < 0.05:  # <5% success rate
            return self._make_finding(
                exp, Verdict.FLAG_FOR_REVIEW, VerificationLevel.RULE_BASED,
                f"Very low success rate: only {kept}/{len(recent)} ({rate:.0%}) "
                f"experiments kept in last 20. Agent may be exploring unproductively.",
            )

        return None

    def _last_kept(self) -> ExperimentRecord | None:
        """Get the last kept experiment."""
        for exp in reversed(self.experiments[:-1]):
            if exp.status == "keep":
                return exp
        return None

    def _read_new_experiments(self) -> list[ExperimentRecord]:
        """Read new lines from results.tsv since last check."""
        if not self.results_path.exists():
            return []

        all_lines = self.results_path.read_text().strip().split("\n")
        if len(all_lines) <= self._last_line_count:
            return []

        new_lines = all_lines[self._last_line_count:]
        self._last_line_count = len(all_lines)

        records = []
        for i, line in enumerate(new_lines):
            if line.startswith("commit") or not line.strip():
                continue  # skip header
            record = self._parse_tsv_line(line, self._last_line_count - len(new_lines) + i)
            if record:
                records.append(record)

        return records

    def _read_all_experiments(self) -> list[ExperimentRecord]:
        """Read all experiments from results.tsv."""
        if not self.results_path.exists():
            return []

        records = []
        with open(self.results_path) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header:
                return []
            for i, row in enumerate(reader):
                record = self._parse_row(row, i)
                if record:
                    records.append(record)

        return records

    def _parse_tsv_line(self, line: str, index: int) -> ExperimentRecord | None:
        """Parse a single TSV line into an ExperimentRecord."""
        parts = line.split("\t")
        return self._parse_row(parts, index)

    def _parse_row(self, parts: list[str], index: int) -> ExperimentRecord | None:
        """Parse a row (list of fields) into an ExperimentRecord."""
        if len(parts) < 4:
            return None
        try:
            return ExperimentRecord(
                commit=parts[0].strip(),
                val_bpb=float(parts[1].strip()) if parts[1].strip() else 0.0,
                memory_gb=float(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else 0.0,
                status=parts[3].strip() if len(parts) > 3 else "",
                description=parts[4].strip() if len(parts) > 4 else "",
                step_index=index,
            )
        except (ValueError, IndexError):
            return None

    def _make_finding(
        self,
        exp: ExperimentRecord,
        verdict: Verdict,
        level: VerificationLevel,
        explanation: str,
        expected: str | None = None,
        actual: str | None = None,
    ) -> Finding:
        """Create a Finding for an autoresearch experiment."""
        return Finding(
            step_id=exp.step_index,
            claim=Claim(
                claim_type=ClaimType.METRIC,
                source_step=exp.step_index,
                text=f"Experiment: {exp.description} (val_bpb={exp.val_bpb}, status={exp.status})",
                verifiable=True,
                evidence={"val_bpb": exp.val_bpb, "status": exp.status},
            ),
            verdict=verdict,
            level=level,
            explanation=explanation,
            expected=expected,
            actual=actual,
        )

    def _default_on_finding(self, f: Finding) -> None:
        """Default finding handler: print to terminal."""
        icons = {
            Verdict.VERIFIED_FAIL: "FAIL",
            Verdict.RULE_VIOLATION: "WARN",
            Verdict.FLAG_FOR_REVIEW: "FLAG",
        }
        icon = icons.get(f.verdict)
        if icon:
            print(f"[loop-guard:autoresearch] Step {f.step_id} [{icon}] {f.explanation}")
            if f.expected and f.actual:
                print(f"  Expected: {f.expected}")
                print(f"  Actual:   {f.actual}")

    def _print_summary(self) -> None:
        """Print summary of autoresearch run."""
        total = len(self.experiments)
        kept = sum(1 for e in self.experiments if e.status == "keep")
        discarded = sum(1 for e in self.experiments if e.status == "discard")
        crashed = sum(1 for e in self.experiments if e.status == "crash")
        best_bpb = min((e.val_bpb for e in self.experiments if e.status == "keep"), default=0)

        print("\n[loop-guard:autoresearch] Summary:")
        print(f"  Total experiments: {total}")
        print(f"  Kept: {kept} | Discarded: {discarded} | Crashed: {crashed}")
        if best_bpb > 0:
            print(f"  Best val_bpb: {best_bpb:.6f}")
        print(f"  LoopGuard findings: {len(self.guard.findings)}")

    @property
    def summary(self) -> dict:
        """Return combined summary."""
        return {
            "autoresearch": {
                "total_experiments": len(self.experiments),
                "kept": sum(1 for e in self.experiments if e.status == "keep"),
                "discarded": sum(1 for e in self.experiments if e.status == "discard"),
                "crashed": sum(1 for e in self.experiments if e.status == "crash"),
                "best_val_bpb": min(
                    (e.val_bpb for e in self.experiments if e.status == "keep"),
                    default=0,
                ),
            },
            "loopguard": self.guard.summary,
        }
