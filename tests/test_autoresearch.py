"""Tests for autoresearch integration."""

import tempfile
from pathlib import Path

from loop_guard.integrations.autoresearch import AutoresearchGuard
from loop_guard.models import Verdict

BASIC_TSV = """\
commit\tval_bpb\tmemory_gb\tstatus\tdescription
abc123\t0.997\t44.0\tkeep\tbaseline
def456\t0.993\t44.2\tkeep\tincrease LR
ghi789\t1.005\t44.0\tdiscard\tswitch activation
"""


class TestAutoresearchGuard:
    def test_reads_results_tsv(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(BASIC_TSV)
            guard = AutoresearchGuard(tmp)
            guard.check()
            assert len(guard.experiments) == 3

    def test_detects_crash_loop(self):
        tsv = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        for i in range(6):
            tsv += f"crash{i}\t0.0\t0.0\tcrash\tOOM attempt {i}\n"

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(tsv)
            guard = AutoresearchGuard(tmp, crash_limit=5)
            findings = guard.check()
            crash_findings = [f for f in findings if "crash" in f.explanation.lower()]
            assert len(crash_findings) > 0

    def test_detects_plateau(self):
        tsv = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        # 12 experiments with tiny improvements
        for i in range(12):
            bpb = 0.985 - i * 0.00001  # only 0.00001 improvement each
            tsv += f"plat{i}\t{bpb:.6f}\t44.0\tkeep\texperiment {i}\n"

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(tsv)
            guard = AutoresearchGuard(tmp, plateau_window=8, plateau_threshold=0.001)
            findings = guard.check()
            plateau_findings = [f for f in findings if "plateau" in f.explanation.lower()]
            assert len(plateau_findings) > 0

    def test_detects_convergence_stall(self):
        tsv = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        # Same description repeated
        for i in range(5):
            tsv += f"stall{i}\t0.99\t44.0\tdiscard\tincrease learning rate\n"

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(tsv)
            guard = AutoresearchGuard(tmp, similarity_threshold=0.85)
            findings = guard.check()
            stall_findings = [f for f in findings
                             if "convergence" in f.explanation.lower()
                             or "stall" in f.explanation.lower()]
            assert len(stall_findings) > 0

    def test_detects_impossible_val_bpb(self):
        tsv = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        tsv += "bad1\t-0.5\t44.0\tkeep\timpossible metric\n"

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(tsv)
            guard = AutoresearchGuard(tmp)
            findings = guard.check()
            fail_findings = [f for f in findings if f.verdict == Verdict.VERIFIED_FAIL]
            assert len(fail_findings) > 0

    def test_detects_suspiciously_high_val_bpb(self):
        tsv = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        tsv += "bad1\t15.5\t44.0\tkeep\tsuspicious metric\n"

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(tsv)
            guard = AutoresearchGuard(tmp)
            findings = guard.check()
            violation_findings = [f for f in findings if f.verdict == Verdict.RULE_VIOLATION]
            assert len(violation_findings) > 0

    def test_no_false_positives_on_healthy_run(self):
        tsv = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        tsv += "a1\t0.997\t44.0\tkeep\tbaseline model\n"
        tsv += "a2\t0.985\t44.2\tkeep\tadd cosine schedule\n"
        tsv += "a3\t0.970\t44.5\tkeep\tscale to 1024 width\n"
        tsv += "a4\t1.010\t44.0\tdiscard\tswitch to GeLU\n"
        tsv += "a5\t0.960\t45.0\tkeep\tadd layer normalization\n"

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(tsv)
            guard = AutoresearchGuard(tmp)
            findings = guard.check()
            # Only PASS and SKIP findings, no violations or failures
            bad_findings = [f for f in findings
                          if f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION)]
            assert len(bad_findings) == 0

    def test_summary_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "results.tsv").write_text(BASIC_TSV)
            guard = AutoresearchGuard(tmp)
            guard.check()
            summary = guard.summary
            assert summary["autoresearch"]["total_experiments"] == 3
            assert summary["autoresearch"]["kept"] == 2
            assert summary["autoresearch"]["discarded"] == 1
            assert summary["autoresearch"]["crashed"] == 0

    def test_missing_results_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = AutoresearchGuard(tmp)
            findings = guard.check()
            assert len(findings) == 0
            assert len(guard.experiments) == 0
