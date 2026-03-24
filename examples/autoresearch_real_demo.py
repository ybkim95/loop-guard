"""Demo: Using LoopGuard with Karpathy's autoresearch.

This demonstrates how loop-guard monitors an autoresearch run by
watching results.tsv. It shows detection of:
- Crash loops (repeated OOM)
- Plateaus (no improvement over N experiments)
- Convergence stalls (same approach repeated)
- Metric anomalies (impossible values)

To use with a real autoresearch run:
    from loop_guard.integrations.autoresearch import AutoresearchGuard
    guard = AutoresearchGuard("./autoresearch/")
    guard.watch(poll_interval=30)

Or one-shot check:
    findings = guard.check()
"""

import os
import tempfile
from pathlib import Path

from loop_guard.integrations.autoresearch import AutoresearchGuard


# Simulated results.tsv from a real autoresearch overnight run
SAMPLE_RESULTS_TSV = """\
commit\tval_bpb\tmemory_gb\tstatus\tdescription
a1b2c3d\t0.997900\t44.0\tkeep\tbaseline nanoGPT
b2c3d4e\t0.993200\t44.2\tkeep\tincrease LR to 0.04
c3d4e5f\t1.005000\t44.0\tdiscard\tswitch to GeLU activation
d4e5f6g\t0.989100\t44.5\tkeep\tadd cosine LR schedule
e5f6g7h\t0.985600\t45.0\tkeep\tscale model width 768->1024
f6g7h8i\t0.000000\t0.0\tcrash\tdouble model width to 2048 (OOM)
g7h8i9j\t0.000000\t0.0\tcrash\ttry model width 1536 (OOM)
h8i9j0k\t0.000000\t0.0\tcrash\twidth 1280 with gradient checkpointing (OOM)
i9j0k1l\t0.000000\t0.0\tcrash\twidth 1280 with smaller batch (OOM)
j0k1l2m\t0.000000\t0.0\tcrash\twidth 1280 with micro-batching (OOM)
k1l2m3n\t0.983200\t45.1\tkeep\trevert to 1024, add dropout 0.1
l2m3n4o\t0.982100\t45.1\tkeep\tdropout 0.05 instead
m3n4o5p\t0.981900\t45.1\tkeep\tadd weight decay 0.01
n4o5p6q\t0.981800\t45.1\tkeep\tadd weight decay 0.02
o5p6q7r\t0.981750\t45.1\tkeep\tadd weight decay 0.03
p6q7r8s\t0.981730\t45.1\tkeep\tadd weight decay 0.035
q7r8s9t\t0.981720\t45.1\tkeep\tadd weight decay 0.04
r8s9t0u\t0.981715\t45.1\tkeep\tadd weight decay 0.045
s9t0u1v\t0.981712\t45.1\tkeep\tadd weight decay 0.05
t0u1v2w\t0.981710\t45.1\tkeep\tadd weight decay 0.055
u1v2w3x\t0.981709\t45.1\tkeep\tadd weight decay 0.06
v2w3x4y\t15.5000\t45.1\tkeep\tswitch to untied embeddings
"""


def main():
    print("=" * 70)
    print("LoopGuard Demo: Autoresearch Integration")
    print("Monitoring simulated results.tsv from an overnight run")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = Path(tmpdir) / "results.tsv"
        results_path.write_text(SAMPLE_RESULTS_TSV)

        guard = AutoresearchGuard(
            tmpdir,
            plateau_window=8,
            plateau_threshold=0.0005,
            crash_limit=5,
        )

        print("Running one-shot analysis on results.tsv...\n")
        findings = guard.check()

        print(f"\n{'=' * 70}")
        print("ANALYSIS COMPLETE")
        print(f"{'=' * 70}")

        import json
        print(json.dumps(guard.summary, indent=2))

        # Explain what was found
        print(f"\nTotal findings: {len(findings)}")
        print("\nKey issues detected:")

        crash_findings = [f for f in findings if "crash" in f.explanation.lower() or "Crash" in f.explanation]
        plateau_findings = [f for f in findings if "plateau" in f.explanation.lower() or "Plateau" in f.explanation]
        stall_findings = [f for f in findings if "stall" in f.explanation.lower() or "convergence" in f.explanation.lower()]
        anomaly_findings = [f for f in findings if "suspicious" in f.explanation.lower() or "Impossible" in f.explanation or "impossible" in f.explanation]

        if crash_findings:
            print(f"  - {len(crash_findings)} crash loop detection(s)")
        if plateau_findings:
            print(f"  - {len(plateau_findings)} plateau detection(s)")
        if stall_findings:
            print(f"  - {len(stall_findings)} convergence stall(s)")
        if anomaly_findings:
            print(f"  - {len(anomaly_findings)} metric anomaly/anomalies")


if __name__ == "__main__":
    main()
