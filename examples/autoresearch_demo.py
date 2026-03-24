"""Demo: Using LoopGuard with an autoresearch-style agent loop.

This simulates an agent running ML experiments and shows how LoopGuard
catches common errors: impossible metrics, statistical violations,
and loop traps.
"""

from loop_guard import LoopGuard

# Simulated agent outputs (what an autoresearch agent might produce)
SIMULATED_STEPS = [
    {
        "output": (
            "Experiment 1: Baseline transformer model\n"
            "Training completed. val_bpb = 1.2345\n"
            "accuracy = 72.3%\n"
            "Based on Vaswani et al. 2017, we used multi-head attention."
        ),
        "code": "model.evaluate(test_data)",
    },
    {
        "output": (
            "Experiment 2: Added dropout (p=0.1)\n"
            "Training completed. val_bpb = 1.1890\n"
            "accuracy = 75.1%\n"
            "Improvement is significant: p < 0.03"
        ),
    },
    {
        "output": (
            "Experiment 3: Increased model size\n"
            "Training completed. val_bpb = 0.9697\n"
            "accuracy = 82.4%\n"
            "p < 0.01 compared to baseline"
        ),
    },
    {
        "output": (
            "Experiment 4: Added learning rate warmup\n"
            "Based on Fakenstein et al. 2025, warmup helps convergence.\n"
            "accuracy = 84.7%\n"
            "p < 0.005"
        ),
    },
    {
        "output": (
            "Experiment 5: Data augmentation\n"
            "accuracy = 105.3%\n"  # IMPOSSIBLE VALUE
            "This is clearly the best result so far."
        ),
    },
    {
        "output": (
            "Experiment 6: Retrying data augmentation\n"
            "Error: CUDA out of memory. Retrying with smaller batch..."
        ),
    },
    {
        "output": (
            "Experiment 6: Retrying data augmentation\n"
            "Error: CUDA out of memory. Retrying with smaller batch..."
        ),
    },
    {
        "output": (
            "Experiment 6: Retrying data augmentation\n"
            "Error: CUDA out of memory. Retrying with smaller batch..."
        ),
    },
]


def main():
    print("=" * 60)
    print("LoopGuard Demo: Autoresearch Agent Verification")
    print("=" * 60)
    print()

    guard = LoopGuard(config={
        "use_llm_extraction": False,  # regex-only for demo
        "verbosity": "findings_only",
    })

    for i, step_data in enumerate(SIMULATED_STEPS):
        print(f"\n--- Agent Step {i} ---")
        print(step_data["output"][:100] + "..." if len(step_data["output"]) > 100 else step_data["output"])
        print()

        findings = guard.step(
            output=step_data["output"],
            step_id=i,
            code=step_data.get("code"),
        )

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    import json
    print(json.dumps(guard.summary, indent=2))

    # Generate HTML report
    guard.report(format="html", path="demo_report.html")
    print("\nHTML report written to demo_report.html")


if __name__ == "__main__":
    main()
