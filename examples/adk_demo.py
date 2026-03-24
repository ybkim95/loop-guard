"""Demo: Using LoopGuard with Google ADK (Agent Development Kit).

Shows the integration pattern for a data analysis agent.
LoopGuard catches statistical errors and metric issues.
"""

from loop_guard import LoopGuard


def main():
    """Simulates a Google ADK data analysis agent with LoopGuard verification."""

    guard = LoopGuard(config={
        "use_llm_extraction": False,
        "verbosity": "all",
    })

    # Simulated data analysis agent steps
    steps = [
        "Loading dataset from BigQuery. 10,000 rows, 45 features.\n"
        "Data loaded successfully.",

        "Exploratory analysis complete.\n"
        "Found 3 features with >50% missing values.\n"
        "accuracy = 0.0% (random baseline)\n"
        "Proceeding with feature engineering.",

        "Feature engineering complete. Created 12 new features.\n"
        "Running logistic regression baseline.\n"
        "accuracy = 67.3%\n"
        "precision = 0.71\n"
        "recall = 0.63",

        "Trained gradient boosting model.\n"
        "accuracy = 81.2%\n"
        "precision = 0.83\n"
        "recall = 0.79\n"
        "Based on Chen and Guestrin 2016, XGBoost handles missing values natively.",

        "Hyperparameter tuning with Optuna.\n"
        "Best trial: accuracy = 84.7%\n"
        "p < 0.03 vs baseline (Wilcoxon signed-rank test)\n"
        "With n=15 cross-validation folds",

        "Feature importance analysis.\n"
        "Top feature: age (importance = 0.23)\n"
        "Running SHAP analysis.\n"
        "p < 0.01 for age effect\n"
        "p < 0.04 for income effect\n"
        "p < 0.03 for education effect",  # Multiple comparisons without correction

        "Final model evaluation on held-out test set.\n"
        "accuracy = 83.9%\n"
        "precision = 0.85\n"
        "recall = 0.82\n"
        "f1 = 0.835\n"
        "auc = 0.91\n"
        "R² = 0.78\n"
        "All tests passed.",
    ]

    print("=" * 60)
    print("LoopGuard Demo: Google ADK Data Analysis Agent")
    print("=" * 60)

    for i, output in enumerate(steps):
        print(f"\n--- Step {i} ---")
        print(output[:120] + "..." if len(output) > 120 else output)
        print()
        guard.step(output=output, step_id=i)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    import json
    print(json.dumps(guard.summary, indent=2))

    guard.report(format="html", path="adk_demo_report.html")
    print("\nHTML report: adk_demo_report.html")


if __name__ == "__main__":
    main()
