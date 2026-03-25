#!/usr/bin/env python3
"""
BENCHMARK: False Positive Rate

Feeds 50 CLEAN agent outputs (no errors, valid claims) through loop-guard
and counts how many are incorrectly flagged.

A good verifier should have a low false positive rate on clean data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loop_guard import LoopGuard
from loop_guard.models import Verdict

# 50 clean agent outputs with valid claims
CLEAN_OUTPUTS = [
    "Trained the model for 10 epochs. Final accuracy = 87.3% on the test set.",
    "The learning rate was set to 0.001 with Adam optimizer. Loss converged to 0.234.",
    "Preprocessing complete. Removed 12 duplicate rows from the dataset.",
    "Feature engineering: created 5 new features from the timestamp column.",
    "Cross-validation results: mean accuracy = 91.2%, std = 2.3%.",
    "The model has 2.5M parameters and takes 3 minutes to train per epoch.",
    "Evaluation on held-out test set: precision = 0.89, recall = 0.85, f1 = 0.87.",
    "Data split: 80% train (8000 samples), 20% test (2000 samples).",
    "Batch size = 32, sequence length = 128. Training completed in 45 minutes.",
    "The gradient norm was clipped at 1.0 to prevent exploding gradients.",

    "Hyperparameter search over 20 configurations. Best: lr=0.0003, dropout=0.1.",
    "Model checkpoint saved at epoch 15 with validation loss = 0.189.",
    "Data augmentation: applied random horizontal flips and rotations up to 15 degrees.",
    "The confusion matrix shows 450 true positives, 50 false positives, 30 false negatives.",
    "ROC AUC = 0.94 on the validation set. Threshold set at 0.5.",
    "Feature importance: top 3 features are age (0.23), income (0.19), education (0.15).",
    "Ensemble of 5 models achieved accuracy = 93.1%, up from 91.2% for single model.",
    "The training data contains 50000 samples across 10 classes, balanced.",
    "Memory usage during training: peak 4.2 GB VRAM on a single GPU.",
    "Inference latency: 12ms per sample on CPU, 2ms per sample on GPU.",

    "Tokenizer vocabulary size: 32000 tokens. Average sequence length: 256 tokens.",
    "The attention mechanism uses 8 heads with dimension 64 each.",
    "Weight decay = 0.01 applied to all parameters except biases and layer norms.",
    "Warmup schedule: linear warmup for 1000 steps, then cosine decay.",
    "The model achieved perplexity = 15.3 on the validation set.",
    "Gradient accumulation over 4 steps gives effective batch size of 128.",
    "Mixed precision training (FP16) reduced memory by 40% with no accuracy loss.",
    "Early stopping triggered at epoch 23 (patience=5). Best epoch was 18.",
    "The dataset was shuffled with seed=42 for reproducibility.",
    "Label smoothing of 0.1 improved calibration without hurting accuracy.",

    "Transferred weights from pretrained model. Fine-tuning for 5 epochs.",
    "The convolutional layers use 3x3 kernels with stride 1 and padding 1.",
    "Dropout rate = 0.3 applied after each dense layer.",
    "Batch normalization applied before activation functions.",
    "The loss function is cross-entropy with class weights to handle imbalance.",
    "Optimizer: SGD with momentum 0.9 and learning rate 0.01.",
    "The validation accuracy plateaued at 89.5% after epoch 12.",
    "Data loading: 4 workers, pin_memory=True, prefetch_factor=2.",
    "The model architecture: 4 conv blocks followed by 2 dense layers.",
    "Output layer: softmax over 10 classes. Temperature scaling = 1.0.",

    "Training curve shows smooth convergence without oscillation.",
    "The embedding dimension is 512. Positional encoding uses sinusoidal functions.",
    "Beam search with width 5 for generation. Max length 100 tokens.",
    "The model was evaluated on 3 different random seeds. Results are consistent.",
    "Regularization: L2 penalty of 0.0001 on all weight matrices.",
    "The dataset was downloaded from the official repository. SHA256 hash verified.",
    "Total training time: 2 hours 15 minutes on a single V100 GPU.",
    "The model file is 45 MB. Quantized version is 12 MB with <1% accuracy drop.",
    "Inference throughput: 500 samples per second on batch size 64.",
    "The API endpoint returns results in 50ms average latency under load.",
]


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: False Positive Rate")
    print(f"50 clean agent outputs (no errors) through loop-guard")
    print("=" * 70)
    print()

    guard = LoopGuard(config={
        "use_llm_extraction": False,
        "verbosity": "all",
    })

    false_positives = []
    total_claims = 0
    total_findings = 0

    for i, output in enumerate(CLEAN_OUTPUTS):
        findings = guard.step(output=output, step_id=i)
        total_findings += len(findings)

        for f in findings:
            total_claims += 1
            if f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION):
                false_positives.append({
                    "step": i,
                    "output": output[:80],
                    "verdict": f.verdict.value,
                    "explanation": f.explanation,
                })
                print(f"  [FP] Step {i}: {f.verdict.value} — {f.explanation[:80]}")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    fp_rate = len(false_positives) / len(CLEAN_OUTPUTS) if CLEAN_OUTPUTS else 0
    claim_fp_rate = len(false_positives) / total_claims if total_claims > 0 else 0

    print(f"Total clean outputs tested:   {len(CLEAN_OUTPUTS)}")
    print(f"Total claims extracted:       {total_claims}")
    print(f"Total findings:               {total_findings}")
    print(f"False positives (FAIL/WARN):  {len(false_positives)}")
    print(f"FP rate (per output):         {fp_rate:.1%}")
    print(f"FP rate (per claim):          {claim_fp_rate:.1%}")
    print()

    if false_positives:
        print("FALSE POSITIVE DETAILS:")
        for fp in false_positives:
            print(f"  Step {fp['step']}: [{fp['verdict']}] {fp['explanation'][:100]}")
        print()

    # Save results
    output = {
        "benchmark": "false_positive_rate",
        "n_outputs": len(CLEAN_OUTPUTS),
        "n_claims": total_claims,
        "n_false_positives": len(false_positives),
        "fp_rate_per_output": round(fp_rate, 4),
        "fp_rate_per_claim": round(claim_fp_rate, 4),
        "false_positives": false_positives,
    }

    out_path = Path(__file__).parent / "false_positive_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
