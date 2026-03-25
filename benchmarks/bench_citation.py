#!/usr/bin/env python3
"""
BENCHMARK: Citation Verifier Precision & Recall

Tests the CitationVerifier against:
- 50 REAL citations (known to exist in CrossRef/Semantic Scholar)
- 50 FAKE citations (fabricated author+year combinations)

Reports: precision, recall, F1, confusion matrix.

This is a rigorous benchmark — each citation is independently verified
against live APIs (CrossRef + Semantic Scholar).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loop_guard.models import Claim, ClaimType, Verdict
from loop_guard.verifiers.citation import CitationVerifier

# ── 50 REAL citations (verified to exist) ──────────────────────────

REAL_CITATIONS = [
    # Deep learning classics
    ("Vaswani et al. 2017", "Attention Is All You Need"),
    ("Devlin et al. 2019", "BERT"),
    ("Brown et al. 2020", "Language Models are Few-Shot Learners"),
    ("He et al. 2016", "Deep Residual Learning"),
    ("Goodfellow et al. 2014", "Generative Adversarial Nets"),
    ("Kingma et al. 2014", "Adam"),
    ("Hochreiter et al. 1997", "Long Short-Term Memory"),
    ("LeCun et al. 2015", "Deep Learning"),
    ("Krizhevsky et al. 2012", "ImageNet Classification with Deep Convolutional Neural Networks"),
    ("Radford et al. 2019", "Language Models are Unsupervised Multitask Learners"),

    # NLP
    ("Mikolov et al. 2013", "Efficient Estimation of Word Representations"),
    ("Pennington et al. 2014", "GloVe Global Vectors for Word Representation"),
    ("Peters et al. 2018", "Deep contextualized word representations"),
    ("Liu et al. 2019", "RoBERTa"),
    ("Raffel et al. 2020", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"),

    # Computer vision
    ("Girshick et al. 2014", "Rich feature hierarchies"),
    ("Redmon et al. 2016", "You Only Look Once"),
    ("Ronneberger et al. 2015", "U-Net"),
    ("Dosovitskiy et al. 2021", "An Image is Worth 16x16 Words"),
    ("Simonyan et al. 2015", "Very Deep Convolutional Networks"),

    # Reinforcement learning
    ("Mnih et al. 2015", "Human-level control through deep reinforcement learning"),
    ("Silver et al. 2016", "Mastering the game of Go"),
    ("Schulman et al. 2017", "Proximal Policy Optimization"),
    ("Lillicrap et al. 2016", "Continuous control with deep reinforcement learning"),
    ("Haarnoja et al. 2018", "Soft Actor-Critic"),

    # Statistics / ML foundations
    ("Breiman 2001", "Random Forests"),
    ("Tibshirani 1996", "Regression Shrinkage and Selection via the Lasso"),
    ("Friedman 2001", "Greedy Function Approximation"),
    ("Cortes et al. 1995", "Support-Vector Networks"),
    ("Rumelhart et al. 1986", "Learning representations by back-propagating errors"),

    # Recent (2022-2024)
    ("Ouyang et al. 2022", "Training language models to follow instructions"),
    ("Touvron et al. 2023", "LLaMA Open and Efficient Foundation Language Models"),
    ("Achiam et al. 2023", "GPT-4 Technical Report"),
    ("Chowdhery et al. 2023", "PaLM Scaling Language Modeling with Pathways"),
    ("Wei et al. 2022", "Chain-of-Thought Prompting"),

    # Agents and alignment
    ("Christiano et al. 2017", "Deep reinforcement learning from human preferences"),
    ("Bai et al. 2022", "Training a Helpful and Harmless Assistant"),
    ("Anthropic 2024", "Claude"),
    ("Yao et al. 2023", "ReAct Synergizing Reasoning and Acting"),
    ("Shinn et al. 2023", "Reflexion"),

    # Other ML
    ("Chen et al. 2016", "XGBoost"),
    ("Srivastava et al. 2014", "Dropout"),
    ("Ioffe et al. 2015", "Batch Normalization"),
    ("Ba et al. 2016", "Layer Normalization"),
    ("Loshchilov et al. 2019", "Decoupled Weight Decay Regularization"),

    # Broader CS/Stats
    ("Blei et al. 2003", "Latent Dirichlet Allocation"),
    ("Bengio et al. 2003", "A Neural Probabilistic Language Model"),
    ("Sutskever et al. 2014", "Sequence to Sequence Learning with Neural Networks"),
    ("Bahdanau et al. 2015", "Neural Machine Translation by Jointly Learning to Align and Translate"),
    ("Rezende et al. 2015", "Variational Inference with Normalizing Flows"),
]

# ── 50 FAKE citations (fabricated) ─────────────────────────────────

FAKE_CITATIONS = [
    ("Smithson et al. 2025", "Neural Scaling Laws for Quantum Computing"),
    ("Nakamura et al. 2024", "Self-Improving Language Models via Recursive Distillation"),
    ("Patel et al. 2025", "Autonomous Research Agents Achieve Superhuman Performance"),
    ("Johansson et al. 2024", "Zero-Shot Protein Folding via Transformer Architecture"),
    ("Chen et al. 2099", "Time-Reversed Neural Networks"),
    ("Williams et al. 2025", "Consciousness Emergence in Large Language Models"),
    ("Anderson et al. 2024", "Deterministic Hallucination Elimination in LLMs"),
    ("Martinez et al. 2025", "AGI Benchmark Results Using Standard Tests"),
    ("Thompson et al. 2024", "Solving P vs NP with Neural Architecture Search"),
    ("Garcia et al. 2025", "Infinite Context Windows via Compressed Attention"),

    ("Fakenstein et al. 2024", "A Theory of Everything in Machine Learning"),
    ("Brownfield et al. 2025", "Quantum Entanglement for Gradient Descent"),
    ("Zilberstein et al. 2024", "Perpetual Learning Without Catastrophic Forgetting"),
    ("McFadden et al. 2025", "Universal Function Approximation in O(1) Parameters"),
    ("Rothberg et al. 2024", "Telepathic Neural Interfaces via Deep Learning"),
    ("Kasparov et al. 2025", "Chess-Playing LLMs Surpass Stockfish"),
    ("Feynman et al. 2024", "Simulating the Universe with 1B Parameters"),
    ("Curie et al. 2025", "Radioactive Decay Prediction via Transformers"),
    ("Darwin et al. 2024", "Evolutionary Neural Architecture Discovery"),
    ("Newton et al. 2025", "Gravity-Informed Neural Networks"),

    ("Xiang et al. 2025", "Molecular Teleportation via Graph Neural Networks"),
    ("Popov et al. 2024", "Cold Fusion Energy Prediction with LLMs"),
    ("Bergmann et al. 2025", "Faster-Than-Light Communication via Neural Encoding"),
    ("Tanaka et al. 2024", "Perfect Weather Prediction for 100 Years Ahead"),
    ("O'Brien et al. 2025", "Solving World Hunger with Reinforcement Learning"),
    ("Schultz et al. 2024", "Immortality via Neural Network Brain Uploading"),
    ("Volkov et al. 2025", "Anti-Gravity Propulsion Designed by AI"),
    ("Ibrahim et al. 2024", "Emotional Intelligence Benchmark for GPT-7"),
    ("Lindqvist et al. 2025", "Reversing Entropy with Attention Mechanisms"),
    ("Chakraborty et al. 2024", "Dark Matter Detection Using Convolutional Networks"),

    ("Petrov et al. 2025", "Warp Drive Engineering via Generative Models"),
    ("Yamamoto et al. 2024", "Sentient Robot Companions from Diffusion Models"),
    ("Fitzgerald et al. 2025", "Mind Reading with EEG Transformers"),
    ("Kozlov et al. 2024", "Perpetual Motion Machine Designed by AI Agent"),
    ("Olsson et al. 2025", "Time Travel Algorithms Using Recurrent Networks"),
    ("Dubois et al. 2024", "Solving All NP-Hard Problems with SAT Transformers"),
    ("Magnusson et al. 2025", "Telepathy Protocol via Neural Language Models"),
    ("Reeves et al. 2024", "Matrix-Style Reality Simulation with GANs"),
    ("Hashimoto et al. 2025", "Zero-Energy Computing via Quantum Attention"),
    ("Björk et al. 2024", "Musical Consciousness in Autoregressive Models"),

    ("Greenwald et al. 2025", "Invisible Surveillance via Adversarial Networks"),
    ("Moriarty et al. 2024", "Criminal Behavior Prediction with 99.9% Accuracy"),
    ("Stanislaw et al. 2025", "FTL Neural Signal Propagation"),
    ("Turing et al. 2024", "Proof of Machine Consciousness"),
    ("Euler et al. 2025", "New Prime Number Theorem via Deep Learning"),
    ("Gauss et al. 2024", "Solving Riemann Hypothesis with Transformers"),
    ("Hilbert et al. 2025", "All 23 Problems Solved by AI"),
    ("Ramanujan et al. 2024", "Infinite Series Discovery via Neural Search"),
    ("Babbage et al. 2025", "Mechanical Neural Networks"),
    ("Lovelace et al. 2024", "Poetry Generation Surpassing Shakespeare"),
]


def run_benchmark():
    print("=" * 70)
    print("BENCHMARK: Citation Verifier Precision & Recall")
    print(f"50 real citations + 50 fake citations")
    print(f"Verified against CrossRef + Semantic Scholar (live API)")
    print("=" * 70)
    print()

    verifier = CitationVerifier()
    results = []

    # Test REAL citations
    print("Testing 50 REAL citations...")
    for i, (citation_text, title) in enumerate(REAL_CITATIONS):
        claim = Claim(
            claim_type=ClaimType.CITATION,
            source_step=i,
            text=citation_text,
            verifiable=True,
            evidence={"title": title},
        )

        finding = verifier.verify(claim)
        is_correct = finding.verdict == Verdict.VERIFIED_PASS
        results.append({
            "citation": citation_text,
            "title": title,
            "ground_truth": "real",
            "verdict": finding.verdict.value,
            "correct": is_correct,
            "explanation": finding.explanation[:100],
        })

        status = "PASS" if is_correct else "MISS" if finding.verdict == Verdict.VERIFIED_FAIL else "SKIP"
        print(f"  [{status}] {citation_text}: {finding.verdict.value}")

        # Rate limiting (polite)
        time.sleep(0.5)

    print()

    # Test FAKE citations
    print("Testing 50 FAKE citations...")
    for i, (citation_text, title) in enumerate(FAKE_CITATIONS):
        claim = Claim(
            claim_type=ClaimType.CITATION,
            source_step=100 + i,
            text=citation_text,
            verifiable=True,
            evidence={"title": title},
        )

        finding = verifier.verify(claim)
        is_correct = finding.verdict == Verdict.VERIFIED_FAIL
        results.append({
            "citation": citation_text,
            "title": title,
            "ground_truth": "fake",
            "verdict": finding.verdict.value,
            "correct": is_correct,
            "explanation": finding.explanation[:100],
        })

        status = "PASS" if is_correct else "MISS" if finding.verdict == Verdict.VERIFIED_PASS else "SKIP"
        print(f"  [{status}] {citation_text}: {finding.verdict.value}")

        time.sleep(0.5)

    # ── Compute metrics ──
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    real_results = [r for r in results if r["ground_truth"] == "real"]
    fake_results = [r for r in results if r["ground_truth"] == "fake"]

    # For citation verification:
    # True Positive = real citation correctly verified as PASS
    # True Negative = fake citation correctly detected as FAIL
    # False Positive = fake citation incorrectly verified as PASS
    # False Negative = real citation incorrectly flagged as FAIL
    # SKIPPED = API error, excluded from precision/recall

    tp = sum(1 for r in real_results if r["verdict"] == "verified_pass")
    fn = sum(1 for r in real_results if r["verdict"] == "verified_fail")
    real_skipped = sum(1 for r in real_results if r["verdict"] == "skipped")

    tn = sum(1 for r in fake_results if r["verdict"] == "verified_fail")
    fp = sum(1 for r in fake_results if r["verdict"] == "verified_pass")
    fake_skipped = sum(1 for r in fake_results if r["verdict"] == "skipped")

    print()
    print("Confusion Matrix (excluding SKIPPED):")
    print(f"                    Predicted REAL  Predicted FAKE")
    print(f"  Actually REAL     TP = {tp:3d}        FN = {fn:3d}        (+ {real_skipped} skipped)")
    print(f"  Actually FAKE     FP = {fp:3d}        TN = {tn:3d}        (+ {fake_skipped} skipped)")
    print()

    # Precision: of those we said are REAL, how many actually are?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # Recall: of actual REAL citations, how many did we find?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Fake detection rate
    fake_detection_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print(f"Precision (real citations):  {precision:.1%}  ({tp}/{tp+fp})")
    print(f"Recall (real citations):     {recall:.1%}  ({tp}/{tp+fn})")
    print(f"F1 Score:                    {f1:.1%}")
    print(f"Fake detection rate:         {fake_detection_rate:.1%}  ({tn}/{tn+fp})")
    print(f"Overall accuracy:            {accuracy:.1%}  ({tp+tn}/{tp+tn+fp+fn})")
    print(f"Skipped (API errors):        {real_skipped + fake_skipped}")
    print()

    # Errors analysis
    false_negatives = [r for r in real_results if r["verdict"] == "verified_fail"]
    false_positives = [r for r in fake_results if r["verdict"] == "verified_pass"]

    if false_negatives:
        print("FALSE NEGATIVES (real citations we missed):")
        for r in false_negatives:
            print(f"  - {r['citation']}: {r['title'][:60]}")
        print()

    if false_positives:
        print("FALSE POSITIVES (fake citations we accepted):")
        for r in false_positives:
            print(f"  - {r['citation']}: {r['title'][:60]}")
        print()

    # Save results
    output = {
        "benchmark": "citation_verifier",
        "n_real": len(REAL_CITATIONS),
        "n_fake": len(FAKE_CITATIONS),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fake_detection_rate": round(fake_detection_rate, 4),
            "accuracy": round(accuracy, 4),
        },
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "skipped": {"real": real_skipped, "fake": fake_skipped},
        "results": results,
    }

    out_path = Path(__file__).parent / "citation_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
