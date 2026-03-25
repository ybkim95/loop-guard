# loop-guard: Deterministic Verification for Autonomous Agent Loops

*Not another LLM-as-judge. Re-run the code. Look up the citation. Check the math.*

---

## The Problem

Autonomous agent loops — autoresearch, coding agents, data science pipelines — run for hours without human oversight. Agents make intermediate claims that compound: "accuracy is 94%", "p < 0.05", "Smith et al. 2024 showed...". A wrong claim at step 23 becomes the premise for steps 24–100.

loop-guard is a framework-agnostic verification layer that performs deterministic checks on agent claims at each step. It does NOT use LLMs to judge LLMs. It re-runs code, looks up citations in academic databases, and checks statistics using rules.

## Benchmarks

We benchmarked each verifier on curated datasets. Here are the numbers, honestly reported with limitations.

### Statistical Verifier (n=100)

Tested on 50 correct and 50 incorrect statistical claims (impossible p-values, accuracy > 100%, negative variance, missing multiple comparison corrections, small sample sizes).

| Metric | Score |
|--------|-------|
| Precision | **94.2%** |
| Recall | **98.0%** |
| F1 | **96.1%** |

**3 false positives:** The regex matched unrelated numbers as sample sizes (e.g., extracted `n=12` from "Standard deviation = 12.4" and flagged it as a small sample). These are fixable pattern improvements.

**1 false negative:** "The variance was -3.7" was missed because the regex requires `variance =` or `variance:` but the text used "variance was". Edge case in natural language variation.

### Citation Verifier (n=100, live API)

Tested on 50 real citations (well-known ML/NLP papers) and 50 fabricated citations, verified against CrossRef and Semantic Scholar APIs.

| Metric | Score |
|--------|-------|
| Precision | **84.8%** |
| Fake detection rate | **89.8%** |
| Recall | **56.0%** |

**Why recall is 56%, not higher:** Many well-known papers (BERT, Adam, Dropout, GPT-4 Technical Report) have short or generic titles that don't match the longer titles returned by APIs. This is a limitation of token-level title matching against academic databases. We report this honestly rather than hiding it.

**5 false positives on fakes:** Some fabricated titles like "Self-Improving Language Models via Recursive Distillation" partially match real papers about language models. Title-level verification has inherent limits when fake titles use real terminology.

**What this means practically:** The citation verifier is good at catching obviously fabricated citations (90% detection rate) and has high precision when it does confirm a citation (85%). It should be used as a screening tool, not as a definitive oracle. Citations it flags as FAIL deserve human review.

### False Positive Rate (n=50)

Tested on 50 clean agent outputs containing no errors — valid metrics, correct statistics, normal training logs.

| Metric | Score |
|--------|-------|
| False positive rate | **0.0%** |

Zero false alarms on clean data. loop-guard does not cry wolf.

## Autoresearch Analysis (Retrospective)

**Important caveat:** This is a retrospective analysis of Karpathy's published results, not a live experiment. We analyzed the `results.tsv` from the first public autoresearch overnight run (126 experiments). loop-guard was **not running during the original experiment**.

What loop-guard's autoresearch monitor detected post-hoc:
- 12 success rate alerts (periods where 0% of experiments were kept over 20-experiment windows)
- Longest unproductive streak: 25 consecutive discards (125 min of GPU time)

**What we can claim:** loop-guard's autoresearch monitor correctly identifies unproductive stretches from `results.tsv` data.

**What we cannot yet claim:** That intervening during these stretches would have improved the final val_bpb. That requires a controlled A/B experiment with live monitoring, which we have not run.

## Gemini API Verification (Single Run)

We called Gemini 2.0 Flash with a 7-step research workflow and loop-guard verifying each step. This is a **single demonstration run**, not a systematic evaluation.

Results from one run: 22 claims extracted, 2 VERIFIED_FAIL (citations), 9 FLAG_FOR_REVIEW (multiple comparisons without correction + tainted provenance chain).

The provenance chain correctly flagged step 3 as tainted because it depended on step 2, which had a VERIFIED_FAIL.

## How It Works

```
Agent Loop → Claim Extractor → Verification Engine → Reporter
               (regex-first)      (3 layers)
```

**Three verification layers:**

| Layer | Method | Reliability |
|-------|--------|-------------|
| L1: Deterministic | Re-execute code, API lookups | Cannot be wrong |
| L2: Rule-based | Pattern matching, sanity checks | Rarely wrong (benchmarked above) |
| L3: LLM-assisted | Soft flagging only | May be wrong, needs human review |

LLMs are used only for claim extraction (a structured task), never for judgment.

## Integration

```python
from loop_guard import LoopGuard

guard = LoopGuard()

for step in agent.run(task):
    findings = guard.step(output=step.text)
```

Works with any agent framework. SDK wrappers available for Google Gemini/ADK, OpenAI, and Anthropic.

```bash
pip install loopguard-ai
```

## Limitations

We are explicit about what loop-guard cannot do:

1. **Citation verifier recall is 56%.** It misses papers with short/generic titles. It should be used as a screening tool, not a definitive oracle.
2. **No A/B experiment yet.** We have not proven that loop-guard improves final agent outcomes. The autoresearch analysis is retrospective.
3. **Code re-execution assumes determinism.** Non-deterministic code (random seeds, network calls) will produce false positives.
4. **Claim extraction is regex-based.** It misses claims that don't match predefined patterns. Coverage depends on how the agent formats its output.
5. **The Gemini demo is a single run.** Not a systematic evaluation across many tasks.

## What's Next

- Controlled A/B experiment on autoresearch (with vs without loop-guard)
- Improve citation recall via DOI-based lookup and arXiv search
- Systematic evaluation across 50+ agent runs
- NeurIPS 2026 submission with empirical results

## Links

- **GitHub:** https://github.com/ybkim95/loop-guard
- **PyPI:** `pip install loopguard-ai`
- **91 tests passing**, Python 3.10+, framework-agnostic

---

*loop-guard catches silent errors in agent loops by re-running code, looking up citations, and checking statistics — not by asking another LLM if the output "looks right."*
