# loop-guard

**Deterministic verification for autonomous agent loops.**

loop-guard catches silent errors in agent loops by re-running code, looking up citations, and checking statistics — not by asking another LLM if the output "looks right."

[![PyPI](https://img.shields.io/pypi/v/loopguard-ai)](https://pypi.org/project/loopguard-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Install

```bash
pip install loopguard-ai
```

## Quick Start

```python
from loop_guard import LoopGuard

guard = LoopGuard()

# Works with ANY agent loop
for i in range(num_experiments):
    result = agent.run(task)
    findings = guard.step(output=result.text)
    for f in findings:
        print(f)
```

```bash
# CLI: pipe any agent's stdout
python my_agent.py 2>&1 | loop-guard watch

# Monitor autoresearch
loop-guard autoresearch ./autoresearch/ --poll 30
```

## Benchmarks

Every claim below is backed by reproducible benchmarks in [`benchmarks/`](benchmarks/).

### Statistical Verifier (n=100)

50 correct + 50 incorrect statistical claims (impossible p-values, accuracy > 100%, negative variance, small sample sizes).

| Metric | Score |
|--------|-------|
| Precision | **94.2%** (49/52) |
| Recall | **98.0%** (49/50) |
| F1 | **96.1%** |

3 false positives (regex matched unrelated numbers as sample sizes). 1 false negative (natural language variant "variance was -3.7" not caught by `variance =` pattern).

### Citation Verifier (n=100, live API)

50 real citations (well-known ML papers) + 50 fabricated citations, verified against CrossRef and Semantic Scholar.

| Metric | Score |
|--------|-------|
| Precision | **84.8%** (28/33) |
| Fake detection | **89.8%** (44/49) |
| Recall | **56.0%** (28/50) |

**Limitation:** Recall is 56% because papers with short/generic titles (BERT, Adam, Dropout) don't match the longer titles returned by APIs. The citation verifier is a screening tool, not a definitive oracle. Citations flagged as FAIL deserve human review.

### False Positive Rate (n=50)

50 clean agent outputs (valid metrics, correct statistics, normal training logs).

| Metric | Score |
|--------|-------|
| False positive rate | **0.0%** |

Zero false alarms on clean data.

## How It Works

```
Agent Loop → Claim Extractor → Verification Engine → Reporter
               (regex-first)      (3 layers)         (terminal/JSON/HTML)
```

| Layer | Method | Reliability |
|-------|--------|-------------|
| **L1: Deterministic** | Re-execute code, API lookup | Cannot be wrong |
| **L2: Rule-based** | Pattern matching, sanity checks | Benchmarked above |
| **L3: LLM-assisted** | Soft flagging only | May be wrong |

LLMs are used only for claim extraction (a structured task), never for judgment.

## Verifiers

| Verifier | Layer | What it catches |
|----------|-------|----------------|
| **StatisticalVerifier** | L2 | Impossible p-values, accuracy > 100%, missing corrections |
| **CitationVerifier** | L1 | Fabricated citations (CrossRef + Semantic Scholar lookup) |
| **LoopTrapVerifier** | L2 | Agent stuck retrying the same approach |
| **RegressionVerifier** | L2 | Agent reverts a file to a previous version |
| **CodeOutputVerifier** | L1 | Code re-execution produces different output |
| **MetricVerifier** | L1 | Metric re-computation doesn't match claim |
| **ToolOutputVerifier** | L1 | Tool re-execution produces different result |
| **ProvenanceChain** | L2 | Downstream claims tainted by upstream failures |

## Framework Integrations

```python
# Google Gemini / ADK
from loop_guard.integrations.google_adk import GeminiGuard
guard = GeminiGuard(api_key="...")
result = guard.generate("Analyze this data...")

# OpenAI
from loop_guard.integrations.openai_agents import OpenAIGuard

# Anthropic Claude
from loop_guard.integrations.anthropic_sdk import AnthropicGuard

# Autoresearch
from loop_guard.integrations.autoresearch import AutoresearchGuard
```

## Limitations

1. **Citation recall is 56%.** Papers with short/generic titles are missed. Use as a screening tool.
2. **No A/B experiment yet.** We have not proven loop-guard improves final agent outcomes.
3. **Code re-execution assumes determinism.** Non-deterministic code produces false positives.
4. **Regex-based extraction.** Claims not matching predefined patterns are missed.
5. **Autoresearch analysis is retrospective.** Not a live monitoring experiment.

## What loop-guard Is NOT

- **NOT an LLM-as-judge system.** Those share the same failure modes as the agent.
- **NOT a prompt injection detector.**
- **NOT a replacement for human review.** It flags issues for humans to investigate.

## Contributing

```bash
git clone https://github.com/ybkim95/loop-guard
cd loop-guard
pip install -e ".[dev]"
pytest
```

## License

MIT
