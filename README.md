# loop-guard

**Deterministic verification for autonomous agent loops.**

loop-guard catches silent errors in agent loops by re-running code, looking up citations, and checking statistics — not by asking another LLM if the output "looks right."

[![PyPI](https://img.shields.io/pypi/v/loopguard-ai)](https://pypi.org/project/loopguard-ai/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## The Problem

Autonomous agent loops run for hours without human oversight. Agents make intermediate claims — "accuracy is 94%", "tests pass", "p < 0.05" — that compound over hundreds of steps. A wrong claim at step 23 becomes the premise for steps 24–100. Nobody catches the error until a human reviews the final output, if at all.

## How loop-guard Works

```
Agent Loop → Integration Layer → Claim Extractor → Verification Engine → Reporter
                                    (regex-first)     (3 layers)         (terminal/JSON/HTML)
```

**Three verification layers, in order of reliability:**

| Layer | Method | Reliability | Example |
|-------|--------|-------------|---------|
| **L1: Deterministic** | Re-execute code, API lookup, re-compute | Cannot be wrong | Citation lookup, code re-run |
| **L2: Rule-based** | Pattern matching, sanity checks | Rarely wrong | p > 1 detection, loop trap |
| **L3: LLM-assisted** | Soft flagging only | May be wrong | General claim flagging |

**Key principle:** Verification must be more reliable than the thing being verified. LLMs are used only for claim extraction (a structured task), never for judgment.

## Install

```bash
pip install loopguard-ai

# With LLM-based claim extraction (optional)
pip install loopguard-ai[llm]
```

## Quick Start

### Python API (2 lines to integrate)

```python
from loop_guard import LoopGuard

guard = LoopGuard()

# Works with ANY agent loop — OpenAI, Anthropic, Google ADK, LangGraph, custom
for i in range(num_experiments):
    result = agent.run(task)
    findings = guard.step(
        output=result.text,
        code=result.code_executed,       # optional
        files=result.files_modified,     # optional
    )
    for f in findings:
        print(f)  # FAIL/WARN/FLAG with explanation

# Generate report
guard.report(format="html", path="audit.html")
```

### CLI (zero code change)

```bash
# Pipe any agent's stdout
python my_agent.py 2>&1 | loop-guard watch

# Watch a log file
loop-guard watch --file agent.log --follow

# Watch git commits (for autoresearch)
loop-guard watch --git-dir ./experiments/ --poll 30

# Check a transcript
loop-guard check --input transcript.txt --format html
```

## What Gets Verified

### Verifiers

| Verifier | Layer | What it catches |
|----------|-------|----------------|
| **LoopTrapVerifier** | L2 | Agent stuck retrying the same failing approach |
| **RegressionVerifier** | L2 | Agent reverts a file to a previous version |
| **CitationVerifier** | L1 | Hallucinated academic citations (CrossRef + Semantic Scholar) |
| **StatisticalVerifier** | L2 | Impossible p-values, missing multiple comparison correction, small samples |
| **CodeOutputVerifier** | L1 | Agent claims code produced output X, but re-execution produces Y |
| **MetricVerifier** | L1 | Agent claims metric = X, but re-computation gives Y |

### Claim Extraction

Claims are extracted from agent output using a **regex-first pipeline**:

1. **Regex patterns** catch citations, metrics, p-values, test results, and file modifications
2. **LLM extraction** (optional) handles remaining unstructured text
3. Claims are typed: `CODE_OUTPUT`, `METRIC`, `STATISTICAL`, `CITATION`, `TEST_RESULT`, `FILE_STATE`, `GENERAL`

## Output

### Terminal (real-time)

```
[loop-guard] Step 4 [FAIL] [L2] Impossible statistical value: accuracy = 105.3% (> 100%)
             Expected: accuracy ∈ [0%, 100%]
             Actual:   105.3%
[loop-guard] Step 3 [WARN] [L1] Citation not found in CrossRef or Semantic Scholar
             Expected: Fakenstein et al. 2025
             Actual:   No matching paper found
[loop-guard] Step 7 [WARN] [L2] Agent appears stuck: 3 consecutive similar outputs
```

### JSON Report

```bash
loop-guard check --input transcript.txt --output report.json --format json
```

### HTML Report

```bash
loop-guard check --input transcript.txt --output report.html --format html
```

Produces a styled, shareable HTML report with findings grouped by step, color-coded by severity.

## Configuration

```python
guard = LoopGuard(config={
    # Claim extraction
    "use_llm_extraction": True,              # Enable LLM fallback extraction
    "extraction_model": "claude-haiku-4-5-20251001",  # Model for extraction

    # Verification
    "sandbox_dir": "/tmp/loopguard_sandbox", # Code execution sandbox
    "timeout": 60,                           # Sandbox timeout (seconds)

    # Loop trap detection
    "similarity_threshold": 0.8,             # Output similarity threshold
    "consecutive_limit": 3,                  # Consecutive similar outputs to trigger

    # Reporting
    "verbosity": "findings_only",            # all | findings_only | failures_only
})
```

## Architecture

```
loop_guard/
├── __init__.py          # Public API
├── models.py            # ClaimType, Verdict, Finding, etc.
├── extractor.py         # Regex-first claim extraction
├── engine.py            # Verification routing engine
├── reporter.py          # Terminal, JSON, HTML output
├── guard.py             # LoopGuard (main entry point)
├── cli.py               # CLI (loop-guard watch/check/report)
└── verifiers/
    ├── loop_trap.py     # Stuck loop detection
    ├── regression.py    # File regression detection
    ├── citation.py      # CrossRef + Semantic Scholar lookup
    ├── statistical.py   # Statistical sanity checks
    ├── code_output.py   # Code re-execution
    └── metric.py        # Metric re-computation
```

## What loop-guard Is NOT

- **NOT an LLM-as-judge system.** Those share the same failure modes as the agent being judged.
- **NOT a prompt injection detector.** Use dedicated security tools for that.
- **NOT a post-hoc evaluation tool.** It runs in-loop, catching errors as they happen.
- **NOT a replacement for human review.** It flags issues for humans to investigate.

## Examples

See the [`examples/`](examples/) directory:

- `autoresearch_demo.py` — ML experiment loop with metric/citation/loop-trap verification
- `openai_sdk_demo.py` — Coding agent integration pattern
- `adk_demo.py` — Data analysis agent with statistical verification

## Contributing

```bash
git clone https://github.com/ybkim95/loop-guard
cd loop-guard
pip install -e ".[dev]"
pytest
```

## License

MIT
