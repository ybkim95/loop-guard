# loop-guard: Your Agent's Claims Are Wrong. Here's Proof.

*Deterministic verification for autonomous agent loops — not another LLM-as-judge.*

---

## The Problem Nobody's Measuring

Autonomous agent loops — autoresearch, coding agents, data science pipelines — run for hours without human oversight. During these loops, agents make hundreds of intermediate claims:

- "accuracy is 94%"
- "all tests pass"
- "Smith et al. 2024 showed..."
- "p < 0.05, statistically significant"

These claims compound. A wrong claim at step 23 becomes the premise for steps 24–100. Nobody catches the error until a human reviews the final output — if they review it at all.

**This isn't hypothetical.** In 2026:
- GPTZero found 50+ hallucinated citations in ICLR papers that peer reviewers missed
- DryRun Security found Claude Code, Codex, and Gemini all introduce broken access control in iterative coding sessions
- Autoresearch users have no way to know if experiment 73 is building on a flawed conclusion from experiment 41

## What if we just... checked?

Not with another LLM (that has the same failure modes as the agent being judged). With deterministic tools:

- **Re-run the code.** Agent says "this code outputs 42"? Run it. Check.
- **Look up the citation.** Agent cites "Smith et al. 2024"? Query CrossRef and Semantic Scholar. Is it real?
- **Check the math.** Agent reports "p = 1.5"? That's impossible. Flag it.
- **Detect loops.** Agent has produced 3 near-identical outputs in a row? It's stuck.

This is **loop-guard**: a framework-agnostic verification layer that attaches to any agent loop and performs deterministic checks on agent claims at each step.

## Real Results: We Ran It on Karpathy's Autoresearch

We analyzed the real results from Karpathy's first public autoresearch overnight run (126 experiments, ~10.5 hours of GPU time).

### What loop-guard found:

```
Total experiments: 126 (23 kept, 102 discarded, 1 crashed)
Keep rate: 18.3%
GPU time wasted on failed experiments: 8.6 hours

Longest unproductive streak: 25 experiments = 125 min
  (Steps 41–65: zero improvements)

Success rate alerts: 12
  (0% keep rate over 20-experiment windows)
```

### The val_bpb improvement trajectory:

```
Step   0: 0.997900  ★ baseline
Step   1: 0.986041  ▲ halve batch size (big win)
Step   2: 0.981773  ▲ extra layer
Step   8: 0.975524  ▲ embedding LR tuning
Step  13: 0.973104  ▲ unembedding LR
   ...then 26 discards...
Step  40: 0.972694    add weight decay
   ...then 25 discards (125 min wasted)...
Step  66: 0.972258    init scale
   ...then 25 more discards (125 min wasted)...
Step 118: 0.969686  ★ final best
```

**If loop-guard had been running:** Alert at step ~45 ("0% keep rate in last 20 experiments") → human intervenes, changes `program.md` → saves ~2 hours of unproductive GPU time. The agent reaches 0.9697 faster.

## Real Results: We Ran It on Gemini

We called the real Gemini 2.0 Flash API with a multi-step research workflow and loop-guard verifying every step:

```
7 Gemini API calls → 22 claims verified

VERIFIED_FAIL: 2
  - Citation "Dinges 1992" not found in CrossRef/Semantic Scholar
  - Citation format mismatch with actual publication

FLAG_FOR_REVIEW: 9
  - Multiple p-value comparisons without Bonferroni correction
  - Claims at step 3 TAINTED because they depend on step 2 (which had VERIFIED_FAIL)

Provenance: 10 dependency edges, max depth 2
  - Step 3 automatically flagged because step 2 was invalidated
```

**The provenance tracking is the key insight:** nobody else tracks causal dependencies between agent claims across loop iterations. When step N depends on step M and M is wrong, N is automatically flagged.

## How It Works

```
Agent Loop → Claim Extractor → Verification Engine → Reporter
               (regex-first)      (3 layers)

Layer 1 (Deterministic): Re-execute code, API lookups. Cannot be wrong.
Layer 2 (Rule-based):    Pattern matching, sanity checks. Rarely wrong.
Layer 3 (LLM-assisted):  Soft flagging only. May be wrong.
```

**Core principle:** Verification must be more reliable than the thing being verified. LLMs are used only for claim extraction (a structured task), never for judgment.

### 8 Verifiers

| Verifier | Layer | What it catches |
|----------|-------|----------------|
| LoopTrapVerifier | L2 | Agent stuck retrying the same approach |
| RegressionVerifier | L2 | Agent reverts code to a previous version |
| CitationVerifier | L1 | Hallucinated citations (CrossRef + Semantic Scholar) |
| StatisticalVerifier | L2 | Impossible p-values, missing corrections, small samples |
| CodeOutputVerifier | L1 | Code re-execution produces different output |
| MetricVerifier | L1 | Metric re-computation doesn't match claim |
| ToolOutputVerifier | L1 | Tool re-execution produces different result |
| ProvenanceChain | L2 | Downstream claims tainted by upstream failures |

## Integration: 2 Lines of Code

```python
from loop_guard import LoopGuard

guard = LoopGuard()

# Works with ANY agent loop
for step in agent.run(task):
    findings = guard.step(output=step.text)
    for f in findings:
        if f.verdict == "verified_fail":
            print(f"ALERT: {f.explanation}")
```

### Framework-specific wrappers

```python
# Google Gemini / ADK
from loop_guard.integrations.google_adk import GeminiGuard
guard = GeminiGuard(api_key="...")
result = guard.generate("Analyze this data...")

# OpenAI
from loop_guard.integrations.openai_agents import OpenAIGuard
guard = OpenAIGuard(api_key="sk-...")

# Anthropic Claude
from loop_guard.integrations.anthropic_sdk import AnthropicGuard
guard = AnthropicGuard(api_key="sk-ant-...")

# Autoresearch
loop-guard autoresearch ./autoresearch/ --poll 30
```

### CLI (zero code change)

```bash
# Pipe any agent's stdout
python my_agent.py | loop-guard watch

# Watch autoresearch
loop-guard autoresearch ./experiments/ --poll 30

# Check a transcript
loop-guard check --input agent_log.txt
```

## Install

```bash
pip install loop-guard
```

## Why Not LLM-as-Judge?

The dominant approach today is using one LLM to evaluate another LLM's output. This is fundamentally limited because the judge shares the same failure modes as the agent:

- LLMs hallucinate citations → an LLM judge won't catch hallucinated citations
- LLMs make statistical errors → an LLM judge will accept wrong p-values
- LLMs can't re-run code → an LLM judge can't verify execution results

loop-guard's design deliberately maximizes deterministic verification (Layer 1) and rule-based checking (Layer 2). The LLM is used only for structured claim extraction — never for judgment.

## What's Next

- **NeurIPS 2026 submission** with controlled A/B experiments: same agent, same tasks, with vs without loop-guard
- **OpenTelemetry integration** for automatic compatibility with every agent framework
- **More verifiers:** DataIntegrityVerifier, PermissionVerifier, CostVerifier

## Links

- **GitHub:** https://github.com/ybkim95/loop-guard
- **PyPI:** `pip install loop-guard`
- **83 tests passing**, framework-agnostic, Python 3.10+

---

*loop-guard catches silent errors in agent loops by re-running code, looking up citations, and checking statistics — not by asking another LLM if the output "looks right."*
