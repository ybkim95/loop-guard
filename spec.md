# loop-guard: Deterministic Verification for Autonomous Agent Loops

## Project Specification v0.1

---

## 1. What LoopGuard Is (and Is Not)

### The Problem

Autonomous agent loops (autoresearch, coding agents, data science pipelines) run for hours without human oversight. During these loops, agents make intermediate claims: "accuracy is 94%", "this paper supports my argument", "tests pass", "p < 0.05". These claims compound. A wrong claim at step 23 becomes the premise for steps 24-100. Nobody catches the error until a human reviews the final output, if at all.

Real examples from 2026:
- GPTZero found 50+ hallucinated citations in ICLR 2026 papers that 3-5 peer reviewers missed
- OpenClaw agent bulk-deleted a Meta AI researcher's emails after losing safety instructions during context compaction
- DryRun Security found Claude Code, Codex, and Gemini all introduce broken access control, business logic failures, and OAuth flaws in iterative coding sessions
- Autoresearch users have no way to know if experiment 73 is building on a flawed conclusion from experiment 41

### What LoopGuard Does

LoopGuard is a framework-agnostic verification layer that attaches to any agent loop and performs deterministic checks on agent claims at each step. It does NOT use LLMs to judge LLMs. It re-executes code, re-runs tests, re-calculates statistics, and looks up citations using deterministic tools.

### What LoopGuard Is NOT

- NOT an LLM-as-judge system (those are unreliable by design: same failure modes as the agent being judged)
- NOT a prompt injection detector or safety filter
- NOT a post-hoc evaluation tool (it runs in-loop, not after the loop)
- NOT a replacement for human review (it flags issues for humans to review)

### Core Design Principle

**Verification must be more reliable than the thing being verified.** This means:
- Layer 1 (deterministic): Code re-execution, API lookups, numerical re-computation. Cannot be wrong.
- Layer 2 (rule-based): Pattern matching, statistical sanity checks, known anti-patterns. Rarely wrong.
- Layer 3 (LLM-assisted): Only for flagging, never for verdict. Explicitly marked as "needs human review".

If a check cannot be deterministic or rule-based, it belongs in Layer 3 and its output is a flag, not a finding.

---

## 2. Architecture

### 2.1 Overview

```
┌──────────────────────────────────────────────────────┐
│                   Any Agent Loop                      │
│  (autoresearch / Google ADK / OpenAI SDK / custom)   │
└──────────────────┬───────────────────────────────────┘
                   │ step output (text, code, data)
                   ▼
┌──────────────────────────────────────────────────────┐
│              Integration Layer                        │
│  Adapter A: callback wrapper (2 lines of code)       │
│  Adapter B: stdout/log pipe (zero code change)       │
│  Adapter C: git commit watcher (for autoresearch)    │
│  Adapter D: OpenTelemetry collector (v0.2)           │
└──────────────────┬───────────────────────────────────┘
                   │ NormalizedStep
                   ▼
┌──────────────────────────────────────────────────────┐
│              Claim Extractor                          │
│  Parses step output into typed, verifiable claims    │
│  e.g. CodeClaim, StatsClaim, CitationClaim,          │
│       MetricClaim, TestClaim                         │
└──────────────────┬───────────────────────────────────┘
                   │ list[Claim]
                   ▼
┌──────────────────────────────────────────────────────┐
│              Verification Engine                      │
│  Routes each Claim to the appropriate Verifier       │
│  Layer 1: deterministic (re-execute, re-compute)     │
│  Layer 2: rule-based (pattern checks)                │
│  Layer 3: LLM-assisted (flag only, never verdict)    │
└──────────────────┬───────────────────────────────────┘
                   │ list[Finding]
                   ▼
┌──────────────────────────────────────────────────────┐
│              Reporter                                 │
│  Terminal alerts, JSON log, HTML report               │
│  Each finding tagged: VERIFIED_FAIL / VERIFIED_PASS  │
│                       / FLAG_FOR_REVIEW              │
└──────────────────────────────────────────────────────┘
```

### 2.2 Data Model

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class ClaimType(Enum):
    CODE_OUTPUT = "code_output"       # "I ran X, got Y"
    METRIC = "metric"                 # "accuracy is 94%"
    STATISTICAL = "statistical"       # "p < 0.05, significant"
    CITATION = "citation"             # "Smith et al. 2024 showed..."
    TEST_RESULT = "test_result"       # "all tests pass"
    FILE_STATE = "file_state"         # "I modified file X"
    GENERAL = "general"               # untyped claim (Layer 3 only)


class VerificationLevel(Enum):
    DETERMINISTIC = 1    # re-execution, API lookup: cannot be wrong
    RULE_BASED = 2       # pattern matching, sanity check: rarely wrong
    LLM_ASSISTED = 3     # soft flag: may be wrong, needs human review


class Verdict(Enum):
    VERIFIED_PASS = "verified_pass"         # deterministic check passed
    VERIFIED_FAIL = "verified_fail"         # deterministic check failed
    RULE_VIOLATION = "rule_violation"       # rule-based check flagged
    FLAG_FOR_REVIEW = "flag_for_review"     # LLM flag, uncertain
    SKIPPED = "skipped"                     # could not verify (no verifier)


@dataclass
class NormalizedStep:
    step_id: int
    timestamp: float
    raw_output: str                         # full agent output
    code_executed: Optional[str] = None     # if agent ran code
    files_modified: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Claim:
    claim_type: ClaimType
    source_step: int
    text: str                               # the literal claim
    verifiable: bool = False                # can this be deterministically checked?
    evidence: Optional[dict] = None         # supporting data (code, numbers, etc.)


@dataclass
class Finding:
    step_id: int
    claim: Claim
    verdict: Verdict
    level: VerificationLevel
    explanation: str
    expected: Optional[str] = None          # what should have been
    actual: Optional[str] = None            # what was found
    timestamp: float = field(default_factory=time.time)
```

### 2.3 Integration Layer

LoopGuard does NOT wrap the agent. It reads the agent's output. Three integration modes (implement all three in v0.1):

#### Adapter A: Python Callback (most frameworks)

```python
from loopguard import LoopGuard

guard = LoopGuard()

# Works with any loop
for i in range(num_experiments):
    result = agent.run(task)
    findings = guard.step(
        step_id=i,
        output=result.text,
        code=result.code_executed,       # optional
        files=result.files_modified,     # optional
    )
    for f in findings:
        print(f)  # or handle programmatically
```

This works with Google ADK, OpenAI Agents SDK, Anthropic SDK, LangGraph, CrewAI, or any custom loop. LoopGuard never imports these frameworks. It only needs the text output.

#### Adapter B: CLI Pipe (zero code change)

```bash
# Pipe any agent's stdout
python my_agent.py 2>&1 | loopguard watch

# Watch a log file
loopguard watch --file agent.log --follow

# Watch git commits (for autoresearch)
loopguard watch --git-dir ./autoresearch/ --poll 30
```

The CLI adapter splits input into steps using configurable delimiters (double newline, timestamps, git commits, JSON boundaries).

#### Adapter C: Structured JSON Input

For frameworks that can emit structured logs:

```python
guard.step(
    step_id=42,
    output="Model accuracy improved to 94.2%",
    code="model.evaluate(test_data)",
    files=["train.py"],
    metadata={"val_bpb": 0.9697}
)
```

---

## 3. Claim Extraction

The Claim Extractor parses raw step output into typed claims. This is the only part that uses an LLM call, and it is a structured extraction task (not a judgment task).

### 3.1 Extraction Prompt (structured output)

```
Given the following agent step output, extract all verifiable claims.
For each claim, identify:
- type: one of [code_output, metric, statistical, citation, test_result, file_state, general]
- text: the exact claim as stated
- verifiable: true if this can be checked by re-running code, looking up a database, or re-computing a number
- evidence: any code, numbers, file paths, or references needed to verify

Output as JSON array. Only extract explicit claims, not implications.

Agent output:
{step_output}
```

### 3.2 Rule-Based Extraction Fallbacks

For common patterns, skip the LLM entirely:

```python
import re

# Citation pattern: "Author et al., YYYY" or "(Author, YYYY)"
CITATION_PATTERN = r'([A-Z][a-z]+(?:\s+(?:et\s+al\.|and\s+[A-Z][a-z]+))?[\s,]+(?:19|20)\d{2})'

# Metric pattern: "accuracy/precision/recall/f1/loss/bpb = X.XX" or "is X.XX%"
METRIC_PATTERN = r'((?:accuracy|precision|recall|f1|loss|val_bpb|auc|rmse|mae)[\s:=]+[\d.]+%?)'

# P-value pattern
PVALUE_PATTERN = r'(p[\s<>=]+[\d.]+(?:e-?\d+)?)'

# Test result pattern
TEST_PATTERN = r'((?:all\s+)?tests?\s+(?:pass|fail|passed|failed)|(\d+)\s+(?:passed|failed))'
```

These regex extractors run first. The LLM extractor runs only on the remaining text that regex did not cover. This keeps LLM cost low and extraction reliable.

---

## 4. Verifiers (The Core)

Each verifier handles one claim type. All Layer 1 verifiers are deterministic. The verification engine routes claims to the appropriate verifier based on ClaimType.

### 4.1 CodeOutputVerifier (Layer 1: Deterministic)

**What it checks:** Agent says "I ran this code and got this result." Verifier re-runs the code in a sandbox and compares.

```python
class CodeOutputVerifier:
    """Re-executes code and compares output to claimed result."""

    def __init__(self, sandbox_dir: str, timeout: int = 60):
        self.sandbox_dir = sandbox_dir
        self.timeout = timeout

    def verify(self, claim: Claim) -> Finding:
        code = claim.evidence.get("code")
        claimed_output = claim.evidence.get("claimed_output")

        if not code or not claimed_output:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.SKIPPED,
                level=VerificationLevel.DETERMINISTIC,
                explanation="Missing code or claimed output for re-execution"
            )

        actual_output = self._execute_in_sandbox(code)

        if self._outputs_match(claimed_output, actual_output):
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation="Re-execution produced matching output",
                expected=claimed_output,
                actual=actual_output
            )
        else:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_FAIL,
                level=VerificationLevel.DETERMINISTIC,
                explanation="Re-execution produced different output",
                expected=claimed_output,
                actual=actual_output
            )

    def _execute_in_sandbox(self, code: str) -> str:
        """Run code in isolated subprocess with timeout."""
        # Use subprocess with timeout, restricted filesystem access
        # Return stdout + captured return value
        ...

    def _outputs_match(self, claimed: str, actual: str, tolerance: float = 0.01) -> bool:
        """Compare outputs with numeric tolerance for floating point."""
        # Try numeric comparison first (within tolerance)
        # Fall back to string comparison
        ...
```

**Scope:** Applicable when agent output contains executable code and a stated result. Common in autoresearch (every experiment), coding agents (test execution), data science pipelines (computation results).

### 4.2 CitationVerifier (Layer 1: Deterministic)

**What it checks:** Agent cites a paper. Verifier looks it up in CrossRef and Semantic Scholar.

```python
class CitationVerifier:
    """Checks if cited papers exist using CrossRef and Semantic Scholar APIs."""

    CROSSREF_API = "https://api.crossref.org/works"
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

    def verify(self, claim: Claim) -> Finding:
        citation_text = claim.text  # e.g., "Smith et al., 2024"

        # Extract author and year
        author, year = self._parse_citation(citation_text)
        title = claim.evidence.get("title")  # if available

        # Check CrossRef first
        found_crossref = self._search_crossref(author, year, title)
        if found_crossref:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Paper found in CrossRef: {found_crossref['title']}"
            )

        # Check Semantic Scholar
        found_ss = self._search_semantic_scholar(author, year, title)
        if found_ss:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.VERIFIED_PASS,
                level=VerificationLevel.DETERMINISTIC,
                explanation=f"Paper found in Semantic Scholar: {found_ss['title']}"
            )

        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.VERIFIED_FAIL,
            level=VerificationLevel.DETERMINISTIC,
            explanation="Citation not found in CrossRef or Semantic Scholar",
            expected=citation_text,
            actual="No matching paper found"
        )

    def _search_crossref(self, author: str, year: int, title: str = None) -> dict | None:
        """Query CrossRef API. Rate limit: polite pool with mailto."""
        ...

    def _search_semantic_scholar(self, author: str, year: int, title: str = None) -> dict | None:
        """Query Semantic Scholar API."""
        ...

    def _parse_citation(self, text: str) -> tuple[str, int]:
        """Extract first author surname and year from citation text."""
        ...
```

**Scope:** Any agent output that references academic papers. Extremely common in research agents, literature review, paper writing.

### 4.3 StatisticalVerifier (Layer 2: Rule-Based)

**What it checks:** Agent reports statistical results. Verifier checks for common errors using deterministic rules.

```python
class StatisticalVerifier:
    """Checks statistical claims against known rules and best practices."""

    RULES = [
        {
            "name": "multiple_comparison_correction",
            "description": "Multiple p-values reported without correction",
            "check": "_check_multiple_comparisons"
        },
        {
            "name": "sample_size_adequacy",
            "description": "Claimed effect with very small sample",
            "check": "_check_sample_size"
        },
        {
            "name": "impossible_values",
            "description": "Statistically impossible values (e.g., p > 1, negative variance)",
            "check": "_check_impossible_values"
        },
    ]

    def verify(self, claim: Claim, step_history: list[NormalizedStep]) -> Finding:
        violations = []
        for rule in self.RULES:
            check_fn = getattr(self, rule["check"])
            violation = check_fn(claim, step_history)
            if violation:
                violations.append(violation)

        if violations:
            return Finding(
                step_id=claim.source_step,
                claim=claim,
                verdict=Verdict.RULE_VIOLATION,
                level=VerificationLevel.RULE_BASED,
                explanation="; ".join(violations)
            )
        return Finding(
            step_id=claim.source_step,
            claim=claim,
            verdict=Verdict.VERIFIED_PASS,
            level=VerificationLevel.RULE_BASED,
            explanation="No statistical rule violations detected"
        )

    def _check_multiple_comparisons(self, claim, history) -> str | None:
        """If >1 p-value in recent steps without mention of Bonferroni/BH/FDR correction."""
        pvalues_in_window = self._count_pvalue_claims(history[-20:])
        has_correction = self._mentions_correction(claim.text)
        if pvalues_in_window > 1 and not has_correction:
            return f"Found {pvalues_in_window} p-value claims without multiple comparison correction"
        return None

    def _check_sample_size(self, claim, history) -> str | None:
        """Flag if sample size < 30 for parametric claims."""
        ...

    def _check_impossible_values(self, claim, history) -> str | None:
        """Flag p > 1, R^2 > 1, negative variance, accuracy > 100%, etc."""
        ...
```

### 4.4 RegressionVerifier (Layer 2: Rule-Based)

**What it checks:** Agent re-introduces a bug or reverts a fix from a previous step.

```python
class RegressionVerifier:
    """Detects when agent re-introduces previously fixed issues."""

    def __init__(self):
        self.file_snapshots: dict[str, list[tuple[int, str]]] = {}  # file -> [(step, content)]

    def verify(self, step: NormalizedStep) -> list[Finding]:
        findings = []
        for filepath in step.files_modified:
            current_content = self._read_file(filepath)
            history = self.file_snapshots.get(filepath, [])

            # Check if current content matches any version older than the previous one
            # (i.e., we went back to an earlier state)
            if len(history) >= 2:
                for old_step, old_content in history[:-1]:  # everything except the immediate predecessor
                    similarity = self._compute_similarity(current_content, old_content)
                    if similarity > 0.95:  # near-identical to an older version
                        findings.append(Finding(
                            step_id=step.step_id,
                            claim=Claim(
                                claim_type=ClaimType.FILE_STATE,
                                source_step=step.step_id,
                                text=f"Modified {filepath}",
                            ),
                            verdict=Verdict.RULE_VIOLATION,
                            level=VerificationLevel.RULE_BASED,
                            explanation=f"File {filepath} reverted to state from step {old_step} (regression)",
                            expected="Forward progress",
                            actual=f"95%+ similarity with step {old_step} version"
                        ))

            # Update snapshot
            history.append((step.step_id, current_content))
            self.file_snapshots[filepath] = history[-50:]  # keep last 50 versions
        return findings

    def _compute_similarity(self, a: str, b: str) -> float:
        """Fast similarity using difflib.SequenceMatcher."""
        ...
```

### 4.5 LoopTrapVerifier (Layer 2: Rule-Based)

**What it checks:** Agent is stuck retrying the same failing approach.

```python
class LoopTrapVerifier:
    """Detects when agent is stuck in a retry loop."""

    def __init__(self, similarity_threshold: float = 0.8, consecutive_limit: int = 3):
        self.recent_outputs: list[tuple[int, str]] = []
        self.similarity_threshold = similarity_threshold
        self.consecutive_limit = consecutive_limit

    def verify(self, step: NormalizedStep) -> Finding | None:
        # Compare current output to recent outputs
        consecutive_similar = 0
        for prev_step, prev_output in reversed(self.recent_outputs):
            sim = self._compute_similarity(step.raw_output, prev_output)
            if sim > self.similarity_threshold:
                consecutive_similar += 1
            else:
                break

        self.recent_outputs.append((step.step_id, step.raw_output))
        self.recent_outputs = self.recent_outputs[-20:]  # sliding window

        if consecutive_similar >= self.consecutive_limit:
            return Finding(
                step_id=step.step_id,
                claim=Claim(
                    claim_type=ClaimType.GENERAL,
                    source_step=step.step_id,
                    text="Agent output",
                ),
                verdict=Verdict.RULE_VIOLATION,
                level=VerificationLevel.RULE_BASED,
                explanation=f"Agent appears stuck: {consecutive_similar + 1} consecutive similar outputs "
                            f"(similarity > {self.similarity_threshold})"
            )
        return None
```

### 4.6 MetricVerifier (Layer 1: Deterministic)

**What it checks:** Agent claims a metric value. If the code/data to recompute it is available, re-compute.

```python
class MetricVerifier:
    """Re-computes claimed metrics when computation is reproducible."""

    def verify(self, claim: Claim, sandbox_dir: str) -> Finding:
        metric_name = claim.evidence.get("metric_name")  # e.g., "val_bpb"
        claimed_value = claim.evidence.get("claimed_value")  # e.g., 0.9697
        computation_code = claim.evidence.get("code")

        if computation_code and claimed_value is not None:
            actual_value = self._recompute(computation_code, sandbox_dir)
            if actual_value is not None:
                if self._values_match(float(claimed_value), actual_value):
                    return self._pass(claim, claimed_value, actual_value)
                else:
                    return self._fail(claim, claimed_value, actual_value)

        return self._skip(claim, "Cannot re-compute: missing code or data")

    def _recompute(self, code: str, sandbox_dir: str) -> float | None:
        """Execute metric computation in sandbox, return numeric result."""
        ...

    def _values_match(self, claimed: float, actual: float, rtol: float = 0.01) -> bool:
        """Relative tolerance comparison for floating point metrics."""
        if actual == 0:
            return abs(claimed) < 1e-9
        return abs(claimed - actual) / abs(actual) <= rtol
```

---

## 5. Verification Engine (Router)

```python
class VerificationEngine:
    """Routes claims to appropriate verifiers and manages verification state."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.code_verifier = CodeOutputVerifier(
            sandbox_dir=self.config.get("sandbox_dir", "/tmp/loopguard_sandbox")
        )
        self.citation_verifier = CitationVerifier()
        self.statistical_verifier = StatisticalVerifier()
        self.regression_verifier = RegressionVerifier()
        self.loop_trap_verifier = LoopTrapVerifier()
        self.metric_verifier = MetricVerifier()
        self.step_history: list[NormalizedStep] = []

    def verify_step(self, step: NormalizedStep, claims: list[Claim]) -> list[Finding]:
        self.step_history.append(step)
        findings = []

        # Always run structural verifiers (no claims needed)
        regression_findings = self.regression_verifier.verify(step)
        findings.extend(regression_findings)

        trap_finding = self.loop_trap_verifier.verify(step)
        if trap_finding:
            findings.append(trap_finding)

        # Route each claim to its verifier
        for claim in claims:
            finding = self._route_claim(claim)
            findings.append(finding)

        return findings

    def _route_claim(self, claim: Claim) -> Finding:
        match claim.claim_type:
            case ClaimType.CODE_OUTPUT:
                return self.code_verifier.verify(claim)
            case ClaimType.CITATION:
                return self.citation_verifier.verify(claim)
            case ClaimType.STATISTICAL:
                return self.statistical_verifier.verify(claim, self.step_history)
            case ClaimType.METRIC:
                return self.metric_verifier.verify(
                    claim, self.config.get("sandbox_dir", "/tmp/loopguard_sandbox")
                )
            case ClaimType.TEST_RESULT:
                return self.code_verifier.verify(claim)  # re-run tests
            case _:
                return Finding(
                    step_id=claim.source_step,
                    claim=claim,
                    verdict=Verdict.SKIPPED,
                    level=VerificationLevel.LLM_ASSISTED,
                    explanation="No deterministic verifier available for this claim type"
                )
```

---

## 6. Reporter

```python
class Reporter:
    """Outputs findings to terminal, JSON log, and HTML report."""

    SEVERITY_ICONS = {
        Verdict.VERIFIED_FAIL: "FAIL",
        Verdict.RULE_VIOLATION: "WARN",
        Verdict.FLAG_FOR_REVIEW: "FLAG",
        Verdict.VERIFIED_PASS: "PASS",
        Verdict.SKIPPED: "SKIP",
    }

    def __init__(self, verbosity: str = "findings_only"):
        # verbosity: "all" | "findings_only" | "failures_only"
        self.verbosity = verbosity
        self.all_findings: list[Finding] = []

    def report_step(self, findings: list[Finding]):
        """Print findings to terminal in real-time."""
        self.all_findings.extend(findings)
        for f in findings:
            if self._should_display(f):
                self._print_finding(f)

    def _should_display(self, f: Finding) -> bool:
        if self.verbosity == "all":
            return True
        if self.verbosity == "failures_only":
            return f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION)
        # findings_only: everything except PASS and SKIP
        return f.verdict not in (Verdict.VERIFIED_PASS, Verdict.SKIPPED)

    def _print_finding(self, f: Finding):
        icon = self.SEVERITY_ICONS[f.verdict]
        level_tag = f"L{f.level.value}"
        print(f"[loop-guard] Step {f.step_id} [{icon}] [{level_tag}] {f.explanation}")
        if f.expected and f.actual:
            print(f"            Expected: {f.expected}")
            print(f"            Actual:   {f.actual}")

    def generate_json_report(self, path: str):
        """Write all findings as JSON for programmatic consumption."""
        ...

    def generate_html_report(self, path: str):
        """Generate shareable HTML audit report."""
        ...

    def summary(self) -> dict:
        """Return summary statistics."""
        total = len(self.all_findings)
        by_verdict = {}
        for f in self.all_findings:
            by_verdict[f.verdict.value] = by_verdict.get(f.verdict.value, 0) + 1
        return {
            "total_claims_checked": total,
            "verified_failures": by_verdict.get("verified_fail", 0),
            "rule_violations": by_verdict.get("rule_violation", 0),
            "flags_for_review": by_verdict.get("flag_for_review", 0),
            "verified_passes": by_verdict.get("verified_pass", 0),
            "skipped": by_verdict.get("skipped", 0),
        }
```

---

## 7. Top-Level API

```python
class LoopGuard:
    """Main entry point. Attach to any agent loop."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.extractor = ClaimExtractor(
            use_llm=self.config.get("use_llm_extraction", True),
            llm_model=self.config.get("extraction_model", "claude-haiku-4-5-20251001")
        )
        self.engine = VerificationEngine(config=self.config)
        self.reporter = Reporter(
            verbosity=self.config.get("verbosity", "findings_only")
        )
        self._step_counter = 0

    def step(
        self,
        output: str,
        step_id: int = None,
        code: str = None,
        files: list[str] = None,
        metadata: dict = None
    ) -> list[Finding]:
        """Process one agent step. Call this after each iteration."""
        if step_id is None:
            step_id = self._step_counter
            self._step_counter += 1

        normalized = NormalizedStep(
            step_id=step_id,
            timestamp=time.time(),
            raw_output=output,
            code_executed=code,
            files_modified=files or [],
            metadata=metadata or {}
        )

        claims = self.extractor.extract(normalized)
        findings = self.engine.verify_step(normalized, claims)
        self.reporter.report_step(findings)
        return findings

    def report(self, format: str = "terminal") -> str | dict:
        """Generate final report."""
        if format == "json":
            return self.reporter.generate_json_report("loopguard_report.json")
        elif format == "html":
            return self.reporter.generate_html_report("loopguard_report.html")
        else:
            return self.reporter.summary()
```

---

## 8. CLI

```bash
# Install
pip install loop-guard

# Watch agent stdout
python my_agent.py | loop-guard watch

# Watch log file
loop-guard watch --file agent.log --follow

# Watch git commits (autoresearch mode)
loop-guard watch --git ./autoresearch/ --poll 30

# Generate report from existing log
loop-guard report --input loop_guard_findings.json --format html

# Run on a single file (batch mode)
loop-guard check --input agent_transcript.txt
```

---

## 9. Implementation Plan

### Phase 1: Core MVP (Week 1-2)

Target: "loopguard works on autoresearch and catches real issues"

Files to create:
```
loop_guard/
  __init__.py              # exports LoopGuard
  models.py                # dataclasses from section 2.2
  extractor.py             # ClaimExtractor (regex-first, LLM fallback)
  verifiers/
    __init__.py
    code_output.py         # CodeOutputVerifier (sandbox re-execution)
    citation.py            # CitationVerifier (CrossRef + Semantic Scholar)
    regression.py          # RegressionVerifier (file diff tracking)
    loop_trap.py           # LoopTrapVerifier (output similarity)
    statistical.py         # StatisticalVerifier (rule-based checks)
    metric.py              # MetricVerifier (numeric re-computation)
  engine.py                # VerificationEngine (router)
  reporter.py              # Reporter (terminal + JSON + HTML)
  guard.py                 # LoopGuard (top-level API)
  cli.py                   # CLI entry point
tests/
  test_citation.py         # test with known real + fake papers
  test_regression.py       # test with known regression sequences
  test_loop_trap.py        # test with known stuck loops
  test_statistical.py      # test with known statistical errors
  test_integration.py      # end-to-end test with mock agent loop
examples/
  autoresearch_demo.py     # LoopGuard on autoresearch
  openai_sdk_demo.py       # LoopGuard on OpenAI Agents SDK
  adk_demo.py              # LoopGuard on Google ADK
pyproject.toml
README.md
```

Priority order for verifiers:
1. LoopTrapVerifier (easiest, highest immediate value, no external deps)
2. RegressionVerifier (easy, high value for coding agents)
3. CitationVerifier (moderate, very visible impact, needs API keys)
4. StatisticalVerifier (moderate, high value for data science)
5. CodeOutputVerifier + MetricVerifier (hardest, needs sandboxing)

### Phase 2: Multi-Framework Demos (Week 3-4)

- Run on autoresearch (git watcher mode) with real overnight experiments
- Run on an OpenAI Agents SDK coding task
- Run on a Google ADK data analysis agent
- Produce comparison: "issues caught by LoopGuard vs issues found by human review"

### Phase 3: Evaluation and Paper (Week 5-8)

Research questions to answer empirically:
1. What percentage of agent claims are deterministically verifiable? (by domain)
2. What is LoopGuard's precision and recall on seeded errors?
3. What is the overhead (latency, cost) per step?
4. How does verification coverage change with step count (1, 10, 50, 100, 500)?
5. Does LoopGuard meaningfully improve final output quality?

Experimental design:
- Autoresearch: 200 runs (100 with LoopGuard, 100 without). Compare final val_bpb and count of "silent errors" found by manual post-hoc audit.
- Coding agent: 50 SWE-bench tasks. Count security vulnerabilities in final code (with vs without LoopGuard intervention).
- Data science pipeline: 30 biomarker discovery runs. Count statistical errors in final reports.

---

## 10. What Makes This Rigorous (Not a Toy)

### The Research Contribution Is NOT "We Built a Tool"

The contribution is the empirical answer to three questions nobody has measured:

**Q1: How do errors compound in autonomous loops?**
We run hundreds of agent loops and taxonomize the errors that occur, when they occur, and how they propagate. This is the first systematic measurement of error compounding in agentic systems.

**Q2: What fraction of agent claims are deterministically verifiable?**
We measure this across three domains (ML research, coding, data science). This tells the field what the ceiling is for deterministic verification, and where LLM-based methods are unavoidable.

**Q3: Does in-loop verification improve final outcomes?**
Controlled experiment: same agent, same tasks, with and without LoopGuard. If verification catches errors early, does the agent produce better final results? By how much? At what cost?

### Why LLM-as-Judge Approaches Would Fail Here

The core insight (and the argument in the paper): using an LLM to judge an LLM's trajectory is fundamentally limited because the judge shares the same failure modes as the agent. LoopGuard's design deliberately maximizes deterministic verification (Layer 1) and rule-based checking (Layer 2), using LLMs only for claim extraction (a structured task) and soft flagging (explicitly marked as uncertain). The paper should include ablation: Layer 1+2 only vs Layer 1+2+3 vs LLM-judge-only baseline, showing that deterministic verification catches errors that LLM judges miss.

### Threat to Validity

Be honest about these in the paper:
- Claim extraction uses an LLM, introducing a potential failure point. Mitigation: regex-first pipeline, LLM only for residual. Report extraction recall separately.
- CodeOutputVerifier assumes deterministic execution. Non-deterministic code (random seeds, network calls) will produce false positives. Mitigation: configurable tolerance, whitelist for known non-deterministic operations.
- CitationVerifier depends on external APIs (CrossRef, Semantic Scholar). Papers not indexed will be false positives. Mitigation: report API coverage rates.
- LoopGuard adds latency and cost per step. Report this honestly and let users configure check frequency (every step vs every N steps).

---

## 11. Naming and Positioning

**Name:** loop-guard (PyPI: `loop-guard`, import: `loop_guard`, CLI: `loop-guard`)

**Tagline:** "Deterministic verification for autonomous agent loops"

**One-liner for README:** loop-guard catches silent errors in agent loops by re-running code, looking up citations, and checking statistics, not by asking another LLM if the output looks right.

**Target venues for paper:**
- Primary: NeurIPS 2026 (deadline ~May 2026, fits the timeline)
- Alternative: EMNLP 2026, AAAI 2027
- Workshop: ICLR 2026 Workshop on Recursive Self-Improvement (if still accepting late submissions)

**Positioning vs existing work:**
- vs Langfuse/Evidently/LangSmith: these are observability platforms (logging and dashboards). loop-guard is active verification (re-execution and checking).
- vs LLM-as-judge: loop-guard prioritizes deterministic verification. LLM is used only for extraction, never for judgment.
- vs Guardrails.md: manual rules written by humans. loop-guard auto-extracts and auto-verifies claims.
- vs autoresearch: complementary. loop-guard is the safety net that autoresearch currently lacks.