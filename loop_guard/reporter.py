import html as html_module
import json
from pathlib import Path

from loop_guard.models import Finding, Verdict


class Reporter:
    SEVERITY_ICONS = {
        Verdict.VERIFIED_FAIL: "FAIL",
        Verdict.RULE_VIOLATION: "WARN",
        Verdict.FLAG_FOR_REVIEW: "FLAG",
        Verdict.VERIFIED_PASS: "PASS",
        Verdict.SKIPPED: "SKIP",
    }

    def __init__(self, verbosity: str = "findings_only"):
        self.verbosity = verbosity
        self.all_findings: list[Finding] = []

    def report_step(self, findings: list[Finding]) -> None:
        self.all_findings.extend(findings)
        for f in findings:
            if self._should_display(f):
                self._print_finding(f)

    def _should_display(self, f: Finding) -> bool:
        if self.verbosity == "all":
            return True
        if self.verbosity == "failures_only":
            return f.verdict in (Verdict.VERIFIED_FAIL, Verdict.RULE_VIOLATION)
        return f.verdict not in (Verdict.VERIFIED_PASS, Verdict.SKIPPED)

    def _print_finding(self, f: Finding) -> None:
        icon = self.SEVERITY_ICONS[f.verdict]
        level_tag = f"L{f.level.value}"
        print(f"[loop-guard] Step {f.step_id} [{icon}] [{level_tag}] {f.explanation}")
        if f.expected and f.actual:
            print(f"            Expected: {f.expected}")
            print(f"            Actual:   {f.actual}")

    def generate_json_report(self, path: str) -> str:
        report = {
            "summary": self.summary(),
            "findings": [f.to_dict() for f in self.all_findings],
        }
        Path(path).write_text(json.dumps(report, indent=2, default=str))
        return path

    def generate_html_report(self, path: str) -> str:
        summary = self.summary()

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>LoopGuard Verification Report</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 960px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }",
            "h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }",
            "h2 { color: #8b949e; }",
            ".summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 20px 0; }",
            ".stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }",
            ".stat .number { font-size: 2em; font-weight: bold; }",
            ".stat .label { color: #8b949e; font-size: 0.85em; margin-top: 4px; }",
            ".fail .number { color: #f85149; }",
            ".warn .number { color: #d29922; }",
            ".flag .number { color: #a371f7; }",
            ".pass .number { color: #3fb950; }",
            ".skip .number { color: #8b949e; }",
            ".finding { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 8px 0; }",
            ".finding.verdict-verified_fail { border-left: 4px solid #f85149; }",
            ".finding.verdict-rule_violation { border-left: 4px solid #d29922; }",
            ".finding.verdict-flag_for_review { border-left: 4px solid #a371f7; }",
            ".finding.verdict-verified_pass { border-left: 4px solid #3fb950; }",
            ".finding.verdict-skipped { border-left: 4px solid #8b949e; }",
            ".badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; margin-right: 8px; }",
            ".badge-fail { background: #f8514922; color: #f85149; }",
            ".badge-warn { background: #d2992222; color: #d29922; }",
            ".badge-flag { background: #a371f722; color: #a371f7; }",
            ".badge-pass { background: #3fb95022; color: #3fb950; }",
            ".badge-skip { background: #8b949e22; color: #8b949e; }",
            ".meta { color: #8b949e; font-size: 0.85em; margin-top: 8px; }",
            ".claim-text { background: #0d1117; padding: 8px 12px; border-radius: 4px; margin: 8px 0; font-family: monospace; font-size: 0.9em; }",
            "</style>",
            "</head><body>",
            "<h1>LoopGuard Verification Report</h1>",
            '<div class="summary">',
        ]

        stats = [
            ("fail", "Failures", summary.get("verified_failures", 0)),
            ("warn", "Violations", summary.get("rule_violations", 0)),
            ("flag", "Flags", summary.get("flags_for_review", 0)),
            ("pass", "Passed", summary.get("verified_passes", 0)),
            ("skip", "Skipped", summary.get("skipped", 0)),
        ]
        for css_class, label, count in stats:
            html_parts.append(f'<div class="stat {css_class}"><div class="number">{count}</div><div class="label">{label}</div></div>')

        html_parts.append("</div>")
        html_parts.append("<h2>Findings</h2>")

        verdict_badge = {
            "verified_fail": ("FAIL", "fail"),
            "rule_violation": ("WARN", "warn"),
            "flag_for_review": ("FLAG", "flag"),
            "verified_pass": ("PASS", "pass"),
            "skipped": ("SKIP", "skip"),
        }

        for f in self.all_findings:
            v = f.verdict.value
            badge_text, badge_class = verdict_badge.get(v, ("?", "skip"))
            escaped_explanation = html_module.escape(f.explanation)
            escaped_claim = html_module.escape(f.claim.text)

            html_parts.append(f'<div class="finding verdict-{v}">')
            html_parts.append(f'<span class="badge badge-{badge_class}">{badge_text}</span>')
            html_parts.append(f'<strong>Step {f.step_id}</strong> &mdash; L{f.level.value}')
            html_parts.append(f'<div>{escaped_explanation}</div>')
            html_parts.append(f'<div class="claim-text">{escaped_claim}</div>')
            if f.expected and f.actual:
                html_parts.append(f'<div class="meta">Expected: {html_module.escape(str(f.expected))}<br>Actual: {html_module.escape(str(f.actual))}</div>')
            html_parts.append("</div>")

        html_parts.append("</body></html>")

        Path(path).write_text("\n".join(html_parts))
        return path

    def summary(self) -> dict:
        total = len(self.all_findings)
        by_verdict: dict[str, int] = {}
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
