"""CLI for loop-guard."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from loop_guard.guard import LoopGuard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loop-guard",
        description="Deterministic verification for autonomous agent loops",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # watch command
    watch_parser = subparsers.add_parser("watch", help="Watch agent output in real-time")
    watch_parser.add_argument("--file", type=str, help="Watch a log file instead of stdin")
    watch_parser.add_argument("--follow", "-f", action="store_true", help="Follow file (like tail -f)")
    watch_parser.add_argument("--git-dir", type=str, help="Watch git commits in a directory")
    watch_parser.add_argument("--poll", type=int, default=10, help="Poll interval in seconds (for --git-dir)")
    watch_parser.add_argument("--delimiter", type=str, default="\n\n", help="Step delimiter")
    watch_parser.add_argument("--verbosity", choices=["all", "findings_only", "failures_only"], default="findings_only")
    watch_parser.add_argument("--no-llm", action="store_true", help="Disable LLM-based claim extraction")
    watch_parser.add_argument("--output", type=str, help="Output JSON report path")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate report from findings JSON")
    report_parser.add_argument("--input", "-i", type=str, required=True, help="Input findings JSON file")
    report_parser.add_argument("--format", choices=["html", "json", "terminal"], default="html")
    report_parser.add_argument("--output", "-o", type=str, help="Output file path")

    # check command
    check_parser = subparsers.add_parser("check", help="Check a single transcript file")
    check_parser.add_argument("--input", "-i", type=str, required=True, help="Agent transcript file")
    check_parser.add_argument("--delimiter", type=str, default="\n\n", help="Step delimiter")
    check_parser.add_argument("--verbosity", choices=["all", "findings_only", "failures_only"], default="findings_only")
    check_parser.add_argument("--no-llm", action="store_true", help="Disable LLM extraction")
    check_parser.add_argument("--output", type=str, help="Output report path")
    check_parser.add_argument("--format", choices=["html", "json", "terminal"], default="terminal")

    # autoresearch command
    ar_parser = subparsers.add_parser(
        "autoresearch", help="Monitor autoresearch experiments via results.tsv"
    )
    ar_parser.add_argument("dir", type=str, help="Autoresearch project directory")
    ar_parser.add_argument("--poll", type=int, default=30, help="Poll interval in seconds")
    ar_parser.add_argument(
        "--plateau-window", type=int, default=10, help="Number of experiments for plateau detection"
    )
    ar_parser.add_argument(
        "--plateau-threshold", type=float, default=0.0001, help="Minimum improvement threshold"
    )
    ar_parser.add_argument("--crash-limit", type=int, default=5, help="Consecutive crashes to flag")
    ar_parser.add_argument(
        "--check", action="store_true", help="One-shot check instead of watching"
    )
    ar_parser.add_argument("--output", type=str, help="Output report path")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "watch":
        return _cmd_watch(args)
    elif args.command == "report":
        return _cmd_report(args)
    elif args.command == "autoresearch":
        return _cmd_autoresearch(args)
    elif args.command == "check":
        return _cmd_check(args)

    return 0


def _cmd_watch(args) -> int:
    """Watch agent output from stdin, file, or git directory."""
    config = {
        "verbosity": args.verbosity,
        "use_llm_extraction": not args.no_llm,
    }
    guard = LoopGuard(config=config)
    step_id = 0

    if args.git_dir:
        return _watch_git(guard, args.git_dir, args.poll, args.output)

    if args.file:
        return _watch_file(guard, args.file, args.follow, args.delimiter, args.output)

    # Read from stdin
    print("[loop-guard] Watching stdin... (Ctrl+C to stop)", file=sys.stderr)
    buffer = ""
    delimiter = args.delimiter.replace("\\n", "\n").replace("\\t", "\t")

    try:
        for line in sys.stdin:
            buffer += line
            while delimiter in buffer:
                step_text, buffer = buffer.split(delimiter, 1)
                step_text = step_text.strip()
                if step_text:
                    guard.step(output=step_text, step_id=step_id)
                    step_id += 1
        # Process remaining buffer
        if buffer.strip():
            guard.step(output=buffer.strip(), step_id=step_id)
    except KeyboardInterrupt:
        print("\n[loop-guard] Stopped.", file=sys.stderr)

    _finalize(guard, args.output)
    return 0


def _watch_file(guard: LoopGuard, filepath: str, follow: bool, delimiter: str, output: str | None) -> int:
    """Watch a log file, optionally following new content."""
    path = Path(filepath)
    if not path.exists():
        print(f"[loop-guard] Error: File not found: {filepath}", file=sys.stderr)
        return 1

    delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
    step_id = 0

    if not follow:
        content = path.read_text()
        steps = content.split(delimiter)
        for step_text in steps:
            step_text = step_text.strip()
            if step_text:
                guard.step(output=step_text, step_id=step_id)
                step_id += 1
        _finalize(guard, output)
        return 0

    # Follow mode
    print(f"[loop-guard] Following {filepath}... (Ctrl+C to stop)", file=sys.stderr)
    try:
        with open(path) as f:
            buffer = ""
            while True:
                line = f.readline()
                if line:
                    buffer += line
                    while delimiter in buffer:
                        step_text, buffer = buffer.split(delimiter, 1)
                        step_text = step_text.strip()
                        if step_text:
                            guard.step(output=step_text, step_id=step_id)
                            step_id += 1
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[loop-guard] Stopped.", file=sys.stderr)

    _finalize(guard, output)
    return 0


def _watch_git(guard: LoopGuard, git_dir: str, poll_interval: int, output: str | None) -> int:
    """Watch git commits in a directory."""
    import subprocess

    path = Path(git_dir)
    if not path.exists():
        print(f"[loop-guard] Error: Directory not found: {git_dir}", file=sys.stderr)
        return 1

    print(f"[loop-guard] Watching git commits in {git_dir} (poll every {poll_interval}s)...", file=sys.stderr)

    last_commit = _get_latest_commit(path)
    step_id = 0

    try:
        while True:
            time.sleep(poll_interval)
            current_commit = _get_latest_commit(path)
            if current_commit and current_commit != last_commit:
                # Get commit diff
                diff_output = _get_commit_diff(path, last_commit, current_commit)
                msg = _get_commit_message(path, current_commit)

                files_modified = _get_changed_files(path, last_commit, current_commit)

                guard.step(
                    output=f"Commit: {msg}\n\n{diff_output}",
                    step_id=step_id,
                    files=files_modified,
                )
                step_id += 1
                last_commit = current_commit
    except KeyboardInterrupt:
        print("\n[loop-guard] Stopped.", file=sys.stderr)

    _finalize(guard, output)
    return 0


def _get_latest_commit(git_dir: Path) -> str | None:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", str(git_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _get_commit_diff(git_dir: Path, old: str | None, new: str) -> str:
    import subprocess
    try:
        if old:
            cmd = ["git", "-C", str(git_dir), "diff", old, new]
        else:
            cmd = ["git", "-C", str(git_dir), "show", new, "--stat"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout[:5000]  # cap at 5KB
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _get_commit_message(git_dir: Path, commit: str) -> str:
    import subprocess
    try:
        result = subprocess.run(
            ["git", "-C", str(git_dir), "log", "--format=%s", "-n1", commit],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _get_changed_files(git_dir: Path, old: str | None, new: str) -> list[str]:
    import subprocess
    try:
        if old:
            cmd = ["git", "-C", str(git_dir), "diff", "--name-only", old, new]
        else:
            cmd = ["git", "-C", str(git_dir), "show", "--name-only", "--format=", new]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return [f for f in result.stdout.strip().split("\n") if f]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _cmd_report(args) -> int:
    """Generate report from existing findings JSON."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[loop-guard] Error: File not found: {args.input}", file=sys.stderr)
        return 1

    data = json.loads(input_path.read_text())

    from loop_guard.models import Claim, ClaimType, Finding, Verdict, VerificationLevel

    reporter_module = __import__("loop_guard.reporter", fromlist=["Reporter"])
    reporter = reporter_module.Reporter()

    # Reconstruct findings from JSON
    for f_data in data.get("findings", []):
        claim_data = f_data.get("claim", {})
        claim = Claim(
            claim_type=ClaimType(claim_data.get("type", "general")),
            source_step=claim_data.get("source_step", 0),
            text=claim_data.get("text", ""),
            verifiable=claim_data.get("verifiable", False),
            evidence=claim_data.get("evidence"),
        )
        finding = Finding(
            step_id=f_data.get("step_id", 0),
            claim=claim,
            verdict=Verdict(f_data.get("verdict", "skipped")),
            level=VerificationLevel(f_data.get("level", 3)),
            explanation=f_data.get("explanation", ""),
            expected=f_data.get("expected"),
            actual=f_data.get("actual"),
            timestamp=f_data.get("timestamp", time.time()),
        )
        reporter.all_findings.append(finding)

    output_path = args.output
    if args.format == "html":
        path = output_path or "loopguard_report.html"
        reporter.generate_html_report(path)
        print(f"[loop-guard] HTML report written to {path}", file=sys.stderr)
    elif args.format == "json":
        path = output_path or "loopguard_report.json"
        reporter.generate_json_report(path)
        print(f"[loop-guard] JSON report written to {path}", file=sys.stderr)
    else:
        summary = reporter.summary()
        print(json.dumps(summary, indent=2))

    return 0


def _cmd_check(args) -> int:
    """Check a single transcript file."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[loop-guard] Error: File not found: {args.input}", file=sys.stderr)
        return 1

    config = {
        "verbosity": args.verbosity,
        "use_llm_extraction": not args.no_llm,
    }
    guard = LoopGuard(config=config)

    content = input_path.read_text()
    delimiter = args.delimiter.replace("\\n", "\n").replace("\\t", "\t")
    steps = content.split(delimiter)

    for step_id, step_text in enumerate(steps):
        step_text = step_text.strip()
        if step_text:
            guard.step(output=step_text, step_id=step_id)

    if args.output:
        guard.report(format=args.format, path=args.output)
    elif args.format != "terminal":
        guard.report(format=args.format)
    else:
        summary = guard.summary
        print(f"\n[loop-guard] Summary: {json.dumps(summary, indent=2)}")

    # Return non-zero if failures found
    s = guard.summary
    if s.get("verified_failures", 0) > 0:
        return 1
    return 0


def _cmd_autoresearch(args) -> int:
    """Monitor autoresearch experiments."""
    from loop_guard.integrations.autoresearch import AutoresearchGuard

    project_dir = Path(args.dir)
    if not project_dir.exists():
        print(f"[loop-guard] Error: Directory not found: {args.dir}", file=sys.stderr)
        return 1

    guard = AutoresearchGuard(
        str(project_dir),
        plateau_window=args.plateau_window,
        plateau_threshold=args.plateau_threshold,
        crash_limit=args.crash_limit,
    )

    if args.check:
        findings = guard.check()
        print(f"\n[loop-guard:autoresearch] Summary: {json.dumps(guard.summary, indent=2)}")
        if args.output:
            guard.guard.report(format="html" if args.output.endswith(".html") else "json", path=args.output)
            print(f"[loop-guard] Report written to {args.output}", file=sys.stderr)
        fail_count = sum(1 for f in findings if f.verdict.value in ("verified_fail", "rule_violation"))
        return 1 if fail_count > 0 else 0
    else:
        guard.watch(poll_interval=args.poll)
        return 0


def _finalize(guard: LoopGuard, output: str | None) -> None:
    """Print summary and optionally write report."""
    summary = guard.summary
    print(f"\n[loop-guard] Summary: {json.dumps(summary, indent=2)}", file=sys.stderr)
    if output:
        if output.endswith(".html"):
            guard.report(format="html", path=output)
        else:
            guard.report(format="json", path=output)
        print(f"[loop-guard] Report written to {output}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
