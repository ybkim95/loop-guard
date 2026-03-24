"""Tests for ToolOutputVerifier."""

from loop_guard.models import Claim, ClaimType, Verdict
from loop_guard.verifiers.tool_output import ToolOutputVerifier


def _make_tool_claim(
    tool_name: str, args: dict, claimed_output: str, step: int = 0
) -> Claim:
    return Claim(
        claim_type=ClaimType.CODE_OUTPUT,
        source_step=step,
        text=f"Called {tool_name}",
        verifiable=True,
        evidence={
            "tool_name": tool_name,
            "tool_args": args,
            "claimed_output": claimed_output,
        },
    )


class TestToolOutputVerifier:
    def test_registered_tool_pass(self):
        v = ToolOutputVerifier()
        v.register_tool("add", lambda a, b: a + b)

        claim = _make_tool_claim("add", {"a": 2, "b": 3}, "5")
        finding = v.verify(claim)
        assert finding.verdict == Verdict.VERIFIED_PASS

    def test_registered_tool_fail(self):
        v = ToolOutputVerifier()
        v.register_tool("add", lambda a, b: a + b)

        claim = _make_tool_claim("add", {"a": 2, "b": 3}, "7")
        finding = v.verify(claim)
        assert finding.verdict == Verdict.VERIFIED_FAIL

    def test_unregistered_tool_skipped(self):
        v = ToolOutputVerifier()
        claim = _make_tool_claim("unknown_tool", {}, "result")
        finding = v.verify(claim)
        assert finding.verdict == Verdict.SKIPPED

    def test_no_evidence_skipped(self):
        v = ToolOutputVerifier()
        claim = Claim(
            claim_type=ClaimType.CODE_OUTPUT,
            source_step=0,
            text="Called something",
            evidence=None,
        )
        finding = v.verify(claim)
        assert finding.verdict == Verdict.SKIPPED

    def test_tool_exception_skipped(self):
        v = ToolOutputVerifier()
        v.register_tool("broken", lambda: 1 / 0)

        claim = _make_tool_claim("broken", {}, "anything")
        finding = v.verify(claim)
        assert finding.verdict == Verdict.SKIPPED

    def test_numeric_tolerance(self):
        v = ToolOutputVerifier()
        v.register_tool("pi", lambda: 3.14159265)

        claim = _make_tool_claim("pi", {}, "3.14159")
        finding = v.verify(claim)
        assert finding.verdict == Verdict.VERIFIED_PASS

    def test_string_match(self):
        v = ToolOutputVerifier()
        v.register_tool("greet", lambda name: f"Hello, {name}!")

        claim = _make_tool_claim("greet", {"name": "World"}, "Hello, World!")
        finding = v.verify(claim)
        assert finding.verdict == Verdict.VERIFIED_PASS

    def test_caching(self):
        call_count = 0

        def counting_fn():
            nonlocal call_count
            call_count += 1
            return 42

        v = ToolOutputVerifier()
        v.register_tool("counter", counting_fn)

        claim = _make_tool_claim("counter", {}, "42")
        v.verify(claim)
        v.verify(claim)  # should hit cache
        assert call_count == 1

    def test_shell_tool(self):
        v = ToolOutputVerifier()
        claim = Claim(
            claim_type=ClaimType.CODE_OUTPUT,
            source_step=0,
            text="Ran echo test",
            verifiable=True,
            evidence={
                "tool_name": "shell",
                "tool_type": "shell",
                "tool_args": {"command": "echo hello"},
                "claimed_output": "hello",
            },
        )
        finding = v.verify(claim)
        assert finding.verdict == Verdict.VERIFIED_PASS

    def test_shell_dangerous_refused(self):
        v = ToolOutputVerifier()
        claim = Claim(
            claim_type=ClaimType.CODE_OUTPUT,
            source_step=0,
            text="Ran rm command",
            verifiable=True,
            evidence={
                "tool_name": "shell",
                "tool_type": "shell",
                "tool_args": {"command": "rm -rf /"},
                "claimed_output": "",
            },
        )
        finding = v.verify(claim)
        assert finding.verdict == Verdict.SKIPPED
        assert "dangerous" in finding.explanation.lower() or "Refusing" in finding.explanation
