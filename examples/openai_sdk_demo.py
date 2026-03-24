"""Demo: Using LoopGuard with the OpenAI Agents SDK.

This shows the integration pattern. LoopGuard never imports the OpenAI SDK;
it only needs the text output from each step.

To run this demo with a real agent, install openai and set OPENAI_API_KEY.
"""

from loop_guard import LoopGuard


def run_with_mock_agent():
    """Simulates an OpenAI Agents SDK coding task with LoopGuard verification."""

    guard = LoopGuard(config={
        "use_llm_extraction": False,
        "verbosity": "all",
    })

    # Simulated agent steps (what a coding agent might produce)
    steps = [
        {
            "output": "I'll implement a user authentication system. Let me start by creating the User model.",
            "code": None,
            "files": ["models/user.py"],
        },
        {
            "output": (
                "Created user.py with User class. All tests passed.\n"
                "Modified models/user.py to add password hashing."
            ),
            "code": "python -m pytest tests/test_user.py",
            "files": ["models/user.py"],
        },
        {
            "output": (
                "Added JWT authentication middleware.\n"
                "Based on RFC 7519, JWTs should include exp, iat, and sub claims.\n"
                "All tests passed."
            ),
            "code": "python -m pytest tests/ -v",
            "files": ["middleware/auth.py", "models/user.py"],
        },
        {
            "output": (
                "Added rate limiting to login endpoint.\n"
                "5 tests passed, 0 failed.\n"
                "Modified middleware/auth.py"
            ),
            "files": ["middleware/auth.py"],
        },
    ]

    print("=" * 60)
    print("LoopGuard Demo: OpenAI Agents SDK Integration")
    print("=" * 60)

    for i, step in enumerate(steps):
        findings = guard.step(
            output=step["output"],
            step_id=i,
            code=step.get("code"),
            files=step.get("files"),
        )

    print("\nSummary:", guard.summary)


def run_with_real_agent():
    """
    Example of real integration with OpenAI Agents SDK.

    This is the actual integration pattern — just 2 lines added to any loop:

        from loop_guard import LoopGuard
        guard = LoopGuard()

        # Your existing agent loop
        for step in agent.run(task):
            findings = guard.step(output=step.text, code=step.code)
    """
    try:
        # This would be your real agent code:
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.responses.create(...)
        print("Install openai and set OPENAI_API_KEY to run with a real agent.")
        print("Running mock demo instead.\n")
        run_with_mock_agent()
    except ImportError:
        print("OpenAI SDK not installed. Running mock demo.\n")
        run_with_mock_agent()


if __name__ == "__main__":
    run_with_real_agent()
