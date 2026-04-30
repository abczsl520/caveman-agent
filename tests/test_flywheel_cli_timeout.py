"""Regression tests for meta-flywheel CLI orchestration."""
from __future__ import annotations

import asyncio
import contextlib

import pytest

from caveman.cli import flywheel


class _HangingLoop:
    async def run(self, prompt: str):
        await asyncio.Event().wait()


class _CancellationResistantLoop:
    async def run(self, prompt: str):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.Event().wait()


class _CapturingLoop:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, prompt: str):
        return "1. S/A findings — none.\n2. Minimal fix plan — none.\n3. Verification plan — targeted tests.\nEND_FLYWHEEL_AUDIT"


@contextlib.asynccontextmanager
async def _short_wait_for_timeout():
    original = asyncio.wait_for

    async def patched(awaitable, timeout=None):
        return await original(awaitable, timeout=0.01)

    asyncio.wait_for = patched
    try:
        yield
    finally:
        asyncio.wait_for = original


@pytest.mark.asyncio
async def test_run_flywheel_times_out_hanging_round(monkeypatch, tmp_path):
    def fake_create_loop(max_iterations: int, **kwargs):
        return _HangingLoop()

    monkeypatch.setattr("caveman.agent.factory.create_loop", fake_create_loop)
    monkeypatch.setattr(flywheel, "_latest_commit", lambda project: "abc123")

    result = await flywheel.run_flywheel(
        rounds=1,
        target="tools",
        project_dir=str(tmp_path),
        max_iterations=60,
        round_timeout_s=0.01,
    )

    assert result["rounds_completed"] == 1
    assert result["successful"] == 0
    assert result["results"][0]["success"] is False
    assert "timed out" in result["results"][0]["error"].lower()
    assert result["results"][0]["subsystem"] == "tools"


@pytest.mark.asyncio
async def test_run_flywheel_timeout_is_hard_when_round_ignores_cancellation(monkeypatch, tmp_path):
    def fake_create_loop(max_iterations: int, **kwargs):
        return _CancellationResistantLoop()

    monkeypatch.setattr("caveman.agent.factory.create_loop", fake_create_loop)
    monkeypatch.setattr(flywheel, "_latest_commit", lambda project: "abc123")

    async with _short_wait_for_timeout():
        result = await flywheel.run_flywheel(
            rounds=1,
            target="tools",
            project_dir=str(tmp_path),
            max_iterations=80,
            round_timeout_s=0.01,
        )

    assert result["rounds_completed"] == 1
    assert result["successful"] == 0
    assert result["results"][0]["success"] is False
    assert "timed out" in result["results"][0]["error"].lower()
    assert result["results"][0]["subsystem"] == "tools"


def test_flywheel_self_audit_prompt_is_bounded_and_diagnostic():
    prompt = flywheel._audit_prompt_for_subsystem("flywheel", project_dir="/repo", python_path="python")

    assert "caveman/cli/flywheel.py" in prompt
    assert "caveman/tools/builtin/flywheel_tool.py" in prompt
    assert "Do not modify files" in prompt
    assert "Do not run the full test suite" in prompt
    assert "Return exactly three sections" in prompt
    assert "git commit" not in prompt
    assert "pytest tests/ -x" not in prompt


def test_flywheel_self_audit_prompt_has_short_end_sentinel():
    prompt = flywheel._audit_prompt_for_subsystem("flywheel", project_dir="/repo", python_path="python")

    assert "Keep the full response under 80 lines" in prompt
    assert "END_FLYWHEEL_AUDIT" in prompt
    assert "Static pre-audit evidence" in prompt
    assert "hard_timeout_helper" in prompt
    assert "continuation_toggle" in prompt


def test_flywheel_evaluator_accepts_self_audit_end_sentinel():
    response = """1. S/A findings — none that break the flywheel.
2. Minimal fix plan — no code change.
3. Verification plan — targeted timeout test.
END_FLYWHEEL_AUDIT
"""

    result = flywheel._evaluate_round_response(response, before_commit="abc", after_commit="abc")

    assert result["success"] is True
    assert result["explicit_flywheel_audit_complete"] is True


def test_flywheel_evaluator_rejects_self_audit_with_p0_even_with_end_sentinel():
    response = """1. S/A findings — P0: missing required meta-flywheel safety controls:
- hard_timeout_helper: MISSING (caveman/cli/flywheel.py)
2. Minimal fix plan — add missing control.
3. Verification plan — targeted test.
END_FLYWHEEL_AUDIT
"""

    result = flywheel._evaluate_round_response(response, before_commit="abc", after_commit="abc")

    assert result["success"] is False
    assert result["p0"] >= 1


def test_flywheel_evaluator_rejects_mixed_no_p0_and_p0_response():
    response = """No P0 overall.
But details: P0: hard timeout is not wired.
END_FLYWHEEL_AUDIT
"""

    result = flywheel._evaluate_round_response(response, before_commit="abc", after_commit="abc")

    assert result["success"] is False
    assert result["p0"] >= 1


def test_run_flywheel_sync_exits_nonzero_when_any_round_fails(monkeypatch):
    async def fake_run_flywheel(**kwargs):
        return {"rounds_completed": 1, "successful": 0, "results": [{"round": 1, "subsystem": "tools", "success": False, "error": "boom"}]}

    monkeypatch.setattr(flywheel, "run_flywheel", fake_run_flywheel)

    with pytest.raises(SystemExit) as exc:
        flywheel.run_flywheel_sync(rounds=1, target="tools")

    assert exc.value.code == 1


def test_flywheel_evaluator_rejects_zero_passed_or_no_tests_ran():
    response = "0 passed, no tests ran due timeout"

    result = flywheel._evaluate_round_response(response, before_commit="abc", after_commit="abc")

    assert result["success"] is False
    assert result["tests_passed"] is False
    assert result["failure_detected"] is True


@pytest.mark.asyncio
async def test_tools_target_still_uses_agent_loop_with_hardening_flags(monkeypatch, tmp_path):
    seen_continuation: list[bool] = []
    seen_diagnostic: list[bool] = []

    def fake_create_loop(max_iterations: int, **kwargs):
        seen_continuation.append(kwargs.get("allow_continuation_repair", True))
        seen_diagnostic.append(kwargs.get("diagnostic_profile", False))
        return _CapturingLoop(**kwargs)

    monkeypatch.setattr("caveman.agent.factory.create_loop", fake_create_loop)
    monkeypatch.setattr(flywheel, "_latest_commit", lambda project: "abc123")

    result = await flywheel.run_flywheel(
        rounds=1,
        target="tools",
        project_dir=str(tmp_path),
        max_iterations=80,
        round_timeout_s=1,
    )

    assert seen_continuation == [True]
    assert seen_diagnostic == [False]
    assert result["successful"] == 1


@pytest.mark.asyncio
async def test_flywheel_target_uses_deterministic_self_audit_without_agent_loop(monkeypatch, tmp_path):
    def fail_create_loop(*args, **kwargs):
        raise AssertionError("flywheel self-audit should not require full AgentLoop")

    for rel, needle in {
        "caveman/cli/flywheel.py": "_run_round_with_hard_timeout END_FLYWHEEL_AUDIT",
        "caveman/cli/utility_commands.py": "--round-timeout",
        "caveman/agent/loop.py": "allow_continuation_repair",
        "caveman/agent/factory.py": "diagnostic_profile",
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(needle, encoding="utf-8")

    monkeypatch.setattr("caveman.agent.factory.create_loop", fail_create_loop)
    monkeypatch.setattr(flywheel, "_latest_commit", lambda project: "abc123")

    result = await flywheel.run_flywheel(
        rounds=1,
        target="flywheel",
        project_dir=str(tmp_path),
        max_iterations=80,
        round_timeout_s=0.01,
    )

    assert result["rounds_completed"] == 1
    assert result["successful"] == 1
    assert result["results"][0]["subsystem"] == "flywheel"
    assert result["results"][0]["success"] is True


def test_flywheel_stats_cli_escapes_subsystem_labels(monkeypatch, capsys):
    class FakeStats:
        def summary(self):
            return {
                "total_rounds": 1,
                "total_p0_found": 0,
                "total_p1_found": 0,
                "total_fixed": 0,
                "avg_duration_s": 0.0,
                "subsystems_audited": ["safe\n\x1b[31mP0"],
            }

    monkeypatch.setattr(flywheel, "FlywheelStats", FakeStats)

    flywheel.flywheel_cli(stats=True)

    out = capsys.readouterr().out
    assert "safe\\n\\x1b[31mP0" in out
    assert "safe\n\x1b[31mP0" not in out
