"""Regression tests for the built-in flywheel tool wrapper."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from caveman.tools.builtin import flywheel_tool


class _FakeStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def readline(self):
        item = self._chunks.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeProc:
    def __init__(self, chunks, returncode=0):
        self.stdout = _FakeStdout(chunks)
        self.returncode = None
        self._final_returncode = returncode
        self.killed = False

    async def wait(self):
        self.returncode = self._final_returncode
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_flywheel_tool_ignores_idle_read_timeout(monkeypatch):
    """A quiet subprocess heartbeat must not be treated as terminal failure."""
    proc = _FakeProc([
        asyncio.TimeoutError(),
        b"12 passed in 0.4s\n",
        b"",
    ])

    async def fake_create_subprocess_exec(*args, **kwargs):
        return proc

    async def fake_wait_for(awaitable, timeout):
        try:
            return await awaitable
        except asyncio.TimeoutError:
            raise

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(flywheel_tool, "_FLYWHEEL_HEARTBEAT_SECONDS", 0.01)

    result = await flywheel_tool.flywheel_exec(target="gateway", rounds=1)

    assert result["ok"] is True
    assert "completed" not in result["message"].lower()
    assert "12 passed" in result["output"]


@pytest.mark.asyncio
async def test_flywheel_tool_reports_failed_rounds_even_if_process_exits_zero(monkeypatch):
    proc = _FakeProc([
        b"==================================================\n",
        b"Flywheel: 0/1 rounds successful\n",
        "  ❌ Round 1: tools\n".encode("utf-8"),
        b"     Error: boom\n",
        b"",
    ], returncode=0)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return proc

    async def fake_wait_for(awaitable, timeout):
        return await awaitable

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    result = await flywheel_tool.flywheel_exec(target="tools", rounds=1)

    assert result["ok"] is False
    assert "0/1" in result["output"]


@pytest.mark.asyncio
async def test_flywheel_tool_kills_subprocess_on_overall_timeout(monkeypatch):
    """Only the overall subprocess deadline should abort the flywheel process."""
    proc = _FakeProc([b"still working\n", b""])

    async def fake_create_subprocess_exec(*args, **kwargs):
        return proc

    calls = {"wait": 0}

    async def fake_wait_for(awaitable, timeout):
        # First readline succeeds; final proc.wait() times out. Close the
        # coroutine to mirror asyncio.wait_for cancellation and avoid warnings.
        if getattr(awaitable, "__name__", "") == "wait":
            calls["wait"] += 1
            if calls["wait"] == 1:
                awaitable.close()
                raise asyncio.TimeoutError()
        return await awaitable

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    result = await flywheel_tool.flywheel_exec(target="gateway", rounds=1)

    assert result["ok"] is False
    assert proc.killed is True
    assert "exceeded" in result["error"]
