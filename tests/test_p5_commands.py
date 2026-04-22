"""Tests for P5: session commands, allowlist, agent runner, ACP lifecycle."""
from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock


# ── Session Commands Tests ──

class TestSessionCommands:
    def test_parse_duration(self):
        from caveman.gateway.session_commands import parse_duration_ms
        assert parse_duration_ms("30m") == 1800000
        assert parse_duration_ms("2h") == 7200000
        assert parse_duration_ms("1d") == 86400000
        assert parse_duration_ms("60s") == 60000
        assert parse_duration_ms("invalid") == 0

    def test_session_binding_expired(self):
        from caveman.gateway.session_commands import SessionBinding
        binding = SessionBinding(
            session_key="s1",
            idle_timeout_ms=1000,
            last_activity=time.time() - 10,
        )
        assert binding.is_expired

    def test_session_binding_not_expired(self):
        from caveman.gateway.session_commands import SessionBinding
        binding = SessionBinding(
            session_key="s1",
            idle_timeout_ms=60000,
            last_activity=time.time(),
        )
        assert not binding.is_expired

    @pytest.mark.asyncio
    async def test_handle_reset(self):
        from caveman.gateway.session_commands import SessionCommandHandler
        reset_fn = AsyncMock()
        handler = SessionCommandHandler(reset_fn=reset_fn)
        result = await handler.handle_reset("s1")
        assert "reset" in result.lower()
        reset_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_idle(self):
        from caveman.gateway.session_commands import SessionCommandHandler
        handler = SessionCommandHandler()
        result = await handler.handle_session_idle("s1", "30m")
        assert "30m" in result
        binding = handler.get_binding("s1")
        assert binding is not None
        assert binding.idle_timeout_ms == 1800000

    @pytest.mark.asyncio
    async def test_handle_max_age(self):
        from caveman.gateway.session_commands import SessionCommandHandler
        handler = SessionCommandHandler()
        result = await handler.handle_session_max_age("s1", "2h")
        assert "2.0h" in result

    def test_reap_expired(self):
        from caveman.gateway.session_commands import SessionCommandHandler, SessionBinding
        handler = SessionCommandHandler()
        handler._bindings["s1"] = SessionBinding(
            session_key="s1", idle_timeout_ms=1, last_activity=time.time() - 10,
        )
        handler._bindings["s2"] = SessionBinding(
            session_key="s2", idle_timeout_ms=60000, last_activity=time.time(),
        )
        reaped = handler.reap_expired()
        assert "s1" in reaped
        assert "s2" not in reaped


# ── Allowlist Tests ──

class TestAllowlist:
    def test_add_and_check(self):
        from caveman.gateway.allowlist_commands import AllowlistManager
        mgr = AllowlistManager()
        mgr.add("user:123")
        assert mgr.is_allowed("user:123")
        assert not mgr.is_allowed("user:456")

    def test_empty_allows_all(self):
        from caveman.gateway.allowlist_commands import AllowlistManager
        mgr = AllowlistManager()
        assert mgr.is_allowed("anything")

    def test_wildcard(self):
        from caveman.gateway.allowlist_commands import AllowlistManager
        mgr = AllowlistManager()
        mgr.add("*")
        assert mgr.is_allowed("user:123")
        assert mgr.is_allowed("channel:456")

    def test_remove(self):
        from caveman.gateway.allowlist_commands import AllowlistManager
        mgr = AllowlistManager()
        mgr.add("user:123")
        mgr.remove("user:123")
        assert mgr.is_allowed("anything")  # Empty = allow all

    def test_format_list(self):
        from caveman.gateway.allowlist_commands import AllowlistManager
        mgr = AllowlistManager()
        mgr.add("user:123", added_by="admin")
        text = mgr.format_list()
        assert "user:123" in text

    def test_persist(self, tmp_path):
        from caveman.gateway.allowlist_commands import AllowlistManager
        path = tmp_path / "allowlist.json"
        mgr1 = AllowlistManager(persist_path=path)
        mgr1.add("user:123")
        # Load from disk
        mgr2 = AllowlistManager(persist_path=path)
        assert mgr2.is_allowed("user:123")

    def test_glob_pattern(self):
        from caveman.gateway.allowlist_commands import AllowlistEntry
        entry = AllowlistEntry(pattern="user:*", entry_type="glob")
        assert entry.matches("user:123")
        assert entry.matches("user:456")
        assert not entry.matches("channel:123")


# ── Agent Runner Tests ──

class TestAgentRunner:
    @pytest.mark.asyncio
    async def test_basic_run(self):
        from caveman.gateway.agent_runner import AgentRunner, RunContext
        agent_fn = AsyncMock(return_value={"text": "hello", "tool_calls": 2})
        runner = AgentRunner(agent_fn=agent_fn)
        ctx = RunContext(session_key="s1", command_body="hi")
        result = await runner.run(ctx)
        assert result.ok
        assert result.text == "hello"
        assert result.tool_calls == 2

    @pytest.mark.asyncio
    async def test_run_with_reset(self):
        from caveman.gateway.agent_runner import AgentRunner, RunContext
        agent_fn = AsyncMock(return_value="done")
        session_mgr = MagicMock()
        session_mgr.reset = MagicMock()
        runner = AgentRunner(agent_fn=agent_fn, session_manager=session_mgr)
        ctx = RunContext(session_key="s1", reset_triggered=True)
        result = await runner.run(ctx)
        assert result.ok
        session_mgr.reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_error(self):
        from caveman.gateway.agent_runner import AgentRunner, RunContext
        agent_fn = AsyncMock(side_effect=RuntimeError("boom"))
        runner = AgentRunner(agent_fn=agent_fn)
        ctx = RunContext(session_key="s1")
        result = await runner.run(ctx)
        assert not result.ok
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_cancel(self):
        from caveman.gateway.agent_runner import AgentRunner
        runner = AgentRunner()
        # No active run, should return False
        result = await runner.cancel("s1")
        assert not result

    def test_active_count(self):
        from caveman.gateway.agent_runner import AgentRunner
        runner = AgentRunner()
        assert runner.active_count() == 0
        assert runner.list_active() == []


# ── ACP Lifecycle Tests ──

class TestACPLifecycle:
    @pytest.mark.asyncio
    async def test_spawn(self):
        from caveman.gateway.acp_lifecycle import ACPLifecycleManager
        spawn_fn = AsyncMock(return_value={"text": "done"})
        mgr = ACPLifecycleManager(spawn_fn=spawn_fn)
        result = await mgr.handle_spawn("claude", "write code")
        assert "completed" in result

    @pytest.mark.asyncio
    async def test_spawn_limit(self):
        from caveman.gateway.acp_lifecycle import ACPLifecycleManager
        mgr = ACPLifecycleManager(max_sessions=2)
        await mgr.handle_spawn("a1", "task1")
        await mgr.handle_spawn("a2", "task2")
        result = await mgr.handle_spawn("a3", "task3")
        assert "Too many" in result

    def test_list_empty(self):
        from caveman.gateway.acp_lifecycle import ACPLifecycleManager
        mgr = ACPLifecycleManager()
        result = mgr.handle_list()
        assert "No ACP" in result

    @pytest.mark.asyncio
    async def test_list_with_sessions(self):
        from caveman.gateway.acp_lifecycle import ACPLifecycleManager
        mgr = ACPLifecycleManager()
        await mgr.handle_spawn("claude", "task1")
        result = mgr.handle_list()
        assert "claude" in result

    @pytest.mark.asyncio
    async def test_kill(self):
        from caveman.gateway.acp_lifecycle import ACPLifecycleManager
        mgr = ACPLifecycleManager()
        await mgr.handle_spawn("claude", "task1")
        session_id = list(mgr._sessions.keys())[0]
        result = await mgr.handle_kill(session_id[:8])
        assert "Killed" in result

    @pytest.mark.asyncio
    async def test_status(self):
        from caveman.gateway.acp_lifecycle import ACPLifecycleManager
        mgr = ACPLifecycleManager()
        await mgr.handle_spawn("claude", "write tests")
        session_id = list(mgr._sessions.keys())[0]
        result = mgr.handle_status(session_id[:8])
        assert "claude" in result
        assert "write tests" in result
