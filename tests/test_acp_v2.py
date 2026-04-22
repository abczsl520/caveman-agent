"""Tests for ACP v2 — session manager, events, server."""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from caveman.acp.session import ACPSessionManager, ACPSessionState
from caveman.acp.events import (
    ACPEventEmitter, ACPEvent, get_tool_kind, build_tool_title, make_tool_call_id,
)
from caveman.acp.server import ACPServer, SLASH_COMMANDS


# ── Event Tests ──

class TestACPEvents:
    def test_tool_kind_mapping(self):
        assert get_tool_kind("bash") == "execute"
        assert get_tool_kind("file_read") == "read"
        assert get_tool_kind("file_write") == "edit"
        assert get_tool_kind("web_search") == "fetch"
        assert get_tool_kind("unknown_tool") == "other"

    def test_tool_title(self):
        assert "bash:" in build_tool_title("bash", {"command": "ls -la"})
        assert "file_read:" in build_tool_title("file_read", {"path": "/tmp/x"})
        assert "search:" in build_tool_title("web_search", {"query": "test"})

    def test_tool_title_truncation(self):
        long_cmd = "x" * 100
        title = build_tool_title("bash", {"command": long_cmd})
        assert len(title) < 90

    def test_tool_call_id_unique(self):
        ids = {make_tool_call_id() for _ in range(100)}
        assert len(ids) == 100

    def test_event_to_dict(self):
        evt = ACPEvent("message", {"text": "hello"})
        d = evt.to_dict()
        assert d["type"] == "message"
        assert d["text"] == "hello"

    def test_event_to_sse(self):
        evt = ACPEvent("status", {"status": "running"})
        sse = evt.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        parsed = json.loads(sse[6:].strip())
        assert parsed["type"] == "status"

    @pytest.mark.asyncio
    async def test_emitter_tool_lifecycle(self):
        emitter = ACPEventEmitter("test-session")
        tc_id = await emitter.on_tool_start("bash", {"command": "echo hi"})
        assert tc_id.startswith("tc-")
        await emitter.on_tool_complete("bash", "hi")
        assert len(emitter.events) == 2
        assert emitter.events[0].event_type == "tool_call_start"
        assert emitter.events[1].event_type == "tool_call_complete"

    @pytest.mark.asyncio
    async def test_emitter_message_events(self):
        emitter = ACPEventEmitter("test-session")
        await emitter.on_thinking("hmm...")
        await emitter.on_message("Hello!")
        await emitter.on_message_delta(" world")
        assert len(emitter.events) == 3

    @pytest.mark.asyncio
    async def test_emitter_with_send_fn(self):
        received = []
        async def capture(evt):
            received.append(evt)
        emitter = ACPEventEmitter("test", send_fn=capture)
        await emitter.on_status("running")
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_emitter_empty_text_ignored(self):
        emitter = ACPEventEmitter("test")
        await emitter.on_thinking("")
        await emitter.on_message("")
        await emitter.on_message_delta("")
        assert len(emitter.events) == 0

    @pytest.mark.asyncio
    async def test_emitter_tool_complete_truncation(self):
        emitter = ACPEventEmitter("test")
        await emitter.on_tool_start("bash", {})
        long_result = "x" * 10000
        await emitter.on_tool_complete("bash", long_result)
        evt = emitter.events[1]
        assert len(evt.data["result"]) < 6000


# ── Session Manager Tests ──

class TestACPSessionManager:
    @pytest.mark.asyncio
    async def test_create_session(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock())
        state = await mgr.create_session(cwd="/tmp", model="test-model")
        assert state.session_id
        assert state.cwd == "/tmp"
        assert state.model == "test-model"

    @pytest.mark.asyncio
    async def test_get_session(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock())
        state = await mgr.create_session()
        found = await mgr.get_session(state.session_id)
        assert found is state

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock())
        assert await mgr.get_session("nonexistent") is None

    @pytest.mark.asyncio
    async def test_remove_session(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock())
        state = await mgr.create_session()
        assert await mgr.remove_session(state.session_id)
        assert await mgr.get_session(state.session_id) is None

    @pytest.mark.asyncio
    async def test_fork_session(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock())
        original = await mgr.create_session(model="opus")
        original.history = [{"role": "user", "text": "hello"}]
        forked = await mgr.fork_session(original.session_id)
        assert forked is not None
        assert forked.session_id != original.session_id
        assert forked.history == original.history
        assert forked.history is not original.history  # deep copy

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock())
        await mgr.create_session(model="a")
        await mgr.create_session(model="b")
        sessions = await mgr.list_sessions()
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_cancel_session(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock())
        state = await mgr.create_session()
        assert not state.cancel_event.is_set()
        assert await mgr.cancel_session(state.session_id)
        assert state.cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_eviction(self):
        mgr = ACPSessionManager(agent_factory=lambda **kw: MagicMock(), max_sessions=3)
        s1 = await mgr.create_session()
        await mgr.create_session()
        await mgr.create_session()
        await mgr.create_session()  # Should evict s1
        assert await mgr.get_session(s1.session_id) is None


# ── Server Tests ──

class TestACPServer:
    @pytest.mark.asyncio
    async def test_create_and_get_task(self):
        server = ACPServer(agent_fn=AsyncMock(return_value="result"))
        msg = {"role": "user", "parts": [{"type": "text", "text": "hello"}]}
        result = await server.handle_create_task(msg)
        assert result["status"] == "completed"
        assert "result" in result["result"]["parts"][0]["text"]

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        async def slow_fn(text):
            await asyncio.sleep(10)
            return "done"
        server = ACPServer(agent_fn=slow_fn)
        msg = {"role": "user", "parts": [{"type": "text", "text": "slow"}]}
        # Create task without waiting
        from caveman.acp.server import ACPTask
        from caveman.acp.events import ACPEventEmitter
        import uuid
        task = ACPTask(id=f"task-{uuid.uuid4().hex[:12]}", message=msg, emitter=ACPEventEmitter(""))
        server._tasks[task.id] = task
        # Cancel it
        result = await server.handle_cancel_task(task.id)
        assert result["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_slash_help(self):
        server = ACPServer()
        result = await server._handle_slash("/help", "")
        assert result is not None
        assert "/model" in result

    @pytest.mark.asyncio
    async def test_slash_version(self):
        server = ACPServer()
        result = await server._handle_slash("/version", "")
        assert "Caveman" in result

    @pytest.mark.asyncio
    async def test_slash_unknown(self):
        server = ACPServer()
        result = await server._handle_slash("/nonexistent", "")
        assert result is None  # Not a slash command

    @pytest.mark.asyncio
    async def test_slash_model_get(self):
        server = ACPServer()
        state = await server._session_mgr.create_session(model="opus")
        result = await server._handle_slash("/model", state.session_id)
        assert "opus" in result

    @pytest.mark.asyncio
    async def test_slash_model_set(self):
        server = ACPServer()
        state = await server._session_mgr.create_session()
        result = await server._handle_slash("/model sonnet", state.session_id)
        assert "sonnet" in result
        assert state.model == "sonnet"

    @pytest.mark.asyncio
    async def test_slash_reset(self):
        server = ACPServer()
        state = await server._session_mgr.create_session()
        state.history = [{"role": "user", "text": "hi"}]
        result = await server._handle_slash("/reset", state.session_id)
        assert "reset" in result.lower() or "cleared" in result.lower()
        assert len(state.history) == 0

    @pytest.mark.asyncio
    async def test_extract_text(self):
        msg = {"parts": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}
        assert ACPServer._extract_text(msg) == "hello world"

    @pytest.mark.asyncio
    async def test_extract_text_fallback(self):
        msg = {"text": "fallback"}
        assert ACPServer._extract_text(msg) == "fallback"
