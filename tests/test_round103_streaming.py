"""Tests for Round 103 — streaming support."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from caveman.agent.stream import StreamEvent, StreamBuffer


# ── StreamEvent ─────────────────────────────────────────────────────

class TestStreamEvent:
    def test_creation(self):
        e = StreamEvent(type="token", data="hello")
        assert e.type == "token"
        assert e.data == "hello"
        assert e.timestamp > 0

    def test_to_dict(self):
        e = StreamEvent(type="done", data="result")
        d = e.to_dict()
        assert d["type"] == "done"
        assert d["data"] == "result"
        assert "ts" in d

    def test_error_event(self):
        e = StreamEvent(type="error", data="something broke")
        assert e.type == "error"

    def test_tool_call_event(self):
        e = StreamEvent(type="tool_call", data={"name": "bash", "args": {"cmd": "ls"}})
        assert e.data["name"] == "bash"

    def test_thinking_event(self):
        e = StreamEvent(type="thinking", data="let me think...")
        assert e.type == "thinking"


# ── StreamBuffer ────────────────────────────────────────────────────

class TestStreamBuffer:
    def test_empty(self):
        buf = StreamBuffer()
        assert buf.text == ""
        assert len(buf) == 0
        assert buf.events == []

    def test_accumulate_tokens(self):
        buf = StreamBuffer()
        buf.add(StreamEvent(type="token", data="Hello"))
        buf.add(StreamEvent(type="token", data=" world"))
        assert buf.text == "Hello world"
        assert len(buf) == 2

    def test_non_token_events_dont_add_text(self):
        buf = StreamBuffer()
        buf.add(StreamEvent(type="tool_call", data={"name": "bash"}))
        buf.add(StreamEvent(type="token", data="result"))
        assert buf.text == "result"
        assert len(buf) == 2

    def test_clear(self):
        buf = StreamBuffer()
        buf.add(StreamEvent(type="token", data="hi"))
        buf.clear()
        assert buf.text == ""
        assert len(buf) == 0

    def test_events_returns_copy(self):
        buf = StreamBuffer()
        buf.add(StreamEvent(type="token", data="a"))
        events = buf.events
        events.append(StreamEvent(type="token", data="b"))
        assert len(buf) == 1  # original unchanged


# ── run_stream ──────────────────────────────────────────────────────

class TestRunStream:
    @pytest.mark.asyncio
    async def test_run_stream_yields_events(self):
        """Mock provider.complete(), verify events are yielded."""
        from caveman.agent.loop import AgentLoop

        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        mock_provider.context_length = 200_000

        async def fake_complete(messages, system=None, tools=None, stream=True, **kw):
            yield {"type": "delta", "text": "Hello"}
            yield {"type": "delta", "text": " world"}
            yield {"type": "done", "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}

        mock_provider.complete = fake_complete
        mock_provider.safe_complete = fake_complete

        with patch("caveman.agent.loop.phase_prepare") as mock_prepare, \
             patch("caveman.agent.loop.phase_finalize", new_callable=AsyncMock, return_value="done"), \
             patch("caveman.agent.loop.record_assistant_turn"):

            mock_context = MagicMock()
            mock_context.should_compress.return_value = False
            mock_context.messages = []
            mock_prepare.return_value = (mock_context, "system", [])

            loop = AgentLoop.__new__(AgentLoop)
            loop.provider = mock_provider
            loop.max_iterations = 5
            loop.bus = MagicMock()
            loop.bus.emit = AsyncMock()
            loop.skill_manager = MagicMock()
            loop.memory_manager = MagicMock()
            loop.trajectory_recorder = MagicMock()
            loop.trajectory_recorder.record_turn = AsyncMock()
            loop._recall = MagicMock()
            loop.engine_flags = MagicMock()
            loop.tool_registry = MagicMock()
            loop.tool_registry.get_schemas.return_value = []
            loop.permission_manager = MagicMock()
            loop.permission_manager.request = AsyncMock(return_value=True)
            loop._tool_call_count = 0
            loop._bg_skill_nudge = AsyncMock()
            loop._nudge_task_ref = ""
            loop._turn_number = 0
            loop._turn_count = 0
            loop._persistent_context = None
            loop._system_prompt_cache = None
            loop.surface = "cli"
            loop.metrics = MagicMock()
            loop._shield = None
            loop._reflect = None
            loop._nudge = MagicMock()
            loop._lint = None
            loop._ripple = None
            loop._llm_fn = AsyncMock()
            loop._check_termination = AsyncMock(return_value=True)
            loop._post_task_engines = AsyncMock()
            loop._offer_matching_skill = AsyncMock()
            loop._record_turn_metrics = MagicMock()
            loop._safe_bg = MagicMock()

            events = []
            async for event in loop.run_stream("test task"):
                events.append(event)

            types = [e.type for e in events]
            assert "token" in types
            assert "done" in types
            token_data = [e.data for e in events if e.type == "token"]
            assert "Hello" in token_data

    @pytest.mark.asyncio
    async def test_run_stream_with_tool_calls(self):
        """Provider yields tool_call events, verify they propagate."""
        from caveman.agent.loop import AgentLoop

        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        mock_provider.context_length = 200_000

        async def fake_complete(messages, system=None, tools=None, stream=True, **kw):
            yield {"type": "delta", "text": "Let me check..."}
            yield {"type": "tool_call", "id": "tc1", "name": "bash", "input": {"cmd": "ls"}}
            yield {"type": "done", "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5}}

        mock_provider.complete = fake_complete
        mock_provider.safe_complete = fake_complete

        with patch("caveman.agent.loop.phase_prepare") as mock_prepare, \
             patch("caveman.agent.loop.phase_finalize", new_callable=AsyncMock, return_value="done"), \
             patch("caveman.agent.loop.record_assistant_turn"), \
             patch("caveman.agent.tools_exec.phase_tool_execution", new_callable=AsyncMock, return_value=1):

            mock_context = MagicMock()
            mock_context.should_compress.return_value = False
            mock_context.messages = []
            mock_prepare.return_value = (mock_context, "system", [])

            loop = AgentLoop.__new__(AgentLoop)
            loop.provider = mock_provider
            loop.max_iterations = 5
            loop.bus = MagicMock()
            loop.bus.emit = AsyncMock()
            loop.skill_manager = MagicMock()
            loop.memory_manager = MagicMock()
            loop.trajectory_recorder = MagicMock()
            loop.trajectory_recorder.record_turn = AsyncMock()
            loop.trajectory_recorder.to_sharegpt.return_value = []
            loop._recall = MagicMock()
            loop.engine_flags = MagicMock()
            loop.tool_registry = MagicMock()
            loop.tool_registry.get_schemas.return_value = []
            loop.permission_manager = MagicMock()
            loop.permission_manager.request = AsyncMock(return_value=True)
            loop._tool_call_count = 0
            loop._bg_skill_nudge = AsyncMock()
            loop._nudge_task_ref = ""
            loop._turn_number = 0
            loop._turn_count = 0
            loop._persistent_context = None
            loop._system_prompt_cache = None
            loop.surface = "cli"
            loop.metrics = MagicMock()
            loop._shield = None
            loop._reflect = None
            loop._nudge = MagicMock()
            loop._lint = None
            loop._ripple = None
            loop._llm_fn = AsyncMock()
            loop._check_termination = AsyncMock(return_value=True)
            loop._post_task_engines = AsyncMock()
            loop._offer_matching_skill = AsyncMock()
            loop._record_turn_metrics = MagicMock()
            loop._safe_bg = MagicMock()

            events = []
            async for event in loop.run_stream("test"):
                events.append(event)

            types = [e.type for e in events]
            assert "token" in types
            assert "tool_call" in types
            assert "done" in types


# ── Gateway streaming ──────────────────────────────────────────────

class TestGatewayStreaming:
    @pytest.mark.asyncio
    async def test_default_send_streaming(self):
        """Base gateway collects tokens and sends as one message."""
        from caveman.gateway.base import Gateway

        class FakeGateway(Gateway):
            def __init__(self):
                self.sent: list[tuple[str, str]] = []
            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, channel_id, text):
                self.sent.append((channel_id, text))
            async def on_message(self, handler): pass
            @property
            def name(self): return "fake"

        gw = FakeGateway()

        async def fake_stream():
            yield StreamEvent(type="token", data="Hello")
            yield StreamEvent(type="token", data=" world")
            yield StreamEvent(type="done", data="Hello world")

        result = await gw.send_streaming("ch1", fake_stream())
        assert result["ok"] is True
        assert len(gw.sent) == 1
        assert gw.sent[0] == ("ch1", "Hello world")

    @pytest.mark.asyncio
    async def test_send_streaming_empty(self):
        from caveman.gateway.base import Gateway

        class FakeGateway(Gateway):
            def __init__(self):
                self.sent = []
            async def start(self): pass
            async def stop(self): pass
            async def send_message(self, channel_id, text):
                self.sent.append((channel_id, text))
            async def on_message(self, handler): pass
            @property
            def name(self): return "fake"

        gw = FakeGateway()

        async def empty_stream():
            yield StreamEvent(type="done", data="")

        result = await gw.send_streaming("ch1", empty_stream())
        assert result["ok"] is True
        assert len(gw.sent) == 0  # no text, no send
