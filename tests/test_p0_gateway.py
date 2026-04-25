"""Tests for P0 gateway modules — preflight, processor, execution, session, outbound."""
from __future__ import annotations

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.gateway.platform_types import MessageEvent, MessageType, ProcessingOutcome, SessionSource


# ── Preflight Tests ──

class TestPreflight:
    def _make_event(self, text="hello", user_id="u1", chat_id="c1", chat_type="dm"):
        src = SessionSource(platform="discord", chat_id=chat_id, user_id=user_id, chat_type=chat_type)
        return MessageEvent(text=text, source=src, message_id="m1")

    def test_self_message_blocked(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(bot_user_id="bot1"))
        event = self._make_event(user_id="bot1")
        result = pf.check(event)
        assert not result.passed
        assert result.drop_reason == "self_message"

    def test_blocked_user(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(blocked_users={"bad_user"}))
        event = self._make_event(user_id="bad_user")
        assert not pf.check(event).passed

    def test_user_allowlist(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(allowed_users={"good_user"}))
        assert not pf.check(self._make_event(user_id="other")).passed
        assert pf.check(self._make_event(user_id="good_user")).passed

    def test_channel_allowlist(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(allowed_channels={"ch1"}))
        assert not pf.check(self._make_event(chat_id="ch2")).passed
        assert pf.check(self._make_event(chat_id="ch1")).passed

    def test_dm_disabled(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(allow_dms=False))
        assert not pf.check(self._make_event(chat_type="dm")).passed

    def test_groups_disabled(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(allow_groups=False))
        assert not pf.check(self._make_event(chat_type="group")).passed

    def test_empty_content_blocked(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig())
        assert not pf.check(self._make_event(text="")).passed

    def test_rate_limit(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(rate_limit=2, rate_window=60, dedup_window=0))
        assert pf.check(self._make_event(text="msg1")).passed
        assert pf.check(self._make_event(text="msg2")).passed
        assert not pf.check(self._make_event(text="msg3")).passed

    def test_dedup(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(dedup_window=5.0))
        assert pf.check(self._make_event(text="same")).passed
        assert not pf.check(self._make_event(text="same")).passed
        assert pf.check(self._make_event(text="different")).passed

    def test_command_detection(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig())
        result = pf.check(self._make_event(text="/stop"))
        assert result.passed
        assert result.command == "stop"
        assert result.is_command_bypass

    def test_mention_required(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig(
            require_mention=True, bot_user_id="bot1", dedup_window=0,
        ))
        # Group without mention → blocked
        assert not pf.check(self._make_event(text="hello there", chat_type="group")).passed
        # Group with mention → passed
        assert pf.check(self._make_event(text="<@bot1> hello", chat_type="group")).passed
        # DM → no mention needed
        assert pf.check(self._make_event(text="hello dm", chat_type="dm")).passed

    def test_normal_message_passes(self):
        from caveman.gateway.preflight import MessagePreflight, PreflightConfig
        pf = MessagePreflight(PreflightConfig())
        result = pf.check(self._make_event())
        assert result.passed


# ── Processor Tests ──

class TestProcessor:
    def test_extract_media_tags(self):
        from caveman.gateway.processor import extract_media_tags
        tags = extract_media_tags("Here is MEDIA:/tmp/audio.ogg and MEDIA:/tmp/image.png")
        assert len(tags) == 2
        assert tags[0] == ("/tmp/audio.ogg", True)
        assert tags[1] == ("/tmp/image.png", False)

    def test_extract_images(self):
        from caveman.gateway.processor import extract_images
        images, text = extract_images("Look: ![cat](https://example.com/cat.jpg) nice")
        assert len(images) == 1
        assert images[0] == ("https://example.com/cat.jpg", "cat")
        assert "cat.jpg" not in text

    def test_extract_local_files(self):
        from caveman.gateway.processor import extract_local_files
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake")
            path = f.name
        try:
            files, text = extract_local_files(f"Check {path} out")
            assert path in files
        finally:
            os.unlink(path)

    def test_is_animation_url(self):
        from caveman.gateway.processor import _is_animation_url
        assert _is_animation_url("https://example.com/funny.gif")
        assert _is_animation_url("https://media.giphy.com/abc")
        assert not _is_animation_url("https://example.com/photo.jpg")

    @pytest.mark.asyncio
    async def test_empty_handler_response_is_not_success_outcome(self):
        from caveman.gateway.processor import MessageProcessor

        adapter = MagicMock()
        adapter._message_handler = AsyncMock(return_value=None)
        adapter.send_typing = AsyncMock()
        adapter.stop_typing = AsyncMock()
        adapter.send = AsyncMock()
        outcomes = []

        async def capture_hook(name, *args):
            if name == "on_processing_complete":
                outcomes.append(args[1])

        processor = MessageProcessor(adapter)
        processor._emit_hook = AsyncMock(side_effect=capture_hook)
        event = MessageEvent(
            text="hi",
            source=SessionSource(platform="discord", chat_id="c1", user_id="u1"),
            message_id="m1",
        )

        await processor._run(event, "discord:c1")

        assert outcomes == [ProcessingOutcome.NO_RESPONSE]
        adapter.send.assert_not_called()


# ── Execution Engine Tests ──

class TestExecution:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        from caveman.gateway.execution import AgentExecutionEngine
        engine = AgentExecutionEngine(
            agent_fn=AsyncMock(return_value="Hello!"),
        )
        result = await engine.execute("hi")
        assert result.success
        assert result.response == "Hello!"

    @pytest.mark.asyncio
    async def test_transient_retry(self):
        from caveman.gateway.execution import AgentExecutionEngine, ExecutionConfig
        call_count = 0
        async def flaky_fn(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("503 service unavailable")
            return "ok"
        engine = AgentExecutionEngine(
            agent_fn=flaky_fn,
            config=ExecutionConfig(retry_delay=0.01),
        )
        result = await engine.execute("test")
        assert result.success
        assert result.retries == 2

    @pytest.mark.asyncio
    async def test_non_transient_fails_fast(self):
        from caveman.gateway.execution import AgentExecutionEngine
        engine = AgentExecutionEngine(
            agent_fn=AsyncMock(side_effect=ValueError("invalid input")),
        )
        result = await engine.execute("test")
        assert not result.success

    @pytest.mark.asyncio
    async def test_context_overflow_compaction(self):
        from caveman.gateway.execution import AgentExecutionEngine, ExecutionConfig
        call_count = 0
        async def overflow_fn(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("context_length_exceeded")
            return "compacted ok"
        engine = AgentExecutionEngine(
            agent_fn=overflow_fn,
            config=ExecutionConfig(retry_delay=0.01),
            compact_fn=AsyncMock(return_value=True),
        )
        result = await engine.execute("test")
        assert result.success
        assert result.compactions == 1

    @pytest.mark.asyncio
    async def test_fallback_model(self):
        from caveman.gateway.execution import (
            AgentExecutionEngine, ExecutionConfig, FallbackCandidate,
        )
        async def fail_primary(*a, **kw):
            if kw.get("model") == "primary":
                raise Exception("model not available")
            return "fallback ok"
        engine = AgentExecutionEngine(
            agent_fn=fail_primary,
            config=ExecutionConfig(
                fallback_candidates=[FallbackCandidate("anthropic", "fallback")],
            ),
        )
        result = await engine.execute("test", model="primary")
        assert result.success
        assert result.fallback_used


# ── Session Manager Tests ──

class TestSessionManager:
    @pytest.mark.asyncio
    async def test_get_or_create(self):
        from caveman.gateway.session_manager import GatewaySessionManager
        mgr = GatewaySessionManager()
        s1 = await mgr.get_or_create("key1", model="opus")
        assert s1.session_key == "key1"
        assert s1.model == "opus"
        assert s1.turn_count == 0  # fresh session
        # Get same session — should touch it
        s2 = await mgr.get_or_create("key1")
        assert s2 is s1
        assert s2.turn_count == 1  # touched once on re-get

    @pytest.mark.asyncio
    async def test_remove(self):
        from caveman.gateway.session_manager import GatewaySessionManager
        mgr = GatewaySessionManager()
        await mgr.get_or_create("key1")
        assert await mgr.remove("key1")
        assert await mgr.get("key1") is None

    @pytest.mark.asyncio
    async def test_reset(self):
        from caveman.gateway.session_manager import GatewaySessionManager
        mgr = GatewaySessionManager()
        s = await mgr.get_or_create("key1")
        s.history = [{"role": "user", "text": "hi"}]
        s.total_tokens = 5000
        await mgr.reset("key1", "test")
        assert len(s.history) == 0
        assert s.total_tokens == 0

    @pytest.mark.asyncio
    async def test_model_override(self):
        from caveman.gateway.session_manager import GatewaySessionManager
        mgr = GatewaySessionManager()
        s = await mgr.get_or_create("key1", model="default")
        assert s.effective_model == "default"
        s.set_model_override("opus", source="user")
        assert s.effective_model == "opus"
        # Reset preserves user override
        await mgr.reset("key1")
        assert s.model_override == "opus"

    @pytest.mark.asyncio
    async def test_should_compact(self):
        from caveman.gateway.session_manager import GatewaySessionManager
        mgr = GatewaySessionManager()
        s = await mgr.get_or_create("key1")
        assert not s.should_compact(200000)
        s.total_tokens = 150000
        assert s.should_compact(200000)

    @pytest.mark.asyncio
    async def test_reap_expired(self):
        from caveman.gateway.session_manager import GatewaySessionManager
        mgr = GatewaySessionManager(ttl=0.01)
        await mgr.get_or_create("key1")
        await asyncio.sleep(0.02)
        count = await mgr.reap_expired()
        assert count == 1

    @pytest.mark.asyncio
    async def test_eviction(self):
        from caveman.gateway.session_manager import GatewaySessionManager
        mgr = GatewaySessionManager(max_sessions=2)
        await mgr.get_or_create("a")
        await mgr.get_or_create("b")
        await mgr.get_or_create("c")  # Should evict "a"
        assert await mgr.get("a") is None


# ── Outbound Tests ──

class TestOutbound:
    def test_chunk_short_message(self):
        from caveman.gateway.outbound import chunk_message
        chunks = chunk_message("hello", 100)
        assert chunks == ["hello"]

    def test_chunk_long_message(self):
        from caveman.gateway.outbound import chunk_message
        text = "word " * 500  # ~2500 chars
        chunks = chunk_message(text, 100)
        assert len(chunks) > 1
        assert all(len(c) <= 110 for c in chunks)  # Allow small overflow for fence closing

    def test_chunk_preserves_code_blocks(self):
        from caveman.gateway.outbound import chunk_message
        text = "Before\n```python\n" + "x = 1\n" * 50 + "```\nAfter"
        chunks = chunk_message(text, 200)
        # Each chunk should have balanced fences
        for chunk in chunks:
            opens = chunk.count("```")
            assert opens % 2 == 0 or chunk.endswith("```")

    def test_strip_markdown(self):
        from caveman.gateway.outbound import strip_markdown
        assert strip_markdown("**bold**") == "bold"
        assert strip_markdown("*italic*") == "italic"
        assert strip_markdown("`code`") == "code"
        assert strip_markdown("# Header") == "Header"
        assert strip_markdown("[link](http://x)") == "link"

    def test_platform_limits(self):
        from caveman.gateway.outbound import PLATFORM_LIMITS
        assert PLATFORM_LIMITS["discord"] == 2000
        assert PLATFORM_LIMITS["telegram"] == 4096

    @pytest.mark.asyncio
    async def test_send_with_retry_success(self):
        from caveman.gateway.outbound import OutboundDelivery
        from caveman.gateway.platform_types import SendResult as SR
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=SR(success=True, message_id="m1"))
        delivery = OutboundDelivery(adapter)
        result = await delivery.send_with_retry("ch1", "hello")
        assert result.success

    @pytest.mark.asyncio
    async def test_send_with_retry_retries(self):
        from caveman.gateway.outbound import OutboundDelivery
        from caveman.gateway.platform_types import SendResult as SR
        call_count = 0
        async def flaky_send(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return SR(success=False, error="429 rate limit")
            return SR(success=True, message_id="m1")
        adapter = MagicMock()
        adapter.send = flaky_send
        delivery = OutboundDelivery(adapter)
        result = await delivery.send_with_retry("ch1", "hello")
        assert result.success
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_outcome_reaction_success_clears_processing_without_success_marker(self):
        from caveman.gateway.outbound import OutboundDelivery, ReactionState
        adapter = MagicMock()
        adapter.add_reaction = AsyncMock()
        adapter.remove_reaction = AsyncMock()
        delivery = OutboundDelivery(adapter)
        state = ReactionState(channel_id="ch1", message_id="m1", current="⏳")

        await delivery.set_outcome_reaction(state, success=True)

        adapter.remove_reaction.assert_awaited_once_with("ch1", "m1", "⏳")
        adapter.add_reaction.assert_not_awaited()
        assert state.current == ""

    @pytest.mark.asyncio
    async def test_outcome_reaction_failure_adds_error_marker(self):
        from caveman.gateway.outbound import OutboundDelivery, ReactionState
        adapter = MagicMock()
        adapter.add_reaction = AsyncMock()
        adapter.remove_reaction = AsyncMock()
        delivery = OutboundDelivery(adapter)
        state = ReactionState(channel_id="ch1", message_id="m1", current="⏳")

        await delivery.set_outcome_reaction(state, success=False)

        adapter.remove_reaction.assert_awaited_once_with("ch1", "m1", "⏳")
        adapter.add_reaction.assert_awaited_once_with("ch1", "m1", "❌")
        assert state.current == "❌"

    @pytest.mark.asyncio
    async def test_legacy_done_reaction_wrapper_does_not_add_success_marker(self):
        from caveman.gateway.outbound import OutboundDelivery, ReactionState
        adapter = MagicMock()
        adapter.add_reaction = AsyncMock()
        adapter.remove_reaction = AsyncMock()
        delivery = OutboundDelivery(adapter)
        state = ReactionState(channel_id="ch1", message_id="m1", current="⏳")

        await delivery.set_done_reaction(state, success=True)

        adapter.remove_reaction.assert_awaited_once_with("ch1", "m1", "⏳")
        adapter.add_reaction.assert_not_awaited()
        assert state.current == ""
