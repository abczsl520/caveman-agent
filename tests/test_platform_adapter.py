"""Tests for BasePlatformAdapter and platform_types."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, Optional

from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    ProcessingOutcome, SendResult, SessionSource, build_session_key,
)
from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_delivery import (
    extract_media, extract_images, is_animation_url, truncate_message,
)


# ── Test fixtures ───────────────────────────────────────────────────────────

class MockAdapter(BasePlatformAdapter):
    """Concrete adapter for testing."""

    def __init__(self, **kwargs):
        config = PlatformConfig(enabled=True)
        super().__init__(config, Platform.DISCORD)
        self.sent_messages: list = []
        self.outcomes: list[ProcessingOutcome] = []
        self.connected = False
        self._fail_send = False

    async def connect(self) -> bool:
        self.connected = True
        self._running = True
        return True

    async def disconnect(self) -> None:
        self.connected = False
        self._running = False

    async def send(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if self._fail_send:
            return SendResult(success=False, error="network error", retryable=True)
        msg_id = f"msg_{len(self.sent_messages)}"
        self.sent_messages.append({
            "chat_id": chat_id, "content": content,
            "reply_to": reply_to, "metadata": metadata,
        })
        return SendResult(success=True, message_id=msg_id)

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome,
    ) -> None:
        self.outcomes.append(outcome)


def _make_event(text="hello", chat_id="123", user_id="u1", platform=Platform.DISCORD) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(platform=platform, chat_id=chat_id, user_id=user_id, chat_type="dm"),
        message_id="evt_1",
    )


# ── Platform Types Tests ────────────────────────────────────────────────────

class TestSessionSource:
    def test_description_dm(self):
        s = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_name="Alice", chat_type="dm")
        assert "DM with Alice" in s.description

    def test_description_group(self):
        s = SessionSource(platform=Platform.DISCORD, chat_id="456", chat_name="general", chat_type="group")
        assert "group: general" in s.description

    def test_description_thread(self):
        s = SessionSource(platform=Platform.SLACK, chat_id="789", chat_name="dev", chat_type="channel", thread_id="t1")
        assert "thread: t1" in s.description

    def test_roundtrip(self):
        s = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_name="Bob", chat_type="group", thread_id="t1")
        d = s.to_dict()
        s2 = SessionSource.from_dict(d)
        assert s2.platform == s.platform
        assert s2.chat_id == s.chat_id
        assert s2.thread_id == s.thread_id


class TestBuildSessionKey:
    def test_dm(self):
        s = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
        key = build_session_key(s)
        assert key == "telegram:123"

    def test_group_per_user(self):
        s = SessionSource(platform=Platform.DISCORD, chat_id="ch1", chat_type="group", user_id="u1")
        key = build_session_key(s, group_sessions_per_user=True)
        assert key == "discord:ch1:u:u1"

    def test_group_shared(self):
        s = SessionSource(platform=Platform.DISCORD, chat_id="ch1", chat_type="group", user_id="u1")
        key = build_session_key(s, group_sessions_per_user=False)
        assert key == "discord:ch1"

    def test_thread(self):
        s = SessionSource(platform=Platform.SLACK, chat_id="ch1", chat_type="channel", thread_id="t1")
        key = build_session_key(s)
        assert key == "slack:ch1:t:t1"


class TestMessageEvent:
    def test_is_command(self):
        e = MessageEvent(text="/reset")
        assert e.is_command()
        assert e.get_command() == "reset"

    def test_command_with_args(self):
        e = MessageEvent(text="/model gpt-4")
        assert e.get_command() == "model"
        assert e.get_command_args() == "gpt-4"

    def test_not_command(self):
        e = MessageEvent(text="hello world")
        assert not e.is_command()
        assert e.get_command() is None

    def test_command_with_bot_mention(self):
        e = MessageEvent(text="/reset@mybot")
        assert e.get_command() == "reset"


class TestSendResult:
    def test_success(self):
        r = SendResult(success=True, message_id="123")
        assert r.success
        assert r.message_id == "123"

    def test_failure(self):
        r = SendResult(success=False, error="rate limited", retryable=True)
        assert not r.success
        assert r.retryable


# ── BasePlatformAdapter Tests ───────────────────────────────────────────────

class TestAdapterBasics:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        adapter = MockAdapter()
        assert not adapter.is_connected
        await adapter.connect()
        assert adapter.is_connected
        await adapter.disconnect()
        assert not adapter.is_connected

    def test_name(self):
        adapter = MockAdapter()
        assert adapter.name == "Discord"

    def test_build_source(self):
        adapter = MockAdapter()
        source = adapter.build_source("ch1", chat_name="general", chat_type="group", user_id="u1")
        assert source.platform == Platform.DISCORD
        assert source.chat_id == "ch1"
        assert source.chat_type == "group"


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_basic_message_flow(self):
        adapter = MockAdapter()
        adapter.set_message_handler(AsyncMock(return_value="Hello back!"))

        event = _make_event("hi")
        await adapter.handle_message(event)
        # Give background task time to complete
        await asyncio.sleep(0.05)

        assert len(adapter.sent_messages) == 1
        assert adapter.sent_messages[0]["content"] == "Hello back!"

    @pytest.mark.asyncio
    async def test_no_handler(self):
        adapter = MockAdapter()
        event = _make_event("hi")
        await adapter.handle_message(event)
        await asyncio.sleep(0.05)
        assert len(adapter.sent_messages) == 0

    @pytest.mark.asyncio
    async def test_empty_response(self):
        adapter = MockAdapter()
        adapter.set_message_handler(AsyncMock(return_value=None))

        event = _make_event("hi")
        await adapter.handle_message(event)
        await asyncio.sleep(0.05)
        assert len(adapter.sent_messages) == 0
        assert adapter.outcomes == [ProcessingOutcome.NO_RESPONSE]

    @pytest.mark.asyncio
    async def test_non_empty_response_marks_success(self):
        adapter = MockAdapter()
        adapter.set_message_handler(AsyncMock(return_value="visible result"))

        event = _make_event("hi")
        await adapter.handle_message(event)
        await asyncio.sleep(0.05)
        assert len(adapter.sent_messages) == 1
        assert adapter.outcomes == [ProcessingOutcome.SUCCESS]

    @pytest.mark.asyncio
    async def test_command_bypass_during_active_session(self):
        adapter = MockAdapter()
        handler = AsyncMock(side_effect=[
            asyncio.sleep(1),  # First message takes long
            "stopped",         # /stop response
        ])
        adapter.set_message_handler(handler)

        # Start a long-running message
        event1 = _make_event("do something long")
        await adapter.handle_message(event1)
        await asyncio.sleep(0.01)

        # Send /stop while first is active
        event2 = _make_event("/stop", chat_id="123", user_id="u1")
        await adapter.handle_message(event2)
        await asyncio.sleep(0.05)

        # /stop should have been processed immediately
        assert any("stopped" in m["content"] for m in adapter.sent_messages)

    @pytest.mark.asyncio
    async def test_interrupt_queues_message(self):
        adapter = MockAdapter()
        gate = asyncio.Event()
        calls = []

        async def handler(event):
            calls.append(event.text)
            if event.text == "first":
                await gate.wait()
                return "first response"
            return "second response"

        adapter.set_message_handler(handler)

        event1 = _make_event("first")
        await adapter.handle_message(event1)
        await asyncio.sleep(0.01)

        # Second message while first is processing must be handed to the
        # canonical GatewayServer handler immediately.  GatewayServer owns the
        # real interrupt/session-lock semantics; adapter-local pending queues
        # hide this message and make interrupts silently fail.
        event2 = _make_event("second", chat_id="123", user_id="u1")
        await adapter.handle_message(event2)
        await asyncio.sleep(0.05)

        assert calls == ["first", "second"]
        assert len(adapter._pending_messages) == 0

        gate.set()
        await asyncio.sleep(0.05)
        assert any("first response" in m["content"] for m in adapter.sent_messages)
        assert any("second response" in m["content"] for m in adapter.sent_messages)


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_on_network_error(self):
        adapter = MockAdapter()
        call_count = 0
        original_send = adapter.send

        async def flaky_send(chat_id, content, reply_to=None, metadata=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SendResult(success=False, error="connectionreset", retryable=True)
            return await original_send(chat_id, content, reply_to, metadata)

        adapter.send = flaky_send
        result = await adapter._send_with_retry("ch1", "hello", base_delay=0.01)
        assert result.success
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_plain_text_fallback_on_format_error(self):
        adapter = MockAdapter()
        call_count = 0

        async def format_fail_send(chat_id, content, reply_to=None, metadata=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SendResult(success=False, error="Bad Request: can't parse entities")
            return SendResult(success=True, message_id="fallback")

        adapter.send = format_fail_send
        result = await adapter._send_with_retry("ch1", "**bold**", base_delay=0.01)
        assert result.success
        assert call_count == 2


class TestMediaExtraction:
    def test_extract_media_tag(self):
        content = "Here's the audio:\nMEDIA:/tmp/voice.ogg\nDone."
        media, cleaned = extract_media(content)
        assert len(media) == 1
        assert media[0][0] == "/tmp/voice.ogg"
        assert "MEDIA:" not in cleaned

    def test_extract_media_voice_directive(self):
        content = "[[audio_as_voice]]\nMEDIA:/tmp/voice.ogg"
        media, cleaned = extract_media(content)
        assert media[0][1] is True  # is_voice
        assert "[[audio_as_voice]]" not in cleaned

    def test_extract_images(self):
        content = "Look at this:\n![cat](https://example.com/cat.jpg)\nNice!"
        images, cleaned = extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "https://example.com/cat.jpg"
        assert images[0][1] == "cat"
        assert "![cat]" not in cleaned

    def test_no_extract_non_image_url(self):
        content = "![link](https://example.com/page)"
        images, cleaned = extract_images(content)
        assert len(images) == 0
        assert cleaned == content


class TestMessageSplitting:
    def test_short_message_no_split(self):
        chunks = truncate_message("hello", 100)
        assert chunks == ["hello"]

    def test_long_message_splits(self):
        content = "word " * 1000  # ~5000 chars
        chunks = truncate_message(content, 200)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)
        assert "(1/" in chunks[0]

    def test_code_block_preservation(self):
        content = "Before\n```python\n" + "x = 1\n" * 100 + "```\nAfter"
        chunks = truncate_message(content, 300)
        assert len(chunks) > 1


class TestCancelBackgroundTasks:
    @pytest.mark.asyncio
    async def test_cancel_clears_state(self):
        adapter = MockAdapter()
        adapter._active_sessions["key1"] = asyncio.Event()
        adapter._pending_messages["key1"] = _make_event("pending")

        await adapter.cancel_background_tasks()
        assert len(adapter._active_sessions) == 0
        assert len(adapter._pending_messages) == 0
        assert len(adapter._background_tasks) == 0
