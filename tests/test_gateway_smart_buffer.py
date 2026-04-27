"""Tests for gateway SmartBuffer."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from caveman.gateway.smart_buffer import _SmartBuffer


@pytest.fixture
def mock_router():
    router = MagicMock()
    router.send = AsyncMock()
    return router


@pytest.fixture
def buf(mock_router):
    return _SmartBuffer(mock_router, gw="test", ch="ch1", interim_enabled=True)


class TestSmartBuffer:
    @pytest.mark.asyncio
    async def test_add_accumulates_text(self, buf):
        await buf.add("hello ")
        await buf.add("world")
        assert buf._full_text == "hello world"

    @pytest.mark.asyncio
    async def test_flush_sends_text(self, buf, mock_router):
        await buf.add("hello world")
        await buf.flush()
        mock_router.send.assert_called_once_with("test", "ch1", "hello world")

    @pytest.mark.asyncio
    async def test_flush_empty_noop(self, buf, mock_router):
        await buf.flush()
        mock_router.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_interim_sends(self, buf, mock_router):
        await buf.add("thinking about it")
        result = await buf.flush_interim()
        assert result == "thinking about it"
        mock_router.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_interim_filters_monologue(self, buf, mock_router):
        await buf.add("Let me check that for you")
        result = await buf.flush_interim()
        assert result == "Let me check that for you"
        mock_router.send.assert_not_called()  # Filtered as monologue

    @pytest.mark.asyncio
    async def test_dedup_prevents_double_send(self, buf, mock_router):
        await buf.add("same text")
        await buf.flush_interim()
        assert mock_router.send.call_count == 1

        # Add same text again
        await buf.add("same text")
        await buf.flush()
        # Should NOT send again (dedup)
        assert mock_router.send.call_count == 1

    @pytest.mark.asyncio
    async def test_strip_think_blocks(self, buf):
        text = "<think>internal reasoning</think>\nActual reply"
        result = buf._strip_think_blocks(text)
        assert "internal reasoning" not in result
        assert "Actual reply" in result

    @pytest.mark.asyncio
    async def test_auto_flush_on_char_limit(self, buf, mock_router):
        buf.CHAR_LIMIT = 50
        await buf.add("x" * 60)
        # Should have auto-flushed
        mock_router.send.assert_called_once()

    def test_cancel_clears_timer(self, buf):
        mock_timer = MagicMock()
        buf._timer = mock_timer
        buf.cancel()
        mock_timer.cancel.assert_called_once()
        assert buf._timer is None

    @pytest.mark.asyncio
    async def test_sent_any_property(self, buf, mock_router):
        assert not buf.sent_any
        await buf.add("text")
        await buf.flush()
        assert buf.sent_any

    @pytest.mark.asyncio
    async def test_flush_can_suppress_user_send_for_auto_continue(self, buf, mock_router):
        # Explicit result flush suppression still works even for a normally
        # send-enabled buffer.
        await buf.add("this looks like a final report")
        await buf.flush(send=False)
        mock_router.send.assert_not_called()
        assert not buf.sent_any
        assert buf._buf == ""

    @pytest.mark.asyncio
    async def test_auto_continue_send_disabled_blocks_char_limit_flush(self, mock_router):
        buf = _SmartBuffer(mock_router, gw="test", ch="ch1", send_enabled=False)
        buf.CHAR_LIMIT = 10
        await buf.add("x" * 12)
        mock_router.send.assert_not_called()
        assert not buf.sent_any
        assert buf._buf == ""

    @pytest.mark.asyncio
    async def test_auto_continue_send_disabled_blocks_interim_flush(self, mock_router):
        buf = _SmartBuffer(mock_router, gw="test", ch="ch1", send_enabled=False)
        await buf.add("interim evidence-looking text")
        result = await buf.flush_interim()
        assert result == "interim evidence-looking text"
        mock_router.send.assert_not_called()
        assert not buf.sent_any

    def test_is_pure_monologue(self, buf):
        assert buf._is_pure_monologue("Let me check that")
        assert buf._is_pure_monologue("OK, I'll look into it")
        assert not buf._is_pure_monologue("Here are the results:\n1. Item one\n2. Item two")
        assert not buf._is_pure_monologue("x" * 200)  # Too long
        assert buf._is_pure_monologue("")
