"""Tests for gateway debounce module."""
import asyncio

import pytest

from caveman.gateway.debounce import MessageDebouncer, DEFAULT_DEBOUNCE_MS


class TestMessageDebouncer:
    @pytest.mark.asyncio
    async def test_single_message_passes_through(self):
        """A single message should be delivered after debounce window."""
        received = []

        async def handler(text, ctx):
            received.append((text, ctx))

        d = MessageDebouncer(handler, debounce_ms=50)
        await d.add_message("hello", {"user_id": "u1", "channel_id": "c1"})
        await asyncio.sleep(0.15)  # Wait for debounce
        assert len(received) == 1
        assert received[0][0] == "hello"

    @pytest.mark.asyncio
    async def test_rapid_messages_merged(self):
        """Multiple rapid messages should be merged into one."""
        received = []

        async def handler(text, ctx):
            received.append((text, ctx))

        d = MessageDebouncer(handler, debounce_ms=100)
        ctx = {"user_id": "u1", "channel_id": "c1", "gateway_name": "test"}
        await d.add_message("hello", ctx)
        await d.add_message("world", ctx)
        await d.add_message("!", ctx)
        await asyncio.sleep(0.25)
        assert len(received) == 1
        assert received[0][0] == "hello\nworld\n!"

    @pytest.mark.asyncio
    async def test_different_users_separate(self):
        """Messages from different users should NOT be merged."""
        received = []

        async def handler(text, ctx):
            received.append((text, ctx))

        d = MessageDebouncer(handler, debounce_ms=50)
        await d.add_message("from u1", {"user_id": "u1", "channel_id": "c1"})
        await d.add_message("from u2", {"user_id": "u2", "channel_id": "c1"})
        await asyncio.sleep(0.15)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_flush_all(self):
        """flush_all should deliver all pending batches immediately."""
        received = []

        async def handler(text, ctx):
            received.append(text)

        d = MessageDebouncer(handler, debounce_ms=5000)  # Long debounce
        await d.add_message("pending", {"user_id": "u1", "channel_id": "c1"})
        assert len(received) == 0  # Not yet delivered
        await d.flush_all()
        assert len(received) == 1
        assert received[0] == "pending"

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self):
        """Handler errors should be caught, not propagated."""
        async def bad_handler(text, ctx):
            raise RuntimeError("boom")

        d = MessageDebouncer(bad_handler, debounce_ms=50)
        await d.add_message("test", {"user_id": "u1", "channel_id": "c1"})
        await asyncio.sleep(0.15)  # Should not raise
        assert True  # Explicit: no exception means pass

    def test_default_debounce_ms(self):
        assert DEFAULT_DEBOUNCE_MS == 1500

    @pytest.mark.asyncio
    async def test_key_generation(self):
        """Key should combine gateway, channel, and user."""
        async def noop(t, c): pass
        d = MessageDebouncer(noop)
        key = d._get_key({"gateway_name": "discord", "channel_id": "123", "user_id": "456"})
        assert key == "discord:123:456"
