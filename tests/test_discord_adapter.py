"""Tests for DiscordAdapter."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig,
    ProcessingOutcome, SendResult, SessionSource,
)
from caveman.gateway.discord_adapter import DiscordAdapter


def _make_config(**kwargs) -> PlatformConfig:
    defaults = {"enabled": True, "token": "fake-token"}
    defaults.update(kwargs)
    return PlatformConfig.from_dict(defaults)


class TestDiscordAdapterInit:
    def test_defaults(self):
        adapter = DiscordAdapter(_make_config())
        assert adapter.name == "Discord"
        assert adapter._max_message_length == 1900
        assert adapter._trigger == "all"

    def test_custom_config(self):
        config = _make_config(trigger="thread", prefix="!ai", allowed_users=[123])
        adapter = DiscordAdapter(config)
        assert adapter._trigger == "thread"
        assert adapter._prefix == "!ai"
        assert 123 in adapter._allowed_users


class TestPermissions:
    def test_no_filters_allows_all(self):
        adapter = DiscordAdapter(_make_config())
        msg = MagicMock()
        msg.channel.id = 999
        msg.author.id = 111
        assert adapter._check_permissions(msg) is True

    def test_channel_filter(self):
        config = _make_config(allowed_channels=[100, 200])
        adapter = DiscordAdapter(config)

        msg = MagicMock()
        msg.channel.id = 100
        msg.channel.parent_id = None
        msg.author.id = 1
        assert adapter._check_permissions(msg) is True

        msg.channel.id = 999
        msg.channel.parent_id = None
        assert adapter._check_permissions(msg) is False

    def test_thread_in_allowed_channel(self):
        config = _make_config(allowed_channels=[100])
        adapter = DiscordAdapter(config)

        msg = MagicMock()
        msg.channel.id = 555  # Thread ID
        msg.channel.parent_id = 100  # Parent is allowed
        msg.author.id = 1
        assert adapter._check_permissions(msg) is True

    def test_user_filter(self):
        config = _make_config(allowed_users=[42])
        adapter = DiscordAdapter(config)

        msg = MagicMock()
        msg.channel.id = 1
        msg.author.id = 42
        assert adapter._check_permissions(msg) is True

        msg.author.id = 99
        assert adapter._check_permissions(msg) is False


class TestRateLimit:
    def test_allows_under_limit(self):
        adapter = DiscordAdapter(_make_config())
        for _ in range(5):
            assert adapter._check_rate_limit(123) is True

    def test_blocks_over_limit(self):
        adapter = DiscordAdapter(_make_config())
        for _ in range(5):
            adapter._check_rate_limit(123)
        assert adapter._check_rate_limit(123) is False

    def test_allowlisted_user_bypasses(self):
        config = _make_config(allowed_users=[42])
        adapter = DiscordAdapter(config)
        for _ in range(10):
            assert adapter._check_rate_limit(42) is True


class TestShouldRespond:
    def _setup(self, trigger="all"):
        config = _make_config(trigger=trigger)
        adapter = DiscordAdapter(config)
        adapter._client = MagicMock()
        adapter._client.user = MagicMock()
        adapter._client.user.id = 999
        return adapter

    def test_all_mode(self):
        adapter = self._setup("all")
        msg = MagicMock()
        msg.mentions = []
        assert adapter._should_respond("hello", msg) is True

    def test_prefix_mode_with_prefix(self):
        adapter = self._setup("prefix")
        msg = MagicMock()
        msg.mentions = []
        msg.channel = MagicMock(spec=[])  # Not DMChannel
        assert adapter._should_respond("!cave hello", msg) is True

    def test_prefix_mode_without_prefix(self):
        adapter = self._setup("prefix")
        msg = MagicMock()
        msg.mentions = []
        msg.channel = MagicMock(spec=[])  # Not DMChannel or Thread
        # Need to make isinstance checks fail
        with patch("caveman.gateway.discord_adapter.discord") as mock_discord:
            mock_discord.DMChannel = type("DMChannel", (), {})
            mock_discord.Thread = type("Thread", (), {})
            assert adapter._should_respond("hello", msg) is False


class TestBuildEvent:
    def _setup(self):
        adapter = DiscordAdapter(_make_config())
        adapter._client = MagicMock()
        adapter._client.user = MagicMock()
        adapter._client.user.id = 999
        return adapter

    @patch("caveman.gateway.discord_adapter.discord")
    def test_basic_text(self, mock_discord):
        mock_discord.DMChannel = type("DMChannel", (), {})
        mock_discord.Thread = type("Thread", (), {})

        adapter = self._setup()
        msg = MagicMock()
        msg.channel.id = 123
        msg.channel.name = "general"
        msg.channel.topic = None
        msg.author.id = 456
        msg.author.__str__ = lambda self: "User#1234"
        msg.id = 789
        msg.attachments = []
        msg.mentions = []
        msg.reference = None

        event = adapter._build_event("hello world", msg)
        assert event.text == "hello world"
        assert event.source.chat_id == "123"
        assert event.source.user_id == "456"
        assert event.source.platform == Platform.DISCORD
        assert event.message_type == MessageType.TEXT

    @patch("caveman.gateway.discord_adapter.discord")
    def test_with_attachment(self, mock_discord):
        mock_discord.DMChannel = type("DMChannel", (), {})
        mock_discord.Thread = type("Thread", (), {})

        adapter = self._setup()
        msg = MagicMock()
        msg.channel.id = 123
        msg.channel.name = "general"
        msg.channel.topic = None
        msg.author.id = 456
        msg.author.__str__ = lambda self: "User"
        msg.id = 789
        msg.mentions = []
        msg.reference = None

        att = MagicMock()
        att.filename = "photo.jpg"
        att.url = "https://cdn.discord.com/photo.jpg"
        att.content_type = "image/jpeg"
        msg.attachments = [att]

        event = adapter._build_event("check this", msg)
        assert event.message_type == MessageType.PHOTO
        assert len(event.media_urls) == 1
        assert "photo.jpg" in event.text


class TestSendResult:
    @pytest.mark.asyncio
    async def test_send_not_connected(self):
        adapter = DiscordAdapter(_make_config())
        result = await adapter.send("123", "hello")
        assert not result.success
        assert "Not connected" in result.error
