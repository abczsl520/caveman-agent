"""Tests for TelegramAdapter."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig, SendResult,
)
from caveman.gateway.telegram_adapter import (
    TelegramAdapter, format_markdown_v2, _escape_mdv2,
)


def _make_config(**kwargs) -> PlatformConfig:
    defaults = {"enabled": True, "token": "123:fake-token"}
    defaults.update(kwargs)
    return PlatformConfig.from_dict(defaults)


class TestMarkdownV2:
    def test_escape_special_chars(self):
        assert _escape_mdv2("hello.world") == r"hello\.world"
        assert _escape_mdv2("a+b=c") == r"a\+b\=c"

    def test_format_bold(self):
        result = format_markdown_v2("**bold**")
        assert "*" in result
        # Should not have ** anymore
        assert "**" not in result

    def test_format_code_block_preserved(self):
        content = "text\n```python\nx = 1\n```\nmore"
        result = format_markdown_v2(content)
        assert "```python" in result
        assert "x = 1" in result

    def test_format_inline_code_preserved(self):
        content = "use `pip install` to install"
        result = format_markdown_v2(content)
        assert "`pip install`" in result

    def test_format_link(self):
        content = "visit [Google](https://google.com)"
        result = format_markdown_v2(content)
        assert "https://google.com" in result


class TestTelegramAdapterInit:
    def test_defaults(self):
        adapter = TelegramAdapter(_make_config())
        assert adapter.name == "Telegram"
        assert adapter._max_message_length == 4096
        assert adapter._token == "123:fake-token"

    def test_webhook_config(self):
        config = _make_config(webhook=True, webhook_url="https://example.com/hook")
        adapter = TelegramAdapter(config)
        assert adapter._use_webhook is True
        assert adapter._webhook_url == "https://example.com/hook"


class TestTelegramSend:
    @pytest.mark.asyncio
    async def test_send_not_connected(self):
        adapter = TelegramAdapter(_make_config())
        result = await adapter.send("123", "hello")
        assert not result.success
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_with_bot(self):
        adapter = TelegramAdapter(_make_config())
        mock_bot = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        adapter._bot = mock_bot

        result = await adapter.send("123", "hello")
        assert result.success
        assert result.message_id == "42"

    @pytest.mark.asyncio
    async def test_send_mdv2_fallback(self):
        """When MDV2 formatting fails (AttributeError on ParseMode), falls back to plain text."""
        adapter = TelegramAdapter(_make_config())
        mock_bot = AsyncMock()
        mock_msg = MagicMock()
        mock_msg.message_id = 42
        mock_bot.send_message = AsyncMock(return_value=mock_msg)
        adapter._bot = mock_bot

        # ParseMode is Any (not installed), so ParseMode.MARKDOWN_V2 raises
        # The except catches it and sends plain text
        result = await adapter.send("123", "**bold**")
        assert result.success
        # Should have been called once (the plain text fallback)
        assert mock_bot.send_message.call_count == 1
        # Verify it was called without parse_mode (plain text)
        call_kwargs = mock_bot.send_message.call_args[1]
        assert "parse_mode" not in call_kwargs


class TestTelegramMedia:
    @pytest.mark.asyncio
    async def test_send_voice_not_connected(self):
        adapter = TelegramAdapter(_make_config())
        result = await adapter.send_voice("123", "/tmp/voice.ogg")
        assert not result.success

    @pytest.mark.asyncio
    async def test_send_image_not_connected(self):
        adapter = TelegramAdapter(_make_config())
        result = await adapter.send_image_file("123", "/tmp/photo.jpg")
        assert not result.success

    @pytest.mark.asyncio
    async def test_edit_message_not_connected(self):
        adapter = TelegramAdapter(_make_config())
        result = await adapter.edit_message("123", "456", "new text")
        assert not result.success


class TestTelegramConnect:
    @pytest.mark.asyncio
    async def test_connect_no_library(self):
        with patch("caveman.gateway.telegram_adapter.TELEGRAM_AVAILABLE", False):
            adapter = TelegramAdapter(_make_config())
            result = await adapter.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_no_token(self):
        config = PlatformConfig(enabled=True, token="")
        adapter = TelegramAdapter(config)
        result = await adapter.connect()
        assert result is False
