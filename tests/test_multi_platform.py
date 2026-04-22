"""Tests for Slack, WhatsApp, and Signal adapters."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, Optional

from caveman.gateway.platform_types import (
    MessageEvent, MessageType, Platform, PlatformConfig, SendResult,
)
from caveman.gateway.slack_adapter import SlackAdapter
from caveman.gateway.whatsapp_adapter import WhatsAppAdapter
from caveman.gateway.signal_adapter import SignalAdapter


def _make_config(platform="slack", **kwargs) -> PlatformConfig:
    defaults = {"enabled": True, "token": "xoxb-fake"}
    defaults.update(kwargs)
    return PlatformConfig.from_dict(defaults)


# ── Slack Tests ─────────────────────────────────────────────────────────────

class TestSlackAdapter:
    def test_init(self):
        config = _make_config(app_token="xapp-fake")
        adapter = SlackAdapter(config)
        assert adapter.name == "Slack"
        assert adapter._max_message_length == 3000

    @pytest.mark.asyncio
    async def test_connect_no_library(self):
        with patch("caveman.gateway.slack_adapter.SLACK_AVAILABLE", False):
            adapter = SlackAdapter(_make_config())
            result = await adapter.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_no_app_token(self):
        config = PlatformConfig(enabled=True, token="xoxb-fake")
        adapter = SlackAdapter(config)
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_send_not_connected(self):
        adapter = SlackAdapter(_make_config())
        result = await adapter.send("C123", "hello")
        assert not result.success

    @pytest.mark.asyncio
    async def test_send_with_app(self):
        adapter = SlackAdapter(_make_config())
        mock_app = MagicMock()
        mock_app.client.chat_postMessage = AsyncMock(return_value={"ts": "123.456"})
        adapter._app = mock_app

        result = await adapter.send("C123", "hello")
        assert result.success
        assert result.message_id == "123.456"


# ── WhatsApp Tests ──────────────────────────────────────────────────────────

class TestWhatsAppAdapter:
    def test_init(self):
        config = _make_config(phone_number_id="12345")
        adapter = WhatsAppAdapter(config)
        assert adapter.name == "WhatsApp"
        assert adapter._phone_id == "12345"

    @pytest.mark.asyncio
    async def test_connect_no_token(self):
        config = PlatformConfig(enabled=True, token="")
        adapter = WhatsAppAdapter(config)
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_send_not_connected(self):
        config = PlatformConfig(enabled=True, token="")
        adapter = WhatsAppAdapter(config)
        result = await adapter.send("+1234567890", "hello")
        assert not result.success

    def test_handle_webhook_text(self):
        config = _make_config(phone_number_id="12345")
        adapter = WhatsAppAdapter(config)

        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "+1234567890",
                            "id": "wamid.123",
                            "type": "text",
                            "text": {"body": "Hello!"},
                        }],
                        "contacts": [{"profile": {"name": "Alice"}}],
                    }
                }]
            }]
        }

        event = adapter.handle_webhook(json.dumps(payload).encode())
        assert event is not None
        assert event.text == "Hello!"
        assert event.source.user_id == "+1234567890"
        assert event.source.user_name == "Alice"
        assert event.message_type == MessageType.TEXT

    def test_handle_webhook_image(self):
        config = _make_config(phone_number_id="12345")
        adapter = WhatsAppAdapter(config)

        payload = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "+1234567890",
                            "id": "wamid.456",
                            "type": "image",
                            "image": {"id": "media_123", "caption": "Look!"},
                        }],
                        "contacts": [{"profile": {"name": "Bob"}}],
                    }
                }]
            }]
        }

        event = adapter.handle_webhook(json.dumps(payload).encode())
        assert event is not None
        assert event.text == "Look!"
        assert event.message_type == MessageType.PHOTO
        assert "media_123" in event.media_urls

    def test_handle_webhook_no_messages(self):
        config = _make_config(phone_number_id="12345")
        adapter = WhatsAppAdapter(config)

        payload = {"entry": [{"changes": [{"value": {"statuses": []}}]}]}
        event = adapter.handle_webhook(json.dumps(payload).encode())
        assert event is None


# ── Signal Tests ────────────────────────────────────────────────────────────

class TestSignalAdapter:
    def test_init(self):
        config = _make_config(phone_number="+1234567890", api_url="http://localhost:8080")
        adapter = SignalAdapter(config)
        assert adapter.name == "Signal"
        assert adapter._phone_number == "+1234567890"

    @pytest.mark.asyncio
    async def test_connect_no_phone(self):
        config = PlatformConfig(enabled=True)
        adapter = SignalAdapter(config)
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_send_not_connected(self):
        config = _make_config(phone_number="+1234567890")
        adapter = SignalAdapter(config)
        # Will fail because signal-cli isn't running
        result = await adapter.send("+9876543210", "hello")
        assert not result.success

    def test_parse_message_text(self):
        config = _make_config(phone_number="+1234567890")
        adapter = SignalAdapter(config)

        raw = {
            "envelope": {
                "source": "+9876543210",
                "sourceName": "Alice",
                "dataMessage": {
                    "timestamp": 1234567890,
                    "message": "Hello from Signal!",
                },
            }
        }

        event = adapter._parse_message(raw)
        assert event is not None
        assert event.text == "Hello from Signal!"
        assert event.source.user_id == "+9876543210"
        assert event.source.chat_type == "dm"

    def test_parse_message_group(self):
        config = _make_config(phone_number="+1234567890")
        adapter = SignalAdapter(config)

        raw = {
            "envelope": {
                "source": "+9876543210",
                "sourceName": "Bob",
                "dataMessage": {
                    "timestamp": 1234567890,
                    "message": "Group message",
                    "groupInfo": {"groupId": "abc123"},
                },
            }
        }

        event = adapter._parse_message(raw)
        assert event is not None
        assert event.source.chat_id == "group.abc123"
        assert event.source.chat_type == "group"

    def test_parse_message_no_data(self):
        config = _make_config(phone_number="+1234567890")
        adapter = SignalAdapter(config)

        raw = {"envelope": {"source": "+9876543210", "syncMessage": {}}}
        event = adapter._parse_message(raw)
        assert event is None
