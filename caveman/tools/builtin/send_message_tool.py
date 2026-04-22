"""Send Message Tool — cross-platform message sending.

Extracted from Hermes send_message_tool.py (1049 lines).
Enables the agent to send messages to any connected platform.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from caveman.tools.registry import tool

__all__ = [
    "register_platform",
    "get_platform",
    "list_platforms",
    "send_message",
    "send_typing_indicator",
]


logger = logging.getLogger("caveman.tools.send_message")

# Platform registry (populated at runtime)
_platform_adapters: Dict[str, Any] = {}


def register_platform(name: str, adapter: Any) -> None:
    _platform_adapters[name.lower()] = adapter


def get_platform(name: str) -> Optional[Any]:
    return _platform_adapters.get(name.lower())


def list_platforms() -> List[str]:
    return sorted(_platform_adapters.keys())


@tool(
    name="send_message",
    description="Send a message to a chat on any connected platform",
    params={
        "platform": {"type": "string", "description": "Platform name (discord, telegram, etc.)"},
        "chat_id": {"type": "string", "description": "Chat/channel ID"},
        "message": {"type": "string", "description": "Message text"},
        "reply_to": {"type": "string", "description": "Message ID to reply to (optional)"},
    },
    required=["platform", "chat_id", "message"],
)
async def send_message(
    platform: str, chat_id: str, message: str,
    reply_to: str = "",
) -> Dict[str, Any]:
    """Send a message to a specific platform and chat."""
    adapter = get_platform(platform)
    if not adapter:
        available = list_platforms()
        return {
            "ok": False,
            "error": f"Platform '{platform}' not connected. Available: {', '.join(available) or 'none'}",
        }

    try:
        result = await adapter.send(
            chat_id=chat_id,
            content=message,
            reply_to=reply_to or None,
        )
        return {
            "ok": result.success,
            "message_id": result.message_id,
            "error": result.error,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="send_typing",
    description="Send typing indicator to a chat",
    params={
        "platform": {"type": "string", "description": "Platform name"},
        "chat_id": {"type": "string", "description": "Chat/channel ID"},
    },
    required=["platform", "chat_id"],
)
async def send_typing_indicator(platform: str, chat_id: str) -> Dict[str, Any]:
    """Send typing indicator."""
    adapter = get_platform(platform)
    if not adapter:
        return {"ok": False, "error": f"Platform '{platform}' not connected"}
    try:
        if hasattr(adapter, "send_typing"):
            await adapter.send_typing(chat_id)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
