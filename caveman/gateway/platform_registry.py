"""Platform Registry — discover and instantiate platform adapters.

Central registry for all available platform adapters. Used by the gateway
runner to start configured platforms.

Usage:
    from caveman.gateway.platform_registry import get_adapter, list_platforms

    adapter = get_adapter("discord", config)
    await adapter.connect()
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from caveman.gateway.platform_adapter import BasePlatformAdapter
from caveman.gateway.platform_types import PlatformConfig

__all__ = [
    "register_adapter",
    "get_adapter",
    "list_platforms",
]


logger = logging.getLogger("caveman.gateway")

# Registry of platform name → adapter class
_ADAPTERS: Dict[str, Type[BasePlatformAdapter]] = {}


def register_adapter(platform: str, adapter_class: Type[BasePlatformAdapter]) -> None:
    """Register a platform adapter class."""
    _ADAPTERS[platform.lower()] = adapter_class


def get_adapter(platform: str, config: PlatformConfig) -> Optional[BasePlatformAdapter]:
    """Get an instantiated adapter for a platform."""
    cls = _ADAPTERS.get(platform.lower())
    if cls is None:
        logger.error("Unknown platform: %s (available: %s)", platform, list(_ADAPTERS.keys()))
        return None
    return cls(config)


def list_platforms() -> list[str]:
    """List all registered platform names."""
    return sorted(_ADAPTERS.keys())


def _register_builtins() -> None:
    """Register all built-in platform adapters."""
    from caveman.gateway.discord_adapter import DiscordAdapter
    from caveman.gateway.telegram_adapter import TelegramAdapter
    from caveman.gateway.slack_adapter import SlackAdapter
    from caveman.gateway.whatsapp_adapter import WhatsAppAdapter
    from caveman.gateway.signal_adapter import SignalAdapter
    from caveman.gateway.matrix_adapter import MatrixAdapter
    from caveman.gateway.feishu_adapter import FeishuAdapter

    register_adapter("discord", DiscordAdapter)
    register_adapter("telegram", TelegramAdapter)
    register_adapter("slack", SlackAdapter)
    register_adapter("whatsapp", WhatsAppAdapter)
    register_adapter("signal", SignalAdapter)
    register_adapter("matrix", MatrixAdapter)
    register_adapter("feishu", FeishuAdapter)


# Auto-register on import
_register_builtins()
