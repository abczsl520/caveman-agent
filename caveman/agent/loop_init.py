"""Initialization helpers for AgentLoop."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def build_fallback_chain():
    """Create the auxiliary fallback chain if configured."""
    try:
        from caveman.agent.auxiliary_client import AuxiliaryConfig, _FallbackChain
        cfg = AuxiliaryConfig.from_env()
        return _FallbackChain(cfg) if cfg.provider else None
    except Exception as exc:
        logger.debug("build_fallback_chain: suppressed %s", exc)
        return None


def ensure_provider(provider, model):
    """Return a concrete LLM provider."""
    if provider is not None:
        return provider
    from caveman.providers.anthropic_provider import AnthropicProvider
    return AnthropicProvider(api_key=os.environ.get("ANTHROPIC_API_KEY", ""), model=model)


def ensure_memory_manager(memory_manager):
    """Return the configured memory manager, falling back to legacy JSON if SQLite is unavailable."""
    if memory_manager is not None:
        return memory_manager
    from caveman.memory.manager import MemoryManager
    try:
        return MemoryManager.with_sqlite()
    except Exception as exc:
        logger.warning("SQLite memory backend unavailable; falling back to legacy JSON memory: %s", exc)
        return MemoryManager()


def set_default_permissions(permission_manager):
    """Default loop-local permissions to AUTO for non-interactive gateway operation."""
    from caveman.security.permissions import PermissionLevel, PermissionManager
    permission_manager = permission_manager or PermissionManager()
    for key in list(permission_manager._permissions):
        permission_manager._permissions[key] = PermissionLevel.AUTO
    return permission_manager
