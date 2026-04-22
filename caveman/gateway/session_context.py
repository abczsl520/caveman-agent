"""Session-scoped context variables for the gateway.

Replaces os.environ-based session state with Python's contextvars.ContextVar.
Each asyncio task gets its own copy, so concurrent messages never interfere.
"""
from __future__ import annotations

from contextvars import ContextVar
import logging

__all__ = [
    "set_session_context",
    "get_session_env",
    "get_session_context",
    "clear_session_context",
]


logger = logging.getLogger(__name__)

# Core session context variables
_session_platform: ContextVar[str] = ContextVar("session_platform", default="")
_session_chat_id: ContextVar[str] = ContextVar("session_chat_id", default="")
_session_thread_id: ContextVar[str] = ContextVar("session_thread_id", default="")
_session_sender_id: ContextVar[str] = ContextVar("session_sender_id", default="")
_session_agent_id: ContextVar[str] = ContextVar("session_agent_id", default="main")
_session_key: ContextVar[str] = ContextVar("session_key", default="")

_VARS = {
    "CAVEMAN_SESSION_PLATFORM": _session_platform,
    "CAVEMAN_SESSION_CHAT_ID": _session_chat_id,
    "CAVEMAN_SESSION_THREAD_ID": _session_thread_id,
    "CAVEMAN_SESSION_SENDER_ID": _session_sender_id,
    "CAVEMAN_SESSION_AGENT_ID": _session_agent_id,
    "CAVEMAN_SESSION_KEY": _session_key,
}


def set_session_context(
    platform: str = "",
    chat_id: str = "",
    thread_id: str = "",
    sender_id: str = "",
    agent_id: str = "main",
    session_key: str = "",
) -> None:
    """Set session context for the current async task."""
    if platform:
        _session_platform.set(platform)
    if chat_id:
        _session_chat_id.set(chat_id)
    if thread_id:
        _session_thread_id.set(thread_id)
    if sender_id:
        _session_sender_id.set(sender_id)
    if agent_id:
        _session_agent_id.set(agent_id)
    if session_key:
        _session_key.set(session_key)


def get_session_env(name: str, default: str = "") -> str:
    """Get a session context variable by name.

    Compatible with the old os.getenv("CAVEMAN_SESSION_*") pattern.
    """
    var = _VARS.get(name)
    if var is None:
        return default
    try:
        return var.get()
    except LookupError:
        return default


def get_session_context() -> dict[str, str]:
    """Get all session context variables."""
    return {name: get_session_env(name) for name in _VARS}


def clear_session_context() -> None:
    """Clear all session context variables."""
    for var in _VARS.values():
        try:
            var.set("")
        except Exception as exc:
            logger.debug("clear_session_context: suppressed %s", exc)
