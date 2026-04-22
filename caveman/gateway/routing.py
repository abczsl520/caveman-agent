"""Message routing — resolve which agent/session handles a message.

Maps inbound messages to the correct agent session based on:
- Platform + chat_id → session binding
- Account ID resolution
- Thread routing
- Multi-agent routing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

__all__ = [
    "Route",
    "SessionBinding",
    "resolve_route",
    "parse_session_key",
    "resolve_account_id",
]


logger = logging.getLogger(__name__)


@dataclass
class Route:
    """Resolved message route."""
    session_key: str
    agent_id: str = "main"
    account_id: str = "default"
    is_new_session: bool = False
    binding: str = ""  # How the route was resolved


@dataclass
class SessionBinding:
    """Maps a platform identity to a session."""
    platform: str
    chat_id: str
    thread_id: str = ""
    agent_id: str = "main"
    account_id: str = "default"


def resolve_route(
    platform: str,
    chat_id: str,
    sender_id: str = "",
    thread_id: str = "",
    agent_id: str = "main",
    bindings: dict[str, SessionBinding] | None = None,
) -> Route:
    """Resolve the session route for an inbound message.

    Priority:
    1. Explicit thread binding (thread_id match)
    2. Chat binding (platform + chat_id match)
    3. Generate new session key
    """
    # Build lookup key
    if thread_id:
        thread_key = _binding_key(platform, chat_id, thread_id)
        if bindings and thread_key in bindings:
            b = bindings[thread_key]
            return Route(
                session_key=_session_key(b.agent_id, platform, chat_id, thread_id),
                agent_id=b.agent_id,
                account_id=b.account_id,
                binding="thread",
            )

    chat_key = _binding_key(platform, chat_id)
    if bindings and chat_key in bindings:
        b = bindings[chat_key]
        return Route(
            session_key=_session_key(b.agent_id, platform, chat_id, thread_id),
            agent_id=b.agent_id,
            account_id=b.account_id,
            binding="chat",
        )

    # New session
    session_key = _session_key(agent_id, platform, chat_id, thread_id)
    return Route(
        session_key=session_key,
        agent_id=agent_id,
        is_new_session=True,
        binding="new",
    )


def _session_key(agent_id: str, platform: str, chat_id: str, thread_id: str = "") -> str:
    """Generate a deterministic session key."""
    parts = [f"agent:{agent_id}", platform, f"channel:{chat_id}"]
    if thread_id:
        parts.append(f"thread:{thread_id}")
    return ":".join(parts)


def _binding_key(platform: str, chat_id: str, thread_id: str = "") -> str:
    parts = [platform.lower(), str(chat_id)]
    if thread_id:
        parts.append(str(thread_id))
    return ":".join(parts)


def parse_session_key(key: str) -> dict[str, str] | None:
    """Parse a session key into components."""
    parts = key.split(":")
    result: dict[str, str] = {}

    i = 0
    while i < len(parts):
        if parts[i] == "agent" and i + 1 < len(parts):
            result["agent_id"] = parts[i + 1]
            i += 2
        elif parts[i] == "channel" and i + 1 < len(parts):
            result["chat_id"] = parts[i + 1]
            i += 2
        elif parts[i] == "thread" and i + 1 < len(parts):
            result["thread_id"] = parts[i + 1]
            i += 2
        elif "platform" not in result:
            result["platform"] = parts[i]
            i += 1
        else:
            i += 1

    return result if result else None


def resolve_account_id(
    sender_id: str,
    platform: str,
    account_map: dict[str, str] | None = None,
) -> str:
    """Resolve account ID from sender identity.

    account_map: {sender_id: account_id} or {platform:sender_id: account_id}
    """
    if not account_map:
        return "default"

    # Try platform-specific lookup
    platform_key = f"{platform}:{sender_id}"
    if platform_key in account_map:
        return account_map[platform_key]

    # Try sender-only lookup
    if sender_id in account_map:
        return account_map[sender_id]

    return "default"
