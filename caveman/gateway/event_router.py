"""Event Listeners — handle platform events beyond messages.

Extracted from OpenClaw listeners.ts (774 lines) and
Hermes event handling patterns.

Features:
- Reaction events (add/remove → command triggers)
- Message edit/delete tracking
- Member join/leave notifications
- Channel/thread lifecycle events
- Typing indicator forwarding
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

__all__ = [
    "EventType",
    "PlatformEvent",
    "EventRouter",
]


logger = logging.getLogger("caveman.gateway.events")


class EventType(str, Enum):
    """Platform event types."""
    REACTION_ADD = "reaction_add"
    REACTION_REMOVE = "reaction_remove"
    MESSAGE_EDIT = "message_edit"
    MESSAGE_DELETE = "message_delete"
    MEMBER_JOIN = "member_join"
    MEMBER_LEAVE = "member_leave"
    THREAD_CREATE = "thread_create"
    THREAD_DELETE = "thread_delete"
    TYPING_START = "typing_start"
    CHANNEL_UPDATE = "channel_update"


@dataclass
class PlatformEvent:
    """A non-message platform event."""
    event_type: EventType
    chat_id: str = ""
    user_id: str = ""
    message_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# Reaction → command mapping (from OpenClaw)
REACTION_COMMANDS: Dict[str, str] = {
    "🛑": "/stop",
    "⏹️": "/stop",
    "🔄": "/reset",
    "♻️": "/reset",
    "❌": "/cancel",
    "👍": "/approve",
    "👎": "/deny",
}


class EventRouter:
    """Routes platform events to handlers.

    Supports:
    - Reaction-to-command mapping
    - Event filtering by type/channel/user
    - Handler registration with priority
    """

    def __init__(self):
        self._handlers: Dict[EventType, List[tuple]] = {}  # type → [(priority, handler)]
        self._reaction_commands = dict(REACTION_COMMANDS)
        self._ignored_users: Set[str] = set()
        self._edit_tracking: Dict[str, str] = {}  # message_id → original_text

    # ── Handler Registration ──

    def on(self, event_type: EventType, handler: Callable, priority: int = 0) -> None:
        """Register an event handler."""
        handlers = self._handlers.setdefault(event_type, [])
        handlers.append((priority, handler))
        handlers.sort(key=lambda x: x[0], reverse=True)

    def off(self, event_type: EventType, handler: Callable) -> None:
        """Remove an event handler."""
        handlers = self._handlers.get(event_type, [])
        self._handlers[event_type] = [(p, h) for p, h in handlers if h is not handler]

    # ── Event Dispatch ──

    async def dispatch(self, event: PlatformEvent) -> bool:
        """Dispatch an event to registered handlers. Returns True if handled."""
        if event.user_id in self._ignored_users:
            return False

        handlers = self._handlers.get(event.event_type, [])
        for _, handler in handlers:
            try:
                result = handler(event)
                if hasattr(result, "__await__"):
                    result = await result
                if result is True:
                    return True  # Handler consumed the event
            except Exception as e:
                logger.warning("Event handler failed for %s: %s", event.event_type, e)
        return False

    # ── Reaction Commands ──

    def resolve_reaction_command(self, emoji: str) -> Optional[str]:
        """Map a reaction emoji to a command string."""
        return self._reaction_commands.get(emoji)

    def set_reaction_command(self, emoji: str, command: str) -> None:
        self._reaction_commands[emoji] = command

    def remove_reaction_command(self, emoji: str) -> None:
        self._reaction_commands.pop(emoji, None)

    # ── Edit Tracking ──

    def track_message(self, message_id: str, text: str) -> None:
        """Track a message for edit detection."""
        self._edit_tracking[message_id] = text
        # Keep cache bounded
        if len(self._edit_tracking) > 1000:
            oldest = list(self._edit_tracking.keys())[:500]
            for k in oldest:
                del self._edit_tracking[k]

    def get_original_text(self, message_id: str) -> Optional[str]:
        return self._edit_tracking.get(message_id)

    def was_edited(self, message_id: str, new_text: str) -> bool:
        """Check if a message was edited."""
        original = self._edit_tracking.get(message_id)
        if original is None:
            return False
        return original != new_text

    # ── User Management ──

    def ignore_user(self, user_id: str) -> None:
        self._ignored_users.add(user_id)

    def unignore_user(self, user_id: str) -> None:
        self._ignored_users.discard(user_id)
