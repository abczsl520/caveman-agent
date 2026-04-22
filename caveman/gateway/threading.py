"""Thread Manager — create, bind, and manage conversation threads.

Extracted from OpenClaw threading.ts (678 lines) and
Hermes thread_bindings.manager.ts (740 lines).

Features:
- Auto-thread creation for long conversations
- Thread binding (session ↔ thread)
- Thread starter context injection
- Thread name sanitization
- Forum/channel thread support
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "MAX_THREAD_NAME_LENGTH",
    "THREAD_STARTER_CACHE_TTL",
    "ThreadBinding",
    "AutoThreadConfig",
    "ThreadManager",
    "sanitize_thread_name",
]


logger = logging.getLogger("caveman.gateway.threading")

MAX_THREAD_NAME_LENGTH = 100
THREAD_STARTER_CACHE_TTL = 300  # 5 minutes


@dataclass
class ThreadBinding:
    """Binding between a session and a platform thread."""
    session_key: str
    thread_id: str
    channel_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    starter_text: str = ""
    starter_user: str = ""

    def touch(self) -> None:
        self.last_active = time.monotonic()


@dataclass
class AutoThreadConfig:
    """Configuration for automatic thread creation."""
    enabled: bool = False
    after_messages: int = 3  # Create thread after N messages in channel
    name_from_content: bool = True  # Use first message as thread name
    max_name_length: int = MAX_THREAD_NAME_LENGTH
    archive_after_minutes: int = 60


class ThreadManager:
    """Manages thread lifecycle and bindings."""

    def __init__(self, config: Optional[AutoThreadConfig] = None):
        self._config = config or AutoThreadConfig()
        self._bindings: Dict[str, ThreadBinding] = {}  # thread_id → binding
        self._session_threads: Dict[str, str] = {}  # session_key → thread_id
        self._channel_counts: Dict[str, int] = {}  # channel_id → message count
        self._starter_cache: Dict[str, tuple] = {}  # thread_id → (text, user, timestamp)

    # ── Binding Management ──

    def bind(self, session_key: str, thread_id: str, channel_id: str,
             starter_text: str = "", starter_user: str = "") -> ThreadBinding:
        """Bind a session to a thread."""
        binding = ThreadBinding(
            session_key=session_key,
            thread_id=thread_id,
            channel_id=channel_id,
            starter_text=starter_text,
            starter_user=starter_user,
        )
        self._bindings[thread_id] = binding
        self._session_threads[session_key] = thread_id
        logger.debug("Bound session %s → thread %s", session_key, thread_id)
        return binding

    def unbind(self, thread_id: str) -> Optional[ThreadBinding]:
        """Remove a thread binding."""
        binding = self._bindings.pop(thread_id, None)
        if binding:
            self._session_threads.pop(binding.session_key, None)
        return binding

    def get_binding(self, thread_id: str) -> Optional[ThreadBinding]:
        binding = self._bindings.get(thread_id)
        if binding:
            binding.touch()
        return binding

    def get_thread_for_session(self, session_key: str) -> Optional[str]:
        return self._session_threads.get(session_key)

    def is_bound_thread(self, thread_id: str) -> bool:
        return thread_id in self._bindings

    def list_bindings(self) -> List[Dict[str, Any]]:
        return [
            {
                "thread_id": b.thread_id,
                "session_key": b.session_key,
                "channel_id": b.channel_id,
                "age_seconds": round(time.monotonic() - b.created_at, 1),
            }
            for b in self._bindings.values()
        ]

    # ── Auto-Thread ──

    def should_create_thread(self, channel_id: str) -> bool:
        """Check if we should auto-create a thread for this channel."""
        if not self._config.enabled:
            return False
        count = self._channel_counts.get(channel_id, 0) + 1
        self._channel_counts[channel_id] = count
        return count >= self._config.after_messages

    def reset_channel_count(self, channel_id: str) -> None:
        self._channel_counts.pop(channel_id, None)

    # ── Thread Starter Context ──

    def cache_starter(self, thread_id: str, text: str, user: str) -> None:
        self._starter_cache[thread_id] = (text, user, time.monotonic())

    def get_starter(self, thread_id: str) -> Optional[tuple]:
        """Get cached thread starter. Returns (text, user) or None."""
        entry = self._starter_cache.get(thread_id)
        if not entry:
            return None
        text, user, ts = entry
        if time.monotonic() - ts > THREAD_STARTER_CACHE_TTL:
            del self._starter_cache[thread_id]
            return None
        return (text, user)

    def build_starter_context(self, thread_id: str) -> str:
        """Build context string from thread starter for injection."""
        starter = self.get_starter(thread_id)
        if not starter:
            binding = self._bindings.get(thread_id)
            if binding and binding.starter_text:
                return f"[Thread starter by {binding.starter_user}]\n{binding.starter_text}"
            return ""
        text, user = starter
        return f"[Thread starter by {user}]\n{text}"

    # ── Cleanup ──

    def reap_stale(self, max_idle: float = 3600) -> int:
        """Remove stale bindings. Returns count removed."""
        now = time.monotonic()
        stale = [
            tid for tid, b in self._bindings.items()
            if now - b.last_active > max_idle
        ]
        for tid in stale:
            self.unbind(tid)
        return len(stale)


# ── Utilities ──

def sanitize_thread_name(raw: str, fallback_id: str = "") -> str:
    """Sanitize a string for use as a thread name."""
    # Remove markdown, mentions, URLs
    name = re.sub(r"<@!?\d+>", "", raw)
    name = re.sub(r"https?://\S+", "", name)
    name = re.sub(r"[*_`~|#]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    if not name:
        return fallback_id or "Thread"

    if len(name) > MAX_THREAD_NAME_LENGTH:
        name = name[:MAX_THREAD_NAME_LENGTH - 3] + "..."

    return name
