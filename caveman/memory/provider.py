"""Memory provider abstraction — pluggable backends.

Ported from Hermes MemoryProvider ABC (MIT, Nous Research) with Caveman
adaptations. Simplified to essential hooks for single-user agent.

Lifecycle (called by MemoryManager):
  initialize()           — connect, create resources
  system_prompt_block()  — static text for system prompt
  prefetch(query)        — recall before each turn
  sync_turn(user, asst)  — persist after each turn
  on_pre_compress(msgs)  — extract before context compression
  on_session_end(msgs)   — end-of-session extraction
  shutdown()             — clean exit

Registration:
  Built-in: SQLiteMemoryStore (always active)
  External: One optional provider via config (memory.provider)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from caveman.memory.types import MemoryType, MemoryEntry

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Abstract base class for memory providers.

    Ported from Hermes (MIT, Nous Research). Simplified for Caveman.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'builtin', 'honcho', 'mem0')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and ready. No network calls."""

    @abstractmethod
    async def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Initialize for a session. Called once at agent startup."""

    @abstractmethod
    async def store(self, content: str, memory_type: MemoryType,
                    metadata: dict | None = None, *, trusted: bool = False) -> str:
        """Store a memory. Returns memory ID."""

    @abstractmethod
    async def recall(self, query: str, memory_type: MemoryType | None = None,
                     top_k: int = 5) -> list[MemoryEntry]:
        """Recall relevant memories for a query."""

    @abstractmethod
    async def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""

    def system_prompt_block(self) -> str:
        """Static text for system prompt. Override to contribute."""
        return ""

    def prefetch(self, query: str) -> str:
        """Recall context for upcoming turn. Should be fast."""
        return ""

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        """Persist a completed turn. Should be non-blocking."""

    def on_pre_compress(self, messages: list[dict]) -> str:
        """Extract insights before context compression.

        Return text to include in compression summary prompt.
        This is the Shield integration point — Shield calls this
        before compaction to preserve provider-specific context.
        """
        return ""

    def on_session_end(self, messages: list[dict]) -> None:
        """End-of-session extraction. Called on exit/reset."""

    async def mark_helpful(self, memory_id: str, helpful: bool = True) -> None:
        """Trust feedback. Override to adjust retrieval weight."""

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""


class BuiltinMemoryProvider(MemoryProvider):
    """Wraps SQLiteMemoryStore as a MemoryProvider.

    Always active. Cannot be removed.
    """

    def __init__(self, store: Any = None) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    async def initialize(self, session_id: str, **kwargs: Any) -> None:
        if self._store is None:
            from caveman.memory.sqlite_store import SQLiteMemoryStore
            self._store = SQLiteMemoryStore()

    async def store(self, content: str, memory_type: MemoryType,
                    metadata: dict | None = None) -> str:
        return await self._store.store(content, memory_type, metadata)

    async def recall(self, query: str, memory_type: MemoryType | None = None,
                     top_k: int = 5) -> list[MemoryEntry]:
        return await self._store.recall(query, memory_type, top_k)

    async def forget(self, memory_id: str) -> bool:
        return await self._store.forget(memory_id)

    async def mark_helpful(self, memory_id: str, helpful: bool = True) -> None:
        if hasattr(self._store, "mark_helpful"):
            await self._store.mark_helpful(memory_id, helpful)

    def shutdown(self) -> None:
        if hasattr(self._store, "close"):
            self._store.close()
