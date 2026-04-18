"""MemoryBackend Protocol — unified interface for pluggable memory storage.

Unifies MemoryManager (JSON) and SQLiteMemoryStore behind a single interface.
Default backend: SQLite + FTS5. JSON kept as fallback.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, List

from .types import MemoryType, MemoryEntry


@runtime_checkable
class MemoryBackend(Protocol):
    """Pluggable memory storage backend.

    Any backend must implement these methods. Both SQLite and JSON
    backends conform to this protocol.
    """

    async def store(self, content: str, memory_type: MemoryType,
                    metadata: dict | None = None, *, trusted: bool = False) -> str:
        """Store a memory. Returns memory ID.

        Args:
            trusted: If True, skip security scanning. Used for internal
                     operations (e.g. nudge, ripple) where content is
                     already validated.
        """
        ...

    async def recall(self, query: str, memory_type: MemoryType | None = None,
                     top_k: int = 5) -> List[MemoryEntry]:
        """Recall relevant memories."""
        ...

    async def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        ...

    @property
    def total_count(self) -> int:
        """Total number of stored memories."""
        ...

    def all_entries(self) -> list[MemoryEntry]:
        """Return all memories as a flat list."""
        ...

    def search_sync(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Synchronous keyword search (no embedding)."""
        ...

    def recent(self, limit: int = 20) -> list[MemoryEntry]:
        """Return most recent memories."""
        ...

    async def update_metadata(self, memory_id: str, metadata: dict) -> bool:
        """Update metadata for an existing memory. Returns True if found."""
        ...

    def close(self) -> None:
        """Clean shutdown."""
        ...
