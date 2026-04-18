"""Content-hash based deduplication for imports."""
from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ImportDedup:
    """Deduplicates import items by SHA256 content hash."""

    def __init__(self, memory_manager: Any | None = None) -> None:
        self.seen_hashes: set[str] = set()
        if memory_manager is not None:
            self._load_existing(memory_manager)

    def _load_existing(self, memory_manager: Any) -> None:
        """Load hashes of existing memories to avoid re-importing."""
        try:
            for entry in memory_manager.all_entries():
                h = content_hash(entry.content)
                self.seen_hashes.add(h)
        except Exception as e:
            logger.debug("Could not load existing hashes: %s", e)

    def is_duplicate(self, content: str) -> bool:
        """Check if content already exists. Adds to seen set if new."""
        h = content_hash(content)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False


def content_hash(text: str) -> str:
    """SHA256 hash of content, truncated to 16 hex chars."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]
