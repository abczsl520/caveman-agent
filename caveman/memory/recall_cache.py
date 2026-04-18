"""LRU recall cache with TTL expiration."""
from __future__ import annotations

import hashlib
import time
from typing import Any

from .types import MemoryType

_CACHE_TTL = 5  # seconds — short TTL so retrieval_count/trust updates aren't skipped
_CACHE_MAX = 50


class RecallCache:
    """Simple TTL + LRU cache for recall results."""

    def __init__(self, max_size: int = _CACHE_MAX, ttl: float = _CACHE_TTL) -> None:
        self._data: dict[str, tuple[float, list]] = {}  # key -> (timestamp, results)
        self._max_size = max_size
        self._ttl = ttl

    @staticmethod
    def _key(query: str, top_k: int, memory_type: MemoryType | None) -> str:
        raw = f"{query}|{top_k}|{memory_type}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, top_k: int, memory_type: MemoryType | None = None) -> list | None:
        k = self._key(query, top_k, memory_type)
        entry = self._data.get(k)
        if entry is None:
            return None
        ts, results = entry
        if time.monotonic() - ts > self._ttl:
            del self._data[k]
            return None
        return results

    def put(self, query: str, top_k: int, memory_type: MemoryType | None, results: list) -> None:
        k = self._key(query, top_k, memory_type)
        # Evict oldest if at capacity
        if len(self._data) >= self._max_size and k not in self._data:
            oldest_key = min(self._data, key=lambda x: self._data[x][0])
            del self._data[oldest_key]
        self._data[k] = (time.monotonic(), results)

    def invalidate(self) -> None:
        self._data.clear()

    @property
    def size(self) -> int:
        return len(self._data)
