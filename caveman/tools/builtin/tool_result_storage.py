"""Tool Result Storage — persist and retrieve tool execution results.

Stores tool results for later reference, deduplication, and
caching. Extracted from Hermes tools/tool_result_storage.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["StoredResult", "ToolResultStore"]


logger = logging.getLogger("caveman.tools.result_storage")

_STORAGE_DIR = Path.home() / ".caveman" / "tool_results"


@dataclass
class StoredResult:
    """A stored tool result."""
    tool_name: str
    input_hash: str
    result: str
    stored_at: float = 0
    ttl: float = 3600  # 1 hour default
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.stored_at > self.ttl


class ToolResultStore:
    """Stores and retrieves tool execution results."""

    def __init__(self, storage_dir: Optional[Path] = None, max_entries: int = 1000):
        self._dir = storage_dir or _STORAGE_DIR
        self._cache: Dict[str, StoredResult] = {}
        self._max_entries = max_entries

    def store(
        self,
        tool_name: str,
        input_data: Any,
        result: str,
        ttl: float = 3600,
    ) -> str:
        """Store a tool result. Returns the storage key."""
        input_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False) if not isinstance(input_data, str) else input_data
        input_hash = hashlib.sha256(f"{tool_name}:{input_str}".encode()).hexdigest()[:16]

        entry = StoredResult(
            tool_name=tool_name,
            input_hash=input_hash,
            result=result[:100_000],  # Cap at 100KB
            stored_at=time.time(),
            ttl=ttl,
        )
        self._cache[input_hash] = entry

        # Evict if over limit
        if len(self._cache) > self._max_entries:
            self._evict_oldest()

        # Persist
        self._persist(entry)
        return input_hash

    def get(self, tool_name: str, input_data: Any) -> Optional[str]:
        """Get a cached tool result if available and not expired."""
        input_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False) if not isinstance(input_data, str) else input_data
        input_hash = hashlib.sha256(f"{tool_name}:{input_str}".encode()).hexdigest()[:16]

        entry = self._cache.get(input_hash)
        if not entry:
            # Try loading from disk
            entry = self._load(input_hash)
            if entry:
                self._cache[input_hash] = entry

        if not entry:
            return None
        if entry.is_expired:
            self._cache.pop(input_hash, None)
            return None

        entry.hit_count += 1
        return entry.result

    def invalidate(self, tool_name: str, input_data: Any) -> bool:
        """Invalidate a cached result."""
        input_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False) if not isinstance(input_data, str) else input_data
        input_hash = hashlib.sha256(f"{tool_name}:{input_str}".encode()).hexdigest()[:16]
        removed = input_hash in self._cache
        self._cache.pop(input_hash, None)
        # Also remove from disk
        path = self._dir / f"{input_hash}.json"
        if path.exists():
            path.unlink(missing_ok=True)
            removed = True
        return removed

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._cache),
            "total_hits": sum(e.hit_count for e in self._cache.values()),
            "expired": sum(1 for e in self._cache.values() if e.is_expired),
        }

    def _evict_oldest(self) -> None:
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].stored_at)
        del self._cache[oldest_key]

    def _persist(self, entry: StoredResult) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{entry.input_hash}.json"
        try:
            path.write_text(json.dumps({
                "tool_name": entry.tool_name,
                "input_hash": entry.input_hash,
                "result": entry.result,
                "stored_at": entry.stored_at,
                "ttl": entry.ttl,
            }, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.debug("_persist: suppressed %s", exc)

    def _load(self, input_hash: str) -> Optional[StoredResult]:
        path = self._dir / f"{input_hash}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return StoredResult(**data)
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return None
