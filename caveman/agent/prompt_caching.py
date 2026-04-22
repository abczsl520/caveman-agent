"""Prompt Caching — cache system prompts to reduce token usage.

Caches the system prompt hash and marks it for provider-level
caching (Anthropic cache_control, OpenAI cached tokens).
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger("caveman.agent.prompt_caching")


@dataclass
class CacheEntry:
    """A cached prompt entry."""
    hash: str
    content: str
    created_at: float = 0
    hit_count: int = 0
    last_hit: float = 0


class PromptCache:
    """Caches system prompts to enable provider-level caching."""

    def __init__(self, max_entries: int = 10):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_entries = max_entries

    def get_or_create(self, content: str) -> CacheEntry:
        """Get cached entry or create new one."""
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        if h in self._cache:
            entry = self._cache[h]
            entry.hit_count += 1
            entry.last_hit = time.time()
            return entry

        # Evict oldest if full
        if len(self._cache) >= self._max_entries:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].last_hit)
            del self._cache[oldest_key]

        entry = CacheEntry(hash=h, content=content, created_at=time.time())
        self._cache[h] = entry
        return entry

    def apply_cache_control(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add cache_control markers for Anthropic API.

        Marks the system message and last user message with
        ephemeral cache_control for prompt caching.
        """
        if not messages:
            return messages

        result = [m.copy() for m in messages]

        # Mark system message
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    result[i] = {
                        **msg,
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                break

        # Mark last user message with tool results
        for i in range(len(result) - 1, -1, -1):
            if result[i].get("role") == "user":
                content = result[i].get("content", "")
                if isinstance(content, str) and len(content) > 1000:
                    result[i] = {
                        **result[i],
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                break

        return result

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._cache),
            "total_hits": sum(e.hit_count for e in self._cache.values()),
        }

    def clear(self) -> None:
        self._cache.clear()
