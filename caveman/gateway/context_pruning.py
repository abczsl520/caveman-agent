"""Context Pruning — cache management and context window optimization.

Manages conversation context to fit within model context windows,
with intelligent pruning strategies and caching.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "PruningConfig",
    "MessageImportance",
    "ContextPruner",
    "ContextCache",
]


logger = logging.getLogger("caveman.gateway.context_pruning")


@dataclass
class PruningConfig:
    """Configuration for context pruning."""
    max_tokens: int = 128000
    target_ratio: float = 0.75  # Prune to this ratio of max
    preserve_system: bool = True
    preserve_recent: int = 10
    cache_ttl: int = 3600  # 1 hour
    strategy: str = "smart"  # smart | fifo | importance


@dataclass
class MessageImportance:
    """Importance score for a message."""
    index: int
    role: str
    tokens: int = 0
    importance: float = 0.5
    has_tool_use: bool = False
    has_tool_result: bool = False
    is_system: bool = False
    is_recent: bool = False


class ContextPruner:
    """Prunes conversation context to fit model limits."""

    def __init__(self, config: Optional[PruningConfig] = None):
        self._config = config or PruningConfig()

    def prune(
        self,
        messages: List[Dict[str, Any]],
        model: str = "",
        current_tokens: int = 0,
    ) -> List[Dict[str, Any]]:
        """Prune messages to fit within context window."""
        target = int(self._config.max_tokens * self._config.target_ratio)

        if current_tokens <= target:
            return messages

        if self._config.strategy == "fifo":
            return self._prune_fifo(messages, target)
        elif self._config.strategy == "importance":
            return self._prune_by_importance(messages, target)
        else:
            return self._prune_smart(messages, target)

    def _prune_smart(
        self, messages: List[Dict[str, Any]], target: int,
    ) -> List[Dict[str, Any]]:
        """Smart pruning: preserve system, recent, and tool pairs."""
        scored = self._score_messages(messages)

        # Always keep system messages and recent messages
        keep_indices = set()
        for s in scored:
            if s.is_system or s.is_recent:
                keep_indices.add(s.index)

        # Keep tool use/result pairs together
        for i, s in enumerate(scored):
            if s.has_tool_use:
                keep_indices.add(s.index)
                # Keep the next message (tool result)
                if i + 1 < len(scored):
                    keep_indices.add(scored[i + 1].index)
            if s.has_tool_result:
                keep_indices.add(s.index)
                # Keep the previous message (tool use)
                if i > 0:
                    keep_indices.add(scored[i - 1].index)

        # Sort remaining by importance, drop lowest until under target
        removable = [s for s in scored if s.index not in keep_indices]
        removable.sort(key=lambda s: s.importance)

        total_tokens = sum(s.tokens for s in scored)
        for s in removable:
            if total_tokens <= target:
                break
            total_tokens -= s.tokens
            # Don't add to keep_indices (will be removed)
        else:
            # Still over target — remove from keep set (except system/recent)
            pass

        removed_indices = {s.index for s in removable if total_tokens > target}
        return [
            msg for i, msg in enumerate(messages)
            if i not in removed_indices
        ]

    def _prune_fifo(
        self, messages: List[Dict[str, Any]], target: int,
    ) -> List[Dict[str, Any]]:
        """FIFO pruning: remove oldest messages first."""
        # Keep system messages
        system = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        # Keep recent
        preserve = self._config.preserve_recent
        if len(non_system) <= preserve:
            return messages

        recent = non_system[-preserve:]
        return system + recent

    def _prune_by_importance(
        self, messages: List[Dict[str, Any]], target: int,
    ) -> List[Dict[str, Any]]:
        """Importance-based pruning."""
        scored = self._score_messages(messages)
        scored.sort(key=lambda s: -s.importance)

        total = 0
        keep = set()
        for s in scored:
            if total + s.tokens <= target:
                keep.add(s.index)
                total += s.tokens

        # Preserve order
        return [msg for i, msg in enumerate(messages) if i in keep]

    def _score_messages(self, messages: List[Dict[str, Any]]) -> List[MessageImportance]:
        """Score messages by importance."""
        total = len(messages)
        preserve = self._config.preserve_recent
        scored = []

        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            tokens = len(str(content)) // 4  # Rough estimate

            importance = 0.5
            is_system = role == "system"
            is_recent = i >= total - preserve
            has_tool_use = False
            has_tool_result = False

            if is_system:
                importance = 1.0
            elif is_recent:
                importance = 0.9
            elif role == "assistant":
                importance = 0.6
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            has_tool_use = True
                            importance = 0.7
            elif role == "user":
                importance = 0.5
            elif role == "tool":
                has_tool_result = True
                importance = 0.65

            # Recency bonus
            importance += (i / total) * 0.1

            scored.append(MessageImportance(
                index=i, role=role, tokens=tokens,
                importance=importance,
                has_tool_use=has_tool_use,
                has_tool_result=has_tool_result,
                is_system=is_system,
                is_recent=is_recent,
            ))

        return scored


# ── Context Cache ──

class ContextCache:
    """Caches processed context to avoid re-computation."""

    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._cache[key]
            return None
        return entry["messages"]

    def set(self, key: str, messages: List[Dict[str, Any]]) -> None:
        self._cache[key] = {"messages": messages, "ts": time.time()}

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count

    @staticmethod
    def cache_key(session_key: str, message_count: int) -> str:
        return hashlib.sha256(f"{session_key}:{message_count}".encode()).hexdigest()[:16]
