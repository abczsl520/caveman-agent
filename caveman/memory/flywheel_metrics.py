"""Flywheel health metrics — quantify whether the flywheel is turning.

Key metrics:
  - trust_distribution: histogram of trust scores (healthy = bell curve, not flat)
  - feedback_rate: % of memories with real helpful feedback
    (helpful_count > 0), not merely a default/passive trust score
  - recall_rate: % of memories that have been recalled at least once
  - top_recalled: heavily reused memories that dominate retrieval context
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FlywheelHealth:
    """Snapshot of flywheel health."""
    total_memories: int = 0
    trust_distribution: dict[str, int] = field(default_factory=dict)
    avg_trust: float = 0.0
    memories_never_recalled: int = 0
    memories_with_feedback: int = 0
    feedback_rate: float = 0.0
    recalled_memories: int = 0
    recall_rate: float = 0.0
    top_recalled: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        status = "✅ healthy" if self.is_healthy else f"⚠️ {len(self.issues)} issues"
        return (
            f"Flywheel {status}: {self.total_memories} memories, "
            f"avg trust={self.avg_trust:.2f}, "
            f"feedback rate={self.feedback_rate:.0%}, "
            f"recall rate={self.recall_rate:.0%}"
        )

    @classmethod
    async def diagnose(cls, backend) -> "FlywheelHealth":
        """Build a FlywheelHealth snapshot from a MemoryManager backend.

        Args:
            backend: MemoryManager instance with .search() and .list_all() support.

        Returns:
            FlywheelHealth with populated metrics and issue diagnostics.
        """
        health = cls()
        try:
            # Support both sync (all_entries) and async (list_all) backends.
            # SQLiteMemoryStore exposes live trust/retrieval counters through
            # MemoryEntry.metadata, so reading all entries is enough for
            # diagnostics without mutating production data.
            if hasattr(backend, 'all_entries'):
                entries = backend.all_entries
                all_memories = entries() if callable(entries) else entries
            elif hasattr(backend, 'list_all'):
                all_memories = await backend.list_all()
            else:
                all_memories = []
        except Exception:
            all_memories = []

        health.total_memories = len(all_memories)
        if not all_memories:
            health.issues.append("No memories stored — flywheel not started")
            return health

        # Trust distribution buckets
        buckets = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        trust_sum = 0.0
        never_recalled = 0
        with_feedback = 0

        for mem in all_memories:
            meta = getattr(mem, "metadata", {}) or {}
            trust = meta.get("trust_score", getattr(mem, "trust_score", 0.5))
            trust_sum += trust

            # Bucket
            if trust < 0.2:
                buckets["0.0-0.2"] += 1
            elif trust < 0.4:
                buckets["0.2-0.4"] += 1
            elif trust < 0.6:
                buckets["0.4-0.6"] += 1
            elif trust < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

            # Feedback tracking: helpful_count is a real judge/user feedback
            # signal. A trust_score alone only means the row has a default score
            # (or passive recall boost), not that the flywheel learned whether
            # the memory helped. Older backends may only expose these counters
            # through metadata, so keep tolerant fallbacks.
            retrieval_count = int(meta.get("retrieval_count", getattr(mem, "retrieval_count", 0)) or 0)
            helpful_count = int(meta.get("helpful_count", getattr(mem, "helpful_count", 0)) or 0)
            if retrieval_count == 0:
                never_recalled += 1
            if helpful_count > 0:
                with_feedback += 1

        health.trust_distribution = buckets
        health.avg_trust = trust_sum / len(all_memories) if all_memories else 0.0
        health.memories_never_recalled = never_recalled
        health.memories_with_feedback = with_feedback
        health.feedback_rate = with_feedback / len(all_memories) if all_memories else 0.0
        health.recalled_memories = len(all_memories) - never_recalled
        health.recall_rate = health.recalled_memories / len(all_memories) if all_memories else 0.0

        top_scored: list[tuple[int, float, Any]] = []
        for mem in all_memories:
            meta = getattr(mem, "metadata", {}) or {}
            retrieval_count = int(meta.get("retrieval_count", getattr(mem, "retrieval_count", 0)) or 0)
            if retrieval_count <= 0:
                continue
            trust = float(meta.get("trust_score", getattr(mem, "trust_score", 0.5)) or 0.0)
            top_scored.append((retrieval_count, trust, mem))
        top_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        health.top_recalled = [
            {
                "id": getattr(mem, "id", ""),
                "type": getattr(getattr(mem, "memory_type", ""), "value", str(getattr(mem, "memory_type", ""))),
                "retrieval_count": retrieval_count,
                "trust_score": trust,
                "preview": str(getattr(mem, "content", ""))[:120],
            }
            for retrieval_count, trust, mem in top_scored[:10]
        ]

        # Diagnostics
        if health.avg_trust < 0.3:
            health.issues.append(f"Low average trust ({health.avg_trust:.2f}) — memories may be unreliable")
        if health.total_memories >= 10 and health.feedback_rate < 0.1:
            health.issues.append(f"Low feedback rate ({health.feedback_rate:.0%}) — flywheel not learning")
        never_pct = never_recalled / len(all_memories) if all_memories else 0
        if health.total_memories >= 10 and never_pct > 0.8:
            health.issues.append(f"{never_pct:.0%} memories never recalled — retrieval may be broken")

        return health

