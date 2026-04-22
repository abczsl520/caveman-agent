"""Flywheel health metrics — quantify whether the flywheel is turning.

Key metrics:
  - trust_distribution: histogram of trust scores (healthy = bell curve, not flat)
  - feedback_rate: % of recalled memories that got trust feedback
  - recall_hit_rate: % of recalls that returned results
  - decay_balance: ratio of trust increases vs decreases
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
            f"feedback rate={self.feedback_rate:.0%}"
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
            # Support both sync (all_entries) and async (list_all) backends
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

            # Recall tracking
            retrieval_count = meta.get("retrieval_count", 0)
            if retrieval_count == 0:
                never_recalled += 1
            if meta.get("trust_score") is not None:
                with_feedback += 1

        health.trust_distribution = buckets
        health.avg_trust = trust_sum / len(all_memories) if all_memories else 0.0
        health.memories_never_recalled = never_recalled
        health.memories_with_feedback = with_feedback
        health.feedback_rate = with_feedback / len(all_memories) if all_memories else 0.0

        # Diagnostics
        if health.avg_trust < 0.3:
            health.issues.append(f"Low average trust ({health.avg_trust:.2f}) — memories may be unreliable")
        if health.feedback_rate < 0.1:
            health.issues.append(f"Low feedback rate ({health.feedback_rate:.0%}) — flywheel not learning")
        never_pct = never_recalled / len(all_memories) if all_memories else 0
        if never_pct > 0.8:
            health.issues.append(f"{never_pct:.0%} memories never recalled — retrieval may be broken")

        return health

