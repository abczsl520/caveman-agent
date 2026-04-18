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


def diagnose(backend: Any) -> FlywheelHealth:
    """Run flywheel health diagnosis against SQLite backend."""
    health = FlywheelHealth()

    try:
        conn = backend._get_conn()
    except Exception:
        health.issues.append("Cannot connect to memory database")
        return health

    # Total memories
    row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    health.total_memories = row[0] if row else 0

    if health.total_memories == 0:
        health.issues.append("Empty memory store — flywheel has no fuel")
        return health

    # Trust distribution
    buckets = [
        ("dead (0-0.1)", 0.0, 0.1),
        ("low (0.1-0.3)", 0.1, 0.3),
        ("default (0.3-0.6)", 0.3, 0.6),
        ("good (0.6-0.8)", 0.6, 0.8),
        ("excellent (0.8-1.0)", 0.8, 1.01),
    ]
    for label, lo, hi in buckets:
        row = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE trust_score >= ? AND trust_score < ?",
            (lo, hi),
        ).fetchone()
        health.trust_distribution[label] = row[0] if row else 0

    # Average trust
    row = conn.execute("SELECT AVG(trust_score) FROM memories").fetchone()
    health.avg_trust = row[0] if row and row[0] else 0.0

    # Never recalled
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE retrieval_count = 0"
    ).fetchone()
    health.memories_never_recalled = row[0] if row else 0

    # Feedback rate: memories with trust != 0.5 (default) / total
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE ABS(trust_score - 0.5) > 0.01"
    ).fetchone()
    health.memories_with_feedback = row[0] if row else 0
    health.feedback_rate = health.memories_with_feedback / health.total_memories

    # Top recalled
    rows = conn.execute(
        "SELECT content, retrieval_count, trust_score FROM memories "
        "ORDER BY retrieval_count DESC LIMIT 5"
    ).fetchall()
    health.top_recalled = [
        {"content": r[0][:80], "recalls": r[1], "trust": r[2]}
        for r in rows
    ]

    # Diagnose issues
    default_pct = health.trust_distribution.get("default (0.3-0.6)", 0) / health.total_memories
    if default_pct > 0.8:
        health.issues.append(
            f"{default_pct:.0%} of memories at default trust — feedback loop not working"
        )

    never_pct = health.memories_never_recalled / health.total_memories
    if never_pct > 0.7:
        health.issues.append(
            f"{never_pct:.0%} of memories never recalled — recall quality may be poor"
        )

    if health.avg_trust < 0.3:
        health.issues.append(
            f"Average trust {health.avg_trust:.2f} is very low — too much demotion"
        )

    dead = health.trust_distribution.get("dead (0-0.1)", 0)
    if dead > health.total_memories * 0.2:
        health.issues.append(
            f"{dead} dead memories (trust<0.1) — GC may not be running"
        )

    return health
