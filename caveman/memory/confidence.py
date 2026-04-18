"""Memory confidence scoring — memories get more accurate with use.

NOTE (Round 107): This in-memory tracker is superseded by SQLite's
trust_score column + mark_helpful() in sqlite_store.py. The real
confidence loop runs through:
  - phase_finalize → mark_helpful(success/failure)
  - Reflect → mark_helpful(True) on success
  - Lint → mark_helpful(False) on stale/contradiction
This module is kept for backward compatibility and tests only.

Tracks how reliable each memory is based on usage feedback:
- Used by Recall + task succeeded → confidence increases
- Used by Recall + task failed → confidence decreases
- Confirmed by Reflect → confidence boost
- Flagged by Lint as stale → confidence drops

Confidence affects Recall ranking: high-confidence memories surface first.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Score deltas
RECALL_SUCCESS = 0.10
RECALL_FAILURE = -0.15
REFLECT_CONFIRM = 0.20
LINT_STALE = -0.30
HELPFUL_FEEDBACK = 0.05
UNHELPFUL_FEEDBACK = -0.10

# Bounds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
DEFAULT_CONFIDENCE = 0.5


@dataclass
class ConfidenceRecord:
    """Confidence metadata for a memory."""
    memory_id: str
    score: float = DEFAULT_CONFIDENCE
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used_at: float = 0.0
    last_updated_at: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def _clamp(self) -> None:
        self.score = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, self.score))
        self.last_updated_at = time.time()


class ConfidenceTracker:
    """Tracks confidence scores for memories."""

    def __init__(self) -> None:
        self._records: dict[str, ConfidenceRecord] = {}

    def get(self, memory_id: str) -> ConfidenceRecord:
        if memory_id not in self._records:
            self._records[memory_id] = ConfidenceRecord(memory_id=memory_id)
        return self._records[memory_id]

    def record_use(self, memory_id: str, success: bool) -> float:
        """Record that a memory was used in Recall. Returns new score."""
        rec = self.get(memory_id)
        rec.use_count += 1
        rec.last_used_at = time.time()

        if success:
            rec.success_count += 1
            rec.score += RECALL_SUCCESS
        else:
            rec.failure_count += 1
            rec.score += RECALL_FAILURE

        rec._clamp()
        return rec.score

    def record_reflect_confirm(self, memory_id: str) -> float:
        """Reflect Engine confirmed this memory is accurate."""
        rec = self.get(memory_id)
        rec.score += REFLECT_CONFIRM
        rec._clamp()
        return rec.score

    def record_lint_stale(self, memory_id: str) -> float:
        """Lint Engine flagged this memory as stale."""
        rec = self.get(memory_id)
        rec.score += LINT_STALE
        rec._clamp()
        return rec.score

    def record_feedback(self, memory_id: str, helpful: bool) -> float:
        """User feedback on memory usefulness."""
        rec = self.get(memory_id)
        delta = HELPFUL_FEEDBACK if helpful else UNHELPFUL_FEEDBACK
        rec.score += delta
        rec._clamp()
        return rec.score

    def rank_memories(
        self, memory_ids: list[str],
        relevance_scores: Optional[list[float]] = None,
        confidence_weight: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Rank memories by combined relevance + confidence.

        Args:
            memory_ids: Memory IDs to rank.
            relevance_scores: Optional relevance scores (0-1) from retrieval.
            confidence_weight: How much confidence affects ranking (0-1).

        Returns:
            List of (memory_id, combined_score) sorted descending.
        """
        if relevance_scores is None:
            relevance_scores = [0.5] * len(memory_ids)

        ranked = []
        for mid, rel in zip(memory_ids, relevance_scores):
            conf = self.get(mid).score
            combined = (1 - confidence_weight) * rel + confidence_weight * conf
            ranked.append((mid, combined))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        if not self._records:
            return {"count": 0, "avg_confidence": 0.0}

        scores = [r.score for r in self._records.values()]
        return {
            "count": len(scores),
            "avg_confidence": sum(scores) / len(scores),
            "min_confidence": min(scores),
            "max_confidence": max(scores),
            "total_uses": sum(r.use_count for r in self._records.values()),
        }
