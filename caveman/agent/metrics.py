"""Agent performance metrics — track timings, counters, and percentiles."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class AgentMetrics:
    """Track agent performance metrics with timing percentiles."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)

    def record_timing(self, name: str, duration_s: float) -> None:
        """Record a timing measurement."""
        self._timings[name].append(duration_s)

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] += amount

    @contextmanager
    def timer(self, name: str):
        """Context manager that records elapsed time."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.record_timing(name, time.monotonic() - start)

    def summary(self) -> dict[str, Any]:
        """Return summary with avg, p50, p95, p99 for each timing."""
        result: dict[str, Any] = {"counters": dict(self._counters), "timings": {}}
        for name, values in self._timings.items():
            if not values:
                continue
            s = sorted(values)
            n = len(s)
            result["timings"][name] = {
                "count": n,
                "avg": sum(s) / n,
                "p50": s[n // 2],
                "p95": s[int(n * 0.95)] if n >= 2 else s[-1],
                "p99": s[int(n * 0.99)] if n >= 2 else s[-1],
                "min": s[0],
                "max": s[-1],
            }
        return result

    def reset(self) -> None:
        """Clear all metrics."""
        self._timings.clear()
        self._counters.clear()

    def flywheel_health(self) -> dict[str, Any]:
        """Flywheel acceleration metrics (PRD §5.5).

        These measure whether the flywheel is actually accelerating:
        - recall_hit_rate: Are memories being found? (should increase over time)
        - skill_match_rate: Are skills being matched? (should increase over time)
        - task_success_rate: Are tasks succeeding? (should increase over time)
        - memory_reuse_rate: Are recalled memories being reused? (compound interest)
        """
        c = self._counters
        total_recalls = c.get("recall_attempts", 0)
        recall_hits = c.get("recall_hits", 0)
        skill_attempts = c.get("skill_match_attempts", 0)
        skill_hits = c.get("skill_match_hits", 0)
        tasks = c.get("turns_completed", 0)
        successes = c.get("task_successes", 0)

        return {
            "recall_hit_rate": recall_hits / total_recalls if total_recalls > 0 else 0.0,
            "skill_match_rate": skill_hits / skill_attempts if skill_attempts > 0 else 0.0,
            "task_success_rate": successes / tasks if tasks > 0 else 0.0,
            "total_turns": tasks,
            "total_memories_recalled": recall_hits,
            "total_skills_matched": skill_hits,
        }
