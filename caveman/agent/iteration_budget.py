"""Iteration budget — thread-safe iteration counter with refund support.

Ported from Hermes IterationBudget (MIT, Nous Research), adapted for Caveman.

Each agent (parent or sub-agent) gets its own budget.
- Parent: capped at config `agent.max_iterations` (default 90)
- Sub-agent: independent budget from `delegation.max_iterations` (default 50)
- `execute_code` / programmatic tool calls can refund iterations
"""
from __future__ import annotations
import threading
import logging

logger = logging.getLogger(__name__)


class IterationBudget:
    """Thread-safe iteration counter for an agent."""

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration. Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self, reason: str = "") -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1
                if reason:
                    logger.debug("Iteration refunded: %s (used=%d)", reason, self._used)

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._used >= self.max_total

    def __repr__(self) -> str:
        return f"IterationBudget(used={self._used}/{self.max_total})"
