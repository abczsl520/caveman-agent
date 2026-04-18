"""Usage tracking and cost estimation.

Ported from Hermes agent/usage_pricing.py (632 lines → ~200 lines).
Tracks per-session token usage and estimates costs using model metadata.
"""
from __future__ import annotations

__all__ = ["UsageTracker", "SessionUsage", "TurnUsage"]

import logging
from dataclasses import dataclass, field
from typing import Any

from caveman.providers.model_metadata import ModelInfo, get_model_info

logger = logging.getLogger(__name__)


@dataclass
class TurnUsage:
    """Token usage for a single LLM call."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class SessionUsage:
    """Aggregated usage for an entire session."""

    session_id: str = ""
    turns: list[TurnUsage] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return sum(t.total_tokens for t in self.turns)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def format_summary(self) -> str:
        """Human-readable usage summary."""
        cost = self.total_cost_usd
        cost_str = f"${cost:.4f}" if cost > 0 else "free"
        return (
            f"{_fmt(self.total_input_tokens)} in / "
            f"{_fmt(self.total_output_tokens)} out / "
            f"{self.turn_count} turns / {cost_str}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "turn_count": self.turn_count,
        }


class UsageTracker:
    """Tracks token usage and costs across sessions.

    Usage:
        tracker = UsageTracker()
        tracker.record("session-1", "claude-opus-4-6", input_tokens=1000, output_tokens=500)
        print(tracker.get_session("session-1").format_summary())
        print(tracker.format_total())
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionUsage] = {}

    def record(
        self,
        session_id: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> TurnUsage:
        """Record a single LLM call's usage."""
        info = get_model_info(model)
        cost = info.estimate_cost(input_tokens, output_tokens)

        turn = TurnUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost,
        )

        if session_id not in self._sessions:
            self._sessions[session_id] = SessionUsage(session_id=session_id)
        self._sessions[session_id].turns.append(turn)

        return turn

    def get_session(self, session_id: str) -> SessionUsage:
        return self._sessions.get(session_id, SessionUsage(session_id=session_id))

    @property
    def all_sessions(self) -> list[SessionUsage]:
        return list(self._sessions.values())

    @property
    def total_cost_usd(self) -> float:
        return sum(s.total_cost_usd for s in self._sessions.values())

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self._sessions.values())

    def format_total(self) -> str:
        """Human-readable total across all sessions."""
        cost = self.total_cost_usd
        cost_str = f"${cost:.4f}" if cost > 0 else "free"
        return (
            f"{_fmt(self.total_tokens)} tokens / "
            f"{len(self._sessions)} sessions / {cost_str}"
        )


def _fmt(n: int) -> str:
    """Format large numbers: 7999856 → '8.0M', 33599 → '33.6K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
