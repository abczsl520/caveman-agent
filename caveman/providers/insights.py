"""Usage tracking and cost estimation.

Inspired by Hermes InsightsEngine (MIT, Nous Research).
Simplified for Caveman — tracks tokens, costs, and session stats.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


# Cost per 1M tokens (USD) — approximate, update as needed
_PRICING = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


@dataclass
class TurnUsage:
    """Usage for a single API turn."""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ms: float = 0.0
    routed: bool = False  # True if cheap model was used

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        pricing = _PRICING.get(self.model, {"input": 5.0, "output": 15.0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class SessionInsights:
    """Aggregated usage for a session."""
    turns: list[TurnUsage] = field(default_factory=list)
    session_start: float = field(default_factory=time.time)

    def record(self, usage: TurnUsage) -> None:
        self.turns.append(usage)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(t.estimated_cost_usd for t in self.turns)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def routed_turns(self) -> int:
        return sum(1 for t in self.turns if t.routed)

    @property
    def avg_tokens_per_turn(self) -> float:
        return self.total_tokens / self.turn_count if self.turns else 0.0

    @property
    def session_duration_seconds(self) -> float:
        return time.time() - self.session_start

    def format_summary(self) -> str:
        """Format usage summary for display."""
        duration = self.session_duration_seconds
        if duration < 60:
            dur_str = f"{int(duration)}s"
        elif duration < 3600:
            dur_str = f"{int(duration // 60)}m {int(duration % 60)}s"
        else:
            dur_str = f"{int(duration // 3600)}h {int((duration % 3600) // 60)}m"

        lines = [
            f"Session Usage ({dur_str}, {self.turn_count} turns):",
            f"  Input:  {self.total_input_tokens:,} tokens",
            f"  Output: {self.total_output_tokens:,} tokens",
            f"  Total:  {self.total_tokens:,} tokens",
            f"  Cost:   ~${self.total_cost_usd:.4f}",
        ]

        if self.routed_turns > 0:
            lines.append(
                f"  Routed: {self.routed_turns}/{self.turn_count} turns "
                f"used cheap model"
            )

        # Per-model breakdown
        models: dict[str, list[TurnUsage]] = {}
        for t in self.turns:
            models.setdefault(t.model, []).append(t)

        if len(models) > 1:
            lines.append("  Models:")
            for model, turns in sorted(models.items()):
                total = sum(t.total_tokens for t in turns)
                cost = sum(t.estimated_cost_usd for t in turns)
                lines.append(f"    {model}: {len(turns)} turns, {total:,} tokens, ~${cost:.4f}")

        return "\n".join(lines)
