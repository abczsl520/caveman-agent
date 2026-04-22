"""Budget configuration for tool result persistence.

Controls when tool results are persisted to disk vs kept inline.
Three layers:
- Per-result: threshold in chars (tool-specific overrides)
- Per-turn: aggregate char budget across all results in one turn
- Preview: inline snippet size after persistence
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_RESULT_SIZE_CHARS",
    "DEFAULT_TURN_BUDGET_CHARS",
    "DEFAULT_PREVIEW_SIZE_CHARS",
    "BudgetConfig",
]


# Tools whose thresholds must never be overridden
PINNED_THRESHOLDS: dict[str, float] = {
    "read_file": float("inf"),  # Prevent persist→read→persist loops
    "file_read": float("inf"),
}

DEFAULT_RESULT_SIZE_CHARS = 100_000
DEFAULT_TURN_BUDGET_CHARS = 200_000
DEFAULT_PREVIEW_SIZE_CHARS = 1_500


@dataclass(frozen=True)
class BudgetConfig:
    """Immutable budget constants for tool result persistence."""

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    tool_overrides: dict[str, int] = field(default_factory=dict)

    def resolve_threshold(self, tool_name: str) -> int | float:
        """Resolve persistence threshold for a tool."""
        if tool_name in PINNED_THRESHOLDS:
            return PINNED_THRESHOLDS[tool_name]
        return self.tool_overrides.get(tool_name, self.default_result_size)

    def should_persist(self, tool_name: str, result_size: int) -> bool:
        """Check if a tool result should be persisted to disk."""
        threshold = self.resolve_threshold(tool_name)
        return result_size > threshold
