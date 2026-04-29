# Internal helpers for flywheel dashboard; not public API.
from __future__ import annotations

from typing import Any


def _optional_number(value: Any) -> float | None:
    """Parse a numeric telemetry field; return None for malformed values."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number_value(value: Any, default: float = 0.0) -> float:
    """Parse a numeric telemetry field; fallback instead of crashing dashboard."""
    parsed = _optional_number(value)
    return default if parsed is None else parsed


def _count_value(value: Any, default: int = 0) -> int:
    """Parse a non-negative integer telemetry counter."""
    parsed = _number_value(value, float(default))
    if parsed < 0:
        return default
    return int(parsed)
