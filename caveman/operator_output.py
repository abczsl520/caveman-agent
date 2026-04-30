"""Operator-facing literal formatting helpers."""
from __future__ import annotations


def operator_literal(value: object, max_length: int | None = None) -> str:
    """Return repr() for operator output so control characters stay escaped."""
    if max_length is not None:
        if not isinstance(max_length, int) or isinstance(max_length, bool):
            raise TypeError("max_length must be an int")
        if max_length < 1:
            raise ValueError("max_length must be positive")
    text = str(value if value is not None else "<missing>")
    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 1] + "…"
    return repr(text)
