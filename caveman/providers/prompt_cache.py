"""Anthropic prompt caching — system_and_3 strategy.

Ported from Hermes prompt_caching.py (MIT, Nous Research).
Reduces input token costs ~75% by caching conversation prefix.
Uses 4 cache_control breakpoints (Anthropic max):
  1. System prompt (stable across turns)
  2-4. Last 3 non-system messages (rolling window)
"""
from __future__ import annotations

import copy
from typing import Any


def _apply_cache_marker(
    msg: dict, marker: dict, native: bool = False,
) -> None:
    """Add cache_control to a message, handling format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native:
            msg["cache_control"] = marker
        return

    if content is None or content == "":
        msg["cache_control"] = marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = marker


def apply_cache_control(
    messages: list[dict[str, Any]],
    cache_ttl: str = "5m",
    native: bool = False,
) -> list[dict[str, Any]]:
    """Apply system_and_3 caching to messages for Anthropic.

    Places up to 4 cache_control breakpoints:
    system prompt + last 3 non-system messages.

    Returns deep copy with cache_control injected.
    """
    result = copy.deepcopy(messages)
    if not result:
        return result

    marker: dict[str, str] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    used = 0

    # System prompt gets first breakpoint
    if result[0].get("role") == "system":
        _apply_cache_marker(result[0], marker, native=native)
        used += 1

    # Last 3 non-system messages get remaining breakpoints
    remaining = 4 - used
    non_sys = [i for i, m in enumerate(result) if m.get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(result[idx], marker, native=native)

    return result
