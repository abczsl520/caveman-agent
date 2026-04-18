# DEPRECATED: Not wired into runtime. See Round 38 audit.
"""Trajectory compressor — clean trajectories for training data quality."""
from __future__ import annotations
import json
import re


def compress_trajectory(turns: list[dict], max_turns: int = 50) -> list[dict]:
    """Clean and compress a trajectory for training.

    Removes:
    - Duplicate consecutive messages
    - Empty turns
    - Verbose tool output (truncated to 500 chars)
    - System/internal noise

    Preserves:
    - User intent (human turns)
    - Key decisions (gpt reasoning)
    - Tool call structure (function_call)
    - Meaningful tool results (function_response, truncated)
    """
    if not turns:
        return []

    result = []
    prev_value = ""

    for turn in turns:
        role = turn.get("from", "")
        value = turn.get("value", "")

        # Skip empty
        if not value or not value.strip():
            continue

        # Skip exact duplicates
        if value == prev_value and role == (result[-1].get("from") if result else ""):
            continue

        # Truncate verbose tool responses
        if role == "function_response" and len(value) > 500:
            value = value[:400] + f"\n...[{len(value)-400} chars truncated]"

        # Clean up tool call format
        if role == "function_call":
            try:
                call_data = json.loads(value)
                # Normalize to compact format
                value = json.dumps(call_data, ensure_ascii=False, separators=(",", ":"))
            except json.JSONDecodeError:
                pass

        # Remove ANSI escape codes
        value = re.sub(r'\x1b\[[0-9;]*m', '', value)

        result.append({"from": role, "value": value})
        prev_value = value

    # Enforce max turns
    if len(result) > max_turns:
        # Keep first 5 + last (max_turns - 5)
        result = result[:5] + result[-(max_turns - 5):]

    return result


def merge_trajectories(trajectories: list[list[dict]]) -> list[dict]:
    """Merge multiple trajectory segments into one coherent conversation."""
    merged = []
    for traj in trajectories:
        for turn in traj:
            # Avoid duplicate joins
            if merged and merged[-1].get("value") == turn.get("value"):
                continue
            merged.append(turn)
    return merged
