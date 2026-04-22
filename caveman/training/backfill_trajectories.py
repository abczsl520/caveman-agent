"""Backfill trajectory metadata — recount tool_calls from conversation data.

Fixes trajectories saved before the tools_exec.py recording fix (commit 69f24d3).
Those trajectories have tool_calls=0 because the recorder was checking for
role=="function_call" but tools_exec was recording role=="tool".

This script re-scans each trajectory's conversations and recounts:
  - function_call turns → tool_calls
  - Recalculates quality_score with accurate tool_calls

Usage:
    python -m caveman.training.backfill_trajectories [--dry-run]
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from caveman.paths import TRAJECTORIES_DIR

logger = logging.getLogger(__name__)

__all__ = ["backfill_tool_calls", "BackfillStats"]


class BackfillStats:
    def __init__(self) -> None:
        self.total = 0
        self.updated = 0
        self.skipped = 0
        self.errors = 0

    def summary(self) -> str:
        return (
            f"Backfill: {self.updated}/{self.total} updated, "
            f"{self.skipped} skipped, {self.errors} errors"
        )


def _recount_tool_calls(conversations: list[dict]) -> int:
    """Count function_call turns in a conversation."""
    return sum(
        1 for turn in conversations
        if turn.get("from") == "function_call"
        or turn.get("role") == "function_call"
    )


def _recalc_quality(conversations: list[dict], tool_calls: int, errors: int = 0) -> float:
    """Recalculate quality score with accurate tool_calls."""
    if not conversations:
        return 0.0

    score = 0.5
    if tool_calls > 0:
        score += 0.15
    if len(conversations) >= 4:
        score += 0.1
    if conversations and conversations[-1].get("from") == "gpt":
        last_val = conversations[-1].get("value", "")
        if len(last_val) > 20:
            score += 0.15
    if errors > 0:
        error_ratio = errors / max(len(conversations), 1)
        score -= min(error_ratio * 0.5, 0.3)
    if len(conversations) <= 2:
        score -= 0.1

    return max(0.0, min(1.0, score))


def backfill_tool_calls(
    trajectories_dir: Path | str | None = None,
    dry_run: bool = False,
) -> BackfillStats:
    """Recount tool_calls in all trajectories and update metadata.

    Args:
        trajectories_dir: Directory containing trajectory JSON files.
        dry_run: If True, don't write changes.

    Returns:
        BackfillStats with update statistics.
    """
    traj_dir = Path(trajectories_dir or TRAJECTORIES_DIR)
    stats = BackfillStats()

    for path in sorted(traj_dir.glob("*.json")):
        stats.total += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                stats.skipped += 1
                continue

            convs = data.get("conversations", [])
            meta = data.get("metadata", {})

            old_tool_calls = meta.get("tool_calls", 0)
            new_tool_calls = _recount_tool_calls(convs)

            # Also check for "tool" role turns (old format before fix)
            tool_role_count = sum(
                1 for turn in convs
                if turn.get("from") == "tool" or turn.get("role") == "tool"
            )
            # Use the higher of the two counts
            actual_tool_calls = max(new_tool_calls, tool_role_count)

            if actual_tool_calls == old_tool_calls:
                stats.skipped += 1
                continue

            # Update metadata
            meta["tool_calls"] = actual_tool_calls
            meta["tool_calls_backfilled"] = True
            meta["tool_calls_old"] = old_tool_calls

            # Recalculate quality score
            errors = meta.get("errors", 0)
            meta["quality_score"] = _recalc_quality(convs, actual_tool_calls, errors)

            data["metadata"] = meta

            if not dry_run:
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            stats.updated += 1

        except Exception as e:
            stats.errors += 1
            logger.debug("Backfill error for %s: %s", path.name, e)

    logger.info(stats.summary())
    return stats


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    dry = "--dry-run" in sys.argv
    result = backfill_tool_calls(dry_run=dry)
    print(result.summary())
