"""Tool result persistence — preserves large outputs instead of truncating.

Inspired by Hermes tool_result_storage.py (MIT, Nous Research).

Three-layer defense against context overflow:
1. Per-tool output cap (inside each tool)
2. Per-result persistence (this module) — large outputs saved to file
3. Per-turn aggregate budget — total tool results capped per turn

Design principle: metadata belongs in metadata fields, not in message content.
Tool results are stored as-is; previews reference the full file path.
"""
from __future__ import annotations

import logging

from caveman.paths import CAVEMAN_HOME

__all__ = [
    "STORAGE_DIR",
    "PERSISTED_TAG",
    "PERSISTED_END",
    "DEFAULT_THRESHOLD",
    "DEFAULT_PREVIEW_SIZE",
    "DEFAULT_TURN_BUDGET",
    "MAX_CONTEXT_SHARE",
    "persist_tool_result",
    "enforce_turn_budget",
    "cleanup_old_results",
]


logger = logging.getLogger("caveman.tools.storage")

STORAGE_DIR = CAVEMAN_HOME / "tool_results"
PERSISTED_TAG = "<persisted-output>"
PERSISTED_END = "</persisted-output>"

# Defaults
DEFAULT_THRESHOLD = 30_000  # chars — persist if larger
DEFAULT_PREVIEW_SIZE = 4_000  # chars — preview in context
DEFAULT_TURN_BUDGET = 200_000  # chars — total per turn
# OpenClaw-inspired: single tool result should not exceed 30% of context window
MAX_CONTEXT_SHARE = 0.3


def _generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    threshold: int = DEFAULT_THRESHOLD,
    preview_size: int = DEFAULT_PREVIEW_SIZE,
    context_budget: int | None = None,
) -> str:
    """Persist large tool result to file, return preview + path reference.

    If content is under threshold, returns it unchanged.
    Otherwise writes full content to STORAGE_DIR and returns a preview.

    Args:
        context_budget: If provided, dynamically cap at 30% of this value.
    """
    # Dynamic threshold: min of static threshold and 30% of context budget
    effective_threshold = threshold
    if context_budget:
        dynamic_cap = int(context_budget * MAX_CONTEXT_SHARE)
        effective_threshold = min(threshold, dynamic_cap)

    if len(content) <= effective_threshold:
        return content

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    file_path = STORAGE_DIR / f"{tool_use_id}.txt"

    try:
        file_path.write_text(content, encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to persist tool result %s: %s", tool_use_id, e)
        # Fallback: inline truncation
        return content[:threshold] + f"\n... (truncated, {len(content)} chars total)"

    preview, has_more = _generate_preview(content, preview_size)
    size_kb = len(content) / 1024
    size_str = f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_TAG}\n"
    msg += f"Tool result too large ({len(content):,} chars, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use file_read tool to access specific sections.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_END}"

    logger.info("Persisted tool result: %s (%s, %d chars -> %s)",
                tool_name, tool_use_id, len(content), file_path)
    return msg


def enforce_turn_budget(
    tool_results: list[dict],
    budget: int = DEFAULT_TURN_BUDGET,
) -> list[dict]:
    """Enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, persist the largest non-persisted results
    until under budget. Mutates in-place and returns the list.
    """
    candidates = []
    total_size = 0
    for i, r in enumerate(tool_results):
        content = r.get("content", "")
        if isinstance(content, list):
            size = sum(len(str(b)) for b in content)
        else:
            size = len(str(content))
        total_size += size
        if PERSISTED_TAG not in str(content):
            candidates.append((i, size))

    if total_size <= budget:
        return tool_results

    candidates.sort(key=lambda x: x[1], reverse=True)
    for idx, size in candidates:
        if total_size <= budget:
            break
        r = tool_results[idx]
        content = r.get("content", "")
        if isinstance(content, str):
            replacement = persist_tool_result(
                content, "__budget__", r.get("tool_use_id", f"budget_{idx}"),
                threshold=0,
            )
            if replacement != content:
                total_size -= size
                total_size += len(replacement)
                tool_results[idx]["content"] = replacement
                logger.info("Budget enforcement: persisted result %d (%d chars)", idx, size)

    return tool_results


def cleanup_old_results(max_age_hours: int = 24) -> int:
    """Remove tool result files older than max_age_hours. Returns count removed."""
    import time
    if not STORAGE_DIR.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for f in STORAGE_DIR.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            removed += 1
    return removed
