"""Grounding Gate — verify recalled memories against reality before use.

Anti-hallucination layer: memories claim things exist, but reality changes.
"Memory says X exists ≠ X exists now" (AGENTS.md Iron Law).

This gate runs AFTER recall, BEFORE prompt injection. It:
  - Checks file paths mentioned in memories → do they still exist?
  - Flags memories with stale/archived metadata → demote in ranking
  - Adds [UNVERIFIED] tag to memories that reference external state
  - Tracks grounding hit rate for flywheel metrics

Design: Zero LLM cost. Pure heuristic + filesystem checks.
Performance: <10ms for 10 memories (stat calls are fast).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryEntry

logger = logging.getLogger(__name__)

# Patterns that reference verifiable external state
_PATH_PATTERN = re.compile(r'(?:/[\w./-]+){2,}')
_IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_VERSION_PATTERN = re.compile(r'\bv?\d+\.\d+(?:\.\d+)?\b')


def ground_memories(
    memories: list[MemoryEntry],
    check_paths: bool = True,
    workspace_root: Path | None = None,
) -> list[tuple[MemoryEntry, dict[str, Any]]]:
    """Verify recalled memories against reality.

    Returns list of (memory, grounding_info) where grounding_info contains:
      - verified: bool — all verifiable claims checked out
      - stale_signals: list[str] — what looks stale
      - confidence_modifier: float — multiply with trust score
    """
    results: list[tuple[MemoryEntry, dict[str, Any]]] = []

    for mem in memories:
        info = _ground_single(mem, check_paths, workspace_root)
        results.append((mem, info))

    grounded = sum(1 for _, info in results if info["verified"])
    total = len(results)
    if total > 0:
        logger.debug(
            "Grounding: %d/%d memories verified (%.0f%%)",
            grounded, total, grounded / total * 100,
        )

    return results


def _ground_single(
    mem: MemoryEntry,
    check_paths: bool,
    workspace_root: Path | None,
) -> dict[str, Any]:
    """Ground a single memory entry."""
    stale_signals: list[str] = []
    confidence_modifier = 1.0

    # Check 1: Already marked as stale/archived/superseded
    meta = mem.metadata or {}
    if meta.get("archived"):
        stale_signals.append("archived")
        confidence_modifier *= 0.1
    if meta.get("superseded_by"):
        stale_signals.append(f"superseded_by: {str(meta['superseded_by'])[:60]}")
        confidence_modifier *= 0.2
    if meta.get("needs_review"):
        stale_signals.append("needs_review")
        confidence_modifier *= 0.5

    # Check 2: File paths — do they still exist?
    if check_paths:
        paths = _PATH_PATTERN.findall(mem.content)
        for p in paths[:5]:  # Cap to avoid slow stat storms
            path = Path(p)
            # Only check paths that look like they're in the workspace
            if workspace_root and not str(path).startswith(str(workspace_root)):
                # Try relative to workspace
                rel = workspace_root / p.lstrip("/")
                if rel.exists():
                    continue
            if path.is_absolute() and not path.exists():
                stale_signals.append(f"missing_path: {p}")
                confidence_modifier *= 0.6

    # Check 3: Low trust score from SQLite
    trust = meta.get("trust_score", 0.5)
    if trust < 0.3:
        stale_signals.append(f"low_trust: {trust:.2f}")
        confidence_modifier *= 0.5

    # Check 4: Nudge-extracted without confirmation
    if meta.get("source") == "nudge" and meta.get("confirmed_count", 0) == 0:
        # Unconfirmed nudge extractions are less reliable
        confidence_modifier *= 0.8

    verified = len(stale_signals) == 0
    return {
        "verified": verified,
        "stale_signals": stale_signals,
        "confidence_modifier": max(0.05, confidence_modifier),
    }


def annotate_for_prompt(
    grounded: list[tuple[MemoryEntry, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Convert grounded memories to prompt-ready format.

    Adds [UNVERIFIED] prefix to memories with stale signals,
    so the LLM knows to treat them with skepticism.
    """
    result: list[dict[str, Any]] = []
    for mem, info in grounded:
        content = mem.content
        if not info["verified"]:
            signals = ", ".join(info["stale_signals"][:3])
            content = f"[UNVERIFIED: {signals}] {content}"

        age_days = 0
        if hasattr(mem, "created_at"):
            from datetime import datetime
            age_days = (datetime.now() - mem.created_at).days

        result.append({
            "content": content,
            "type": mem.memory_type.value,
            "age_days": age_days,
            "grounding": info,
        })
    return result


def filter_grounded(
    grounded: list[tuple[MemoryEntry, dict[str, Any]]],
    min_confidence: float = 0.1,
) -> list[tuple[MemoryEntry, dict[str, Any]]]:
    """Remove memories below minimum confidence after grounding."""
    return [
        (mem, info) for mem, info in grounded
        if info["confidence_modifier"] >= min_confidence
    ]
