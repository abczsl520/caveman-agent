"""Memory Drift Detection — detect and resolve conflicting memories.

Principle from Claude Code: "Memory says X exists ≠ X exists now. Verify before use."

Drift types:
  - Contradiction: "user likes A" vs "user dislikes A" (simultaneous conflicting claims)
  - Supersession: newer info replaces older (temporal progression, NOT contradiction)
  - Stale: memory >90 days not referenced
"""
from __future__ import annotations
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from .types import MemoryType, MemoryEntry

logger = logging.getLogger(__name__)

# Threshold for keyword overlap to consider potential conflict
_CONFLICT_THRESHOLD = 0.4
# Days after which unreferenced memories are considered stale
_STALE_DAYS = 90


class DriftDetector:
    """Detect conflicting, stale, or superseded memories."""

    def __init__(self, memory_manager):
        self.memory = memory_manager
        self._conflict_count = 0
        self._supersede_count = 0
        self._stale_count = 0

    async def check(self, new_content: str) -> Optional[dict]:
        """Check if new content conflicts with existing memories.

        Returns dict with:
          - 'entry': the conflicting MemoryEntry
          - 'drift_type': 'contradiction' | 'supersession'
        Or None if no conflict.
        """
        existing = await self.memory.recall(new_content, top_k=5)
        if not existing:
            return None

        new_lower = new_content.lower()
        new_words = set(new_lower.split())

        for entry in existing:
            old_lower = entry.content.lower()
            old_words = set(old_lower.split())

            if not new_words or not old_words:
                continue
            overlap = len(new_words & old_words) / min(len(new_words), len(old_words))

            if overlap < _CONFLICT_THRESHOLD:
                continue

            # Check supersession FIRST — updates are not contradictions
            if _is_supersede(new_lower, old_lower):
                self._supersede_count += 1
                logger.info(
                    "Drift: supersession detected (#%d)",
                    self._supersede_count,
                )
                return {"entry": entry, "drift_type": "supersession"}

            # Only after ruling out supersession, check for contradiction
            if overlap > 0.6 and _is_contradiction(new_lower, old_lower):
                self._conflict_count += 1
                logger.warning(
                    "Drift: contradiction detected (conflict #%d)",
                    self._conflict_count,
                )
                return {"entry": entry, "drift_type": "contradiction"}

        return None

    async def scan_stale(self, max_age_days: int = _STALE_DAYS) -> list[MemoryEntry]:
        """Find memories older than max_age_days that haven't been accessed.

        Uses all_entries() which works for both SQLite and JSON backends.
        Previous implementation accessed _memories dict directly, which is
        empty in SQLite mode — making stale detection silently fail (P0 fix).
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        stale: list[MemoryEntry] = []

        for entry in self.memory.all_entries():
            if entry.created_at < cutoff:
                if entry.metadata.get("superseded_by"):
                    stale.append(entry)
                    continue
                last_access = entry.metadata.get("last_accessed")
                if last_access:
                    try:
                        la_dt = datetime.fromisoformat(last_access)
                        if la_dt < cutoff:
                            stale.append(entry)
                    except (ValueError, TypeError):
                        stale.append(entry)
                else:
                    stale.append(entry)

        self._stale_count = len(stale)
        return stale

    async def gc(self, max_age_days: int = _STALE_DAYS, dry_run: bool = True) -> dict:
        """Garbage collect stale memories.

        Uses update_metadata() for persistence — works in both SQLite and JSON mode.
        Previous implementation used save() which is a no-op in SQLite mode.
        """
        stale = await self.scan_stale(max_age_days)

        result = {
            "stale_count": len(stale),
            "deleted_count": 0,
            "archived_count": 0,
            "dry_run": dry_run,
        }

        if dry_run:
            return result

        for entry in stale:
            archive_meta = {
                "archived": True,
                "archived_at": datetime.now().isoformat(),
                "archived_reason": "stale",
            }
            entry.metadata.update(archive_meta)
            await self.memory.update_metadata(entry.id, archive_meta)
            result["archived_count"] += 1

        logger.info("GC: archived %d stale memories", result["archived_count"])
        return result

    @property
    def stats(self) -> dict:
        return {
            "conflict_count": self._conflict_count,
            "supersede_count": self._supersede_count,
            "stale_count": self._stale_count,
        }


def _is_contradiction(a: str, b: str) -> bool:
    """Heuristic: detect TRUE contradiction between two texts.

    Only fires for simultaneous conflicting claims (preference/opinion),
    NOT for temporal updates (those are supersessions).
    Uses word boundaries to avoid substring false matches.
    """
    # Only keep pairs that indicate genuine preference/opinion contradiction
    negation_pairs = [
        ("likes", "dislikes"), ("prefers", "avoids"),
        ("should", "should not"), ("enabled", "disabled"),
        ("correct", "incorrect"),
    ]

    for pos, neg in negation_pairs:
        pos_re = re.compile(r'\b' + re.escape(pos) + r'\b')
        neg_re = re.compile(r'\b' + re.escape(neg) + r'\b')
        if (pos_re.search(a) and neg_re.search(b)) or \
           (neg_re.search(a) and pos_re.search(b)):
            return True

    return False


def _is_supersede(new: str, old: str) -> bool:
    """Heuristic: detect if new info supersedes old.

    Key insight: if the new text describes a CHANGE or DECISION,
    it's a temporal update, not a contradiction.
    """
    # Update signals in the NEW text
    update_signals = [
        "now", "updated", "changed", "instead",
        "replaced", "no longer", "actually", "correction",
        "switched to", "migrated", "refactored", "fixed",
        "decided to", "should use", "wired in", "is more reliable",
        "现在", "更新", "改为", "替换", "不再", "修复",
    ]

    # Staleness signals in the OLD text
    stale_signals = [
        "was", "used to", "previously", "dead code",
        "deprecated", "broken", "truncat",
    ]

    if any(signal in new for signal in update_signals):
        return True
    if any(signal in old for signal in stale_signals):
        return True

    # Temporal heuristic: same entity, different numeric values
    new_nums = set(re.findall(r'\b\d+\b', new))
    old_nums = set(re.findall(r'\b\d+\b', old))
    if new_nums and old_nums and new_nums != old_nums:
        new_ctx = set(re.findall(r'\b[a-z]{4,}\b', new))
        old_ctx = set(re.findall(r'\b[a-z]{4,}\b', old))
        if len(new_ctx & old_ctx) >= 2:
            return True

    return False
