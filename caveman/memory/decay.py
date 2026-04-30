"""Memory decay — time-based trust erosion for unused memories.

The flywheel's garbage collector. Without decay, low-quality memories
accumulate forever, polluting retrieval results and wasting resources.

Decay rules (compound interest logic):
  - Memories unused for >30 days: trust -= 0.02/day (slow fade)
  - Memories unused for >90 days with trust < 0.3: candidates for pruning
  - High-trust memories (>0.7) decay 3x slower (earned trust is durable)
  - Recently helpful memories are immune (last_accessed within 14 days)
  - Pruned memories are soft-deleted (moved to archive, not destroyed)

This runs as a background task, not in the hot path.
"""
from __future__ import annotations

import json
import logging
import sqlite3

from caveman.db import connect as db_connect
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

__all__ = ["MemoryDecay", "DecayResult"]

logger = logging.getLogger(__name__)

# Decay parameters
_DECAY_START_DAYS = 30       # Start decaying after 30 days unused
_DECAY_RATE_PER_DAY = 0.02   # Trust loss per day for normal memories
_HIGH_TRUST_THRESHOLD = 0.7  # High-trust memories decay slower
_HIGH_TRUST_SLOWDOWN = 3.0   # 3x slower decay for high-trust
_IMMUNE_DAYS = 14            # Recently accessed memories are immune
_PRUNE_THRESHOLD = 0.05      # Trust below this → candidate for pruning
_PRUNE_AGE_DAYS = 90         # Must be this old to be pruned
_MAX_DECAY_PER_RUN = 2000    # Process enough rows to govern bulk-import noise in one pass
_QUARANTINE_TRUST_THRESHOLD = 0.07
_QUARANTINE_DOWNRANK_TRUST = 0.01
_IMPORT_SOURCE_PREFIX = "import:"


@dataclass
class DecayResult:
    """Result of a decay run."""
    memories_scanned: int = 0
    memories_decayed: int = 0
    memories_pruned: int = 0
    memories_quarantined: int = 0
    trust_total_reduced: float = 0.0

    def summary(self) -> str:
        return (
            f"Decay: scanned={self.memories_scanned}, "
            f"decayed={self.memories_decayed}, "
            f"pruned={self.memories_pruned}, "
            f"quarantined={self.memories_quarantined}, "
            f"trust_reduced={self.trust_total_reduced:.3f}"
        )


class MemoryDecay:
    """Time-based trust decay for unused memories."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        archive_dir: Path | str | None = None,
    ) -> None:
        from caveman.paths import MEMORY_DIR
        self._db_path = Path(db_path) if db_path else MEMORY_DIR / "caveman.db"
        self._archive_dir = Path(archive_dir) if archive_dir else MEMORY_DIR / "archive"

    def run(self, dry_run: bool = False) -> DecayResult:
        """Run decay pass over all memories.

        Args:
            dry_run: If True, compute but don't apply changes.

        Returns:
            DecayResult with statistics.
        """
        result = DecayResult()
        now = datetime.now(timezone.utc)

        if not self._db_path.exists():
            return result

        conn = db_connect((self._db_path))
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(
                "SELECT id, content, type, created_at, trust_score, "
                "retrieval_count, helpful_count, metadata_json "
                "FROM memories ORDER BY trust_score ASC LIMIT ?",
                (_MAX_DECAY_PER_RUN,),
            ).fetchall()

            to_decay: list[tuple[str, float]] = []  # (id, new_trust)
            to_prune: list[dict] = []  # full row data for archive
            to_quarantine: list[tuple[str, float, dict[str, Any]]] = []

            for row in rows:
                result.memories_scanned += 1
                mid = row["id"]
                trust = row["trust_score"]
                retrieval_count = row["retrieval_count"]
                helpful_count = row["helpful_count"]
                created_at = row["created_at"]
                metadata = json.loads(row["metadata_json"] or "{}")

                # Parse dates
                try:
                    created = datetime.fromisoformat(created_at)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    created = now - timedelta(days=365)  # assume old

                last_accessed = metadata.get("last_accessed")
                if last_accessed:
                    try:
                        la = datetime.fromisoformat(last_accessed)
                        if la.tzinfo is None:
                            la = la.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        la = created
                else:
                    la = created

                # Check immunity: recently accessed
                days_since_access = (now - la).total_seconds() / 86400
                if days_since_access < _IMMUNE_DAYS:
                    continue

                # Check if old enough to decay
                if days_since_access < _DECAY_START_DAYS:
                    continue

                # Calculate decay amount
                excess_days = days_since_access - _DECAY_START_DAYS
                rate = _DECAY_RATE_PER_DAY
                if trust >= _HIGH_TRUST_THRESHOLD:
                    rate /= _HIGH_TRUST_SLOWDOWN

                # Memories with high retrieval count decay slower
                if retrieval_count > 10:
                    rate *= 0.5

                decay_amount = rate * min(excess_days, 30)  # cap at 30 days of decay per run
                new_trust = max(0.0, trust - decay_amount)

                if new_trust < trust:
                    to_decay.append((mid, new_trust))
                    result.trust_total_reduced += trust - new_trust
                    result.memories_decayed += 1

                # Check for pruning
                if (new_trust <= _PRUNE_THRESHOLD
                        and (now - created).days >= _PRUNE_AGE_DAYS
                        and retrieval_count == 0):
                    source = str(metadata.get("source", ""))
                    if source.startswith(_IMPORT_SOURCE_PREFIX):
                        if helpful_count > 0:
                            continue
                        metadata.setdefault("previous_trust_score", trust)
                        metadata["governance_state"] = "quarantined"
                        metadata["quarantine_reason"] = "stale_low_signal_import"
                        metadata["quarantined_at"] = now.isoformat()
                        to_quarantine.append((mid, _QUARANTINE_DOWNRANK_TRUST, metadata))
                        result.memories_quarantined += 1
                    else:
                        to_prune.append(dict(row))
                        result.memories_pruned += 1

            if dry_run:
                return result

            # Apply decay
            if to_decay:
                quarantine_ids = {mid for mid, _, _ in to_quarantine}
                decay_updates = [(new_trust, mid) for mid, new_trust in to_decay if mid not in quarantine_ids]
                if decay_updates:
                    conn.executemany(
                        "UPDATE memories SET trust_score = ? WHERE id = ?",
                        decay_updates,
                    )
                    conn.commit()

            if to_quarantine:
                conn.executemany(
                    "UPDATE memories SET trust_score = ?, metadata_json = ? WHERE id = ?",
                    [
                        (new_trust, json.dumps(metadata, ensure_ascii=False), mid)
                        for mid, new_trust, metadata in to_quarantine
                    ],
                )
                conn.commit()

            # Archive and prune
            if to_prune:
                self._archive_and_prune(conn, to_prune)

        finally:
            conn.close()

        if result.memories_decayed > 0 or result.memories_pruned > 0 or result.memories_quarantined > 0:
            logger.info(result.summary())

        return result

    def _archive_and_prune(self, conn: sqlite3.Connection, rows: list[dict]) -> None:
        """Archive pruned memories to JSONL, then delete from DB."""
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self._archive_dir / f"pruned_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(archive_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

        ids = [row["id"] for row in rows]
        # Delete from FTS index first, then main table
        for mid in ids:
            try:
                conn.execute(
                    "DELETE FROM memories_fts WHERE rowid = "
                    "(SELECT rowid FROM memories WHERE id = ?)", (mid,)
                )
            except sqlite3.OperationalError:
                pass  # FTS table might not exist
            conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
        conn.commit()
        logger.info("Archived and pruned %d memories → %s", len(ids), archive_path)
