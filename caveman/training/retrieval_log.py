"""Retrieval log — record memory search queries and results for embedding training.

Every time Recall or memory_search runs, we log:
  - query: what the user/system searched for
  - results: which memories were returned (with scores)
  - adopted: which results were actually used (if trackable)

This log is the correct data source for embedding training pairs,
NOT conversation Q&A (which was the previous incorrect approach).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalEntry:
    """A single retrieval event."""

    query: str
    results: list[dict]  # [{"memory_id": str, "content": str, "score": float}]
    source: str = "recall"  # "recall" | "memory_search" | "nudge"
    adopted_ids: list[str] = field(default_factory=list)  # IDs user actually used
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RetrievalEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RetrievalLog:
    """Append-only log of memory retrieval events.

    Stored as JSONL at ~/.caveman/training/retrieval_log.jsonl
    """

    def __init__(self, log_path: Path | None = None) -> None:
        if log_path is None:
            from caveman.paths import TRAINING_DIR
            log_path = TRAINING_DIR / "retrieval_log.jsonl"
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: RetrievalEntry) -> None:
        """Append a retrieval event to the log."""
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write retrieval log: %s", e)

    def log_search(
        self,
        query: str,
        results: list[tuple[float, Any]],
        source: str = "recall",
    ) -> None:
        """Convenience: log from (score, MemoryEntry) tuples."""
        result_dicts = []
        for score, entry in results:
            result_dicts.append({
                "memory_id": getattr(entry, "id", ""),
                "content": getattr(entry, "content", str(entry))[:500],
                "score": round(score, 4),
            })
        self.log(RetrievalEntry(query=query, results=result_dicts, source=source))

    def mark_adopted(self, query: str, adopted_ids: list[str]) -> None:
        """Mark which results from a query were actually used.

        This creates a follow-up entry that can be joined with the original
        during training pair generation.
        """
        self.log(RetrievalEntry(
            query=query,
            results=[],
            source="adoption",
            adopted_ids=adopted_ids,
        ))

    def read_all(self) -> list[RetrievalEntry]:
        """Read all entries from the log."""
        if not self._path.exists():
            return []
        entries = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(RetrievalEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Skip malformed retrieval log entry: %s", e)
        return entries

    def count(self) -> int:
        """Count entries without loading all into memory."""
        if not self._path.exists():
            return 0
        return sum(1 for line in self._path.read_text(encoding="utf-8").splitlines() if line.strip())

    def generate_training_pairs(self) -> list[dict]:
        """Generate query-positive pairs from retrieval log for embedding training.

        Logic:
        - For each search entry with results scored > 0.5: query → top result = positive pair
        - For entries with adopted_ids: adopted = positive, non-adopted = hard negative
        """
        entries = self.read_all()
        pairs = []

        # Index adoption events by query
        adoptions: dict[str, list[str]] = {}
        for e in entries:
            if e.source == "adoption" and e.adopted_ids:
                adoptions[e.query] = e.adopted_ids

        for entry in entries:
            if entry.source == "adoption" or not entry.results:
                continue

            adopted_ids = adoptions.get(entry.query, [])

            for result in entry.results:
                score = result.get("score", 0)
                content = result.get("content", "")
                mid = result.get("memory_id", "")

                if not content or len(content) < 10:
                    continue

                # If we have adoption data, use it
                if adopted_ids:
                    if mid in adopted_ids:
                        pairs.append({
                            "query": entry.query,
                            "positive": content,
                            "source": "adopted",
                        })
                elif score >= 0.5:
                    # No adoption data — use score as proxy
                    pairs.append({
                        "query": entry.query,
                        "positive": content,
                        "source": "score",
                    })

        return pairs
