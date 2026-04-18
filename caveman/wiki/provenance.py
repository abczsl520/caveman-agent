"""Wiki Provenance — source tracking and contradiction detection.

Tracks where each wiki entry came from (session, tool, user input)
and detects contradictions between entries in the same tier or across tiers.

Contradiction detection uses:
  1. Tag overlap — entries with same tags may conflict
  2. Title similarity — entries about the same topic
  3. Content negation — explicit contradictions ("X is Y" vs "X is not Y")

When contradictions are found, the lower-confidence entry is weakened.
If both have equal confidence, both are flagged for human review.

MIT License
"""
from __future__ import annotations

__all__ = ["ProvenanceTracker", "ContradictionDetector", "Provenance", "Contradiction"]

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from caveman.wiki import WikiEntry, WikiStore, TIERS

logger = logging.getLogger(__name__)

# --- Data structures ---


@dataclass
class Provenance:
    """Source tracking for a wiki entry."""

    entry_id: str
    source_type: str  # "session" | "tool" | "user" | "reflect" | "nudge" | "import"
    source_id: str  # session_id, tool_name, etc.
    timestamp: str = ""
    context: str = ""  # brief description of how this knowledge was acquired

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(
            entry_id=d.get("entry_id", ""),
            source_type=d.get("source_type", "unknown"),
            source_id=d.get("source_id", ""),
            timestamp=d.get("timestamp", ""),
            context=d.get("context", ""),
        )


@dataclass
class Contradiction:
    """A detected contradiction between two wiki entries."""

    entry_a_id: str
    entry_b_id: str
    reason: str  # why they contradict
    confidence: float = 0.5  # how confident we are this is a real contradiction
    resolved: bool = False
    resolution: str = ""  # "a_wins" | "b_wins" | "merged" | "both_kept"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_a_id": self.entry_a_id,
            "entry_b_id": self.entry_b_id,
            "reason": self.reason,
            "confidence": self.confidence,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Contradiction:
        return cls(
            entry_a_id=d.get("entry_a_id", ""),
            entry_b_id=d.get("entry_b_id", ""),
            reason=d.get("reason", ""),
            confidence=d.get("confidence", 0.5),
            resolved=d.get("resolved", False),
            resolution=d.get("resolution", ""),
        )


# --- Negation patterns for contradiction detection ---

_NEGATION_PAIRS = [
    # English
    (r"\b(is|are|was|were)\s+", r"\b(is|are|was|were)\s+not\s+"),
    (r"\b(can|could|should|will)\s+", r"\b(can|could|should|will)\s+not\s+"),
    (r"\b(always)\b", r"\b(never)\b"),
    (r"\b(true)\b", r"\b(false)\b"),
    (r"\b(enable|enabled)\b", r"\b(disable|disabled)\b"),
    (r"\b(yes)\b", r"\b(no)\b"),
    # Chinese
    (r"是", r"不是"),
    (r"可以", r"不可以|不能"),
    (r"应该", r"不应该"),
    (r"总是|一直", r"从不|从来不"),
    (r"启用", r"禁用"),
]


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _word_set(text: str) -> set[str]:
    """Extract word set for Jaccard comparison. Handles CJK characters."""
    # Split on whitespace and punctuation for Latin text
    words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    # Add individual CJK characters (Chinese/Japanese/Korean)
    cjk = set(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
    return words | cjk


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _has_negation_conflict(text_a: str, text_b: str) -> bool:
    """Check if two texts have negation-based contradictions."""
    a_lower = text_a.lower()
    b_lower = text_b.lower()
    for pos_pat, neg_pat in _NEGATION_PAIRS:
        a_pos = bool(re.search(pos_pat, a_lower))
        a_neg = bool(re.search(neg_pat, a_lower))
        b_pos = bool(re.search(pos_pat, b_lower))
        b_neg = bool(re.search(neg_pat, b_lower))
        if (a_pos and b_neg) or (a_neg and b_pos):
            return True
    return False


# --- Provenance Tracker ---


class ProvenanceTracker:
    """Tracks provenance (source history) for wiki entries.

    Stores provenance records alongside the wiki store.
    Each entry can have multiple provenance records (e.g., reinforced
    from different sessions).
    """

    def __init__(self, store: WikiStore | None = None) -> None:
        from caveman.wiki import WikiStore as WS
        self._store = store or WS()
        self._provenance: dict[str, list[Provenance]] = {}
        self._load()

    def _load(self) -> None:
        """Load provenance data from store directory."""
        import json
        path = self._store._dir / "provenance.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for entry_id, records in data.items():
                    self._provenance[entry_id] = [
                        Provenance.from_dict(r) for r in records
                    ]
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to load provenance data, starting fresh")

    def _save(self) -> None:
        """Persist provenance data."""
        import json
        path = self._store._dir / "provenance.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            eid: [p.to_dict() for p in records]
            for eid, records in self._provenance.items()
        }
        content = json.dumps(data, ensure_ascii=False, indent=2)
        # Atomic write: temp file in same dir → os.replace()
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def record(
        self,
        entry_id: str,
        source_type: str,
        source_id: str,
        context: str = "",
    ) -> Provenance:
        """Record a provenance event for an entry."""
        prov = Provenance(
            entry_id=entry_id,
            source_type=source_type,
            source_id=source_id,
            context=context,
        )
        if entry_id not in self._provenance:
            self._provenance[entry_id] = []
        self._provenance[entry_id].append(prov)
        self._save()
        return prov

    def get(self, entry_id: str) -> list[Provenance]:
        """Get all provenance records for an entry."""
        return self._provenance.get(entry_id, [])

    def get_sources(self, entry_id: str) -> list[str]:
        """Get unique source IDs for an entry."""
        return list({p.source_id for p in self.get(entry_id)})

    def count_sources(self, entry_id: str) -> int:
        """Count unique sources that contributed to an entry."""
        return len(self.get_sources(entry_id))

    def cleanup(self, valid_ids: set[str]) -> int:
        """Remove provenance for entries that no longer exist."""
        removed = 0
        for eid in list(self._provenance.keys()):
            if eid not in valid_ids:
                del self._provenance[eid]
                removed += 1
        if removed:
            self._save()
        return removed


# --- Contradiction Detector ---


class ContradictionDetector:
    """Detects contradictions between wiki entries.

    Uses three signals:
      1. Tag overlap (same topic area)
      2. Title/content similarity (Jaccard)
      3. Negation patterns (explicit contradictions)

    Scoring:
      - tag_overlap >= 0.5 AND jaccard >= 0.3 AND negation → high confidence (0.8)
      - tag_overlap >= 0.5 AND jaccard >= 0.5 → medium confidence (0.5)
      - negation alone with jaccard >= 0.4 → medium confidence (0.6)
    """

    def __init__(self, store: WikiStore | None = None) -> None:
        from caveman.wiki import WikiStore as WS
        self._store = store or WS()

    def detect(self, scope: str = "all") -> list[Contradiction]:
        """Detect contradictions across wiki entries.

        Args:
            scope: "all" checks all tiers, or a specific tier name.

        Returns:
            List of detected contradictions, sorted by confidence desc.
        """
        entries: list[WikiEntry] = []
        tiers = TIERS if scope == "all" else (scope,)
        for tier in tiers:
            entries.extend(self._store.load_tier(tier))

        contradictions: list[Contradiction] = []
        seen: set[tuple[str, str]] = set()

        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                pair = (min(a.id, b.id), max(a.id, b.id))
                if pair in seen:
                    continue
                seen.add(pair)

                contradiction = self._check_pair(a, b)
                if contradiction:
                    contradictions.append(contradiction)

        contradictions.sort(key=lambda c: c.confidence, reverse=True)
        return contradictions

    def _check_pair(self, a: WikiEntry, b: WikiEntry) -> Contradiction | None:
        """Check if two entries contradict each other."""
        # Signal 1: Tag overlap
        a_tags = set(a.tags)
        b_tags = set(b.tags)
        tag_overlap = len(a_tags & b_tags) / max(len(a_tags | b_tags), 1)

        # Signal 2: Content similarity (Jaccard on words)
        a_words = _word_set(a.content)
        b_words = _word_set(b.content)
        content_sim = _jaccard(a_words, b_words)

        # Signal 3: Negation conflict
        has_negation = _has_negation_conflict(a.content, b.content)

        # Scoring
        confidence = 0.0
        reason = ""

        if tag_overlap >= 0.5 and content_sim >= 0.3 and has_negation:
            confidence = 0.8
            reason = f"Same topic (tags overlap {tag_overlap:.0%}) with negation conflict"
        elif has_negation and content_sim >= 0.4:
            confidence = 0.6
            reason = f"Negation conflict with content similarity {content_sim:.0%}"
        elif tag_overlap >= 0.5 and content_sim >= 0.5:
            confidence = 0.5
            reason = f"High similarity ({content_sim:.0%}) in same topic area"
        else:
            return None

        return Contradiction(
            entry_a_id=a.id,
            entry_b_id=b.id,
            reason=reason,
            confidence=confidence,
        )

    def auto_resolve(self, contradictions: list[Contradiction]) -> int:
        """Auto-resolve contradictions by weakening lower-confidence entries.

        Returns number of contradictions resolved.
        """
        resolved = 0
        for c in contradictions:
            if c.resolved:
                continue

            a = self._store.get(c.entry_a_id)
            b = self._store.get(c.entry_b_id)
            if not a or not b:
                c.resolved = True
                c.resolution = "entry_missing"
                resolved += 1
                continue

            # Higher confidence wins; weaken the loser
            if a.confidence > b.confidence + 0.1:
                b.confidence = max(0.0, b.confidence - 0.15)
                b.updated_at = datetime.now(timezone.utc).isoformat()
                self._store.add(b)
                c.resolved = True
                c.resolution = "a_wins"
            elif b.confidence > a.confidence + 0.1:
                a.confidence = max(0.0, a.confidence - 0.15)
                a.updated_at = datetime.now(timezone.utc).isoformat()
                self._store.add(a)
                c.resolved = True
                c.resolution = "b_wins"
            else:
                # Too close to call — flag for human review
                c.resolution = "needs_review"
                continue

            resolved += 1
        return resolved
