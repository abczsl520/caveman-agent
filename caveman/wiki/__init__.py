"""Wiki Compiler — compile, don't retrieve.

Inspired by Karpathy's LLM Wiki pattern (April 3, 2026):
Instead of searching memories on every query, periodically compile all
knowledge into a structured wiki. The wiki is a persistent, compounding
artifact that gets richer with every session.

4-tier knowledge pyramid:
  Working   — recent observations, not yet processed (hours)
  Episodic  — session summaries, compressed from working (days)
  Semantic  — cross-session facts, consolidated from episodes (weeks)
  Procedural — workflows and patterns, extracted from semantics (months)

Each tier is more compressed, more confident, and longer-lived.
Knowledge promotes upward as evidence accumulates.

MIT License — Pattern credit: Andrej Karpathy
"""
from __future__ import annotations

__all__ = [
    "TIERS",
    "TIER_CONFIG",
    "WikiEntry",
    "WikiStore",
    "CompilationResult",
    "provenance",
]

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caveman.errors import CavemanError
from caveman.paths import MEMORY_DIR, WIKI_DIR

logger = logging.getLogger(__name__)

# --- Knowledge Tiers ---

TIERS = ("working", "episodic", "semantic", "procedural")

TIER_CONFIG = {
    "working": {"max_age_hours": 24, "min_confidence": 0.0},
    "episodic": {"max_age_hours": 168, "min_confidence": 0.3},  # 7 days
    "semantic": {"max_age_hours": 2160, "min_confidence": 0.5},  # 90 days
    "procedural": {"max_age_hours": 0, "min_confidence": 0.7},  # no expiry
}


@dataclass
class WikiEntry:
    """A single knowledge entry in the wiki."""

    id: str
    tier: str
    title: str
    content: str
    confidence: float = 0.5
    sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)  # wikilinks to other entries
    created_at: str = ""
    updated_at: str = ""
    access_count: int = 0
    reinforcement_count: int = 0

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
        if self.tier not in TIERS:
            raise CavemanError(f"Invalid tier: {self.tier}", context={"valid": list(TIERS)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tier": self.tier,
            "title": self.title,
            "content": self.content,
            "confidence": self.confidence,
            "sources": self.sources,
            "tags": self.tags,
            "links": self.links,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
            "reinforcement_count": self.reinforcement_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WikiEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_markdown(self) -> str:
        """Render as markdown with YAML frontmatter."""
        fm = [
            "---",
            f"id: {self.id}",
            f"tier: {self.tier}",
            f"confidence: {self.confidence:.2f}",
            f"tags: [{', '.join(self.tags)}]",
            f"sources: {len(self.sources)}",
            f"created: {self.created_at[:10]}",
            f"updated: {self.updated_at[:10]}",
            "---",
        ]
        links_section = ""
        if self.links:
            links_section = "\n\n## Related\n" + "\n".join(f"- [[{lnk}]]" for lnk in self.links)
        return "\n".join(fm) + f"\n\n# {self.title}\n\n{self.content}{links_section}\n"


@dataclass
class CompilationResult:
    """Result of a wiki compilation pass."""

    entries_processed: int = 0
    entries_promoted: int = 0
    entries_expired: int = 0
    entries_merged: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class WikiStore:
    """Persistent storage for wiki entries using JSON files per tier."""

    def __init__(self, wiki_dir: Path | None = None) -> None:
        self._dir = wiki_dir or WIKI_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def _tier_path(self, tier: str) -> Path:
        return self._dir / f"{tier}.json"

    def load_tier(self, tier: str) -> list[WikiEntry]:
        path = self._tier_path(tier)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [WikiEntry.from_dict(e) for e in data]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to load tier %s: %s", tier, exc)
            return []

    def save_tier(self, tier: str, entries: list[WikiEntry]) -> None:
        path = self._tier_path(tier)
        data = [e.to_dict() for e in entries]
        content = json.dumps(data, indent=2, ensure_ascii=False)
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

    def load_all(self) -> dict[str, list[WikiEntry]]:
        return {tier: self.load_tier(tier) for tier in TIERS}

    def add(self, entry: WikiEntry) -> None:
        # Remove from any other tier first (handles tier changes)
        for tier in TIERS:
            if tier == entry.tier:
                continue
            old = self.load_tier(tier)
            filtered = [e for e in old if e.id != entry.id]
            if len(filtered) < len(old):
                self.save_tier(tier, filtered)
        entries = self.load_tier(entry.tier)
        # Update if exists, else append
        for i, e in enumerate(entries):
            if e.id == entry.id:
                entries[i] = entry
                self.save_tier(entry.tier, entries)
                return
        entries.append(entry)
        self.save_tier(entry.tier, entries)

    def remove(self, tier: str, entry_id: str) -> bool:
        entries = self.load_tier(tier)
        before = len(entries)
        entries = [e for e in entries if e.id != entry_id]
        if len(entries) < before:
            self.save_tier(tier, entries)
            return True
        return False

    def get(self, entry_id: str) -> WikiEntry | None:
        for tier in TIERS:
            for e in self.load_tier(tier):
                if e.id == entry_id:
                    return e
        return None

    def search(self, query: str, top_k: int = 10) -> list[WikiEntry]:
        """Simple keyword search across all tiers."""
        query_lower = query.lower()
        results: list[tuple[float, WikiEntry]] = []
        for tier in TIERS:
            for entry in self.load_tier(tier):
                score = 0.0
                text = f"{entry.title} {entry.content} {' '.join(entry.tags)}".lower()
                for word in query_lower.split():
                    if word in text:
                        score += 1.0
                # Boost by tier (procedural > semantic > episodic > working)
                tier_boost = {"working": 1.0, "episodic": 1.2, "semantic": 1.5, "procedural": 2.0}
                score *= tier_boost.get(entry.tier, 1.0)
                score *= entry.confidence
                if score > 0:
                    results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:top_k]]

    def stats(self) -> dict[str, int]:
        return {tier: len(self.load_tier(tier)) for tier in TIERS}
