"""Wiki Compiler Engine — the core compilation loop.

Compiles raw memories into structured wiki entries, promotes knowledge
up the tier pyramid, expires stale entries, and generates the compiled
context for system prompt injection.

The compilation loop:
  1. Ingest: raw memories → working tier entries
  2. Consolidate: working → episodic (session summaries)
  3. Promote: episodic → semantic (cross-session facts)
  4. Extract: semantic → procedural (patterns and workflows)
  5. Expire: remove stale entries based on tier config
  6. Compile: generate system prompt context from all tiers

Credit: Karpathy LLM Wiki pattern (April 3, 2026)
"""
from __future__ import annotations

__all__ = ["WikiCompiler"]

import hashlib
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any

from caveman.wiki import (
    TIER_CONFIG,
    TIERS,
    CompilationResult,
    WikiEntry,
    WikiStore,
)

logger = logging.getLogger(__name__)

# Promotion thresholds
_PROMOTE_REINFORCEMENTS = {
    "working": 2,     # 2 reinforcements → episodic
    "episodic": 3,    # 3 reinforcements → semantic
    "semantic": 5,    # 5 reinforcements → procedural
}

# Max entries per tier (prevent unbounded growth)
_MAX_ENTRIES = {
    "working": 100,
    "episodic": 200,
    "semantic": 500,
    "procedural": 200,
}

# Token budgets for compiled context
_TOKEN_BUDGETS = {
    "procedural": 2000,  # highest priority
    "semantic": 1500,
    "episodic": 800,
    "working": 500,
}


def _entry_id(content: str) -> str:
    """Generate a stable ID from content."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _hours_since(iso_timestamp: str) -> float:
    """Hours elapsed since an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        return delta.total_seconds() / 3600
    except (ValueError, TypeError):
        return float("inf")  # corrupted timestamps → treat as ancient → expire


class WikiCompiler:
    """Compiles memories into a structured, tiered wiki.

    Usage:
        compiler = WikiCompiler()

        # Add raw observations
        compiler.ingest("Python uses 0-based indexing", tags=["python"])

        # Run compilation (consolidate + promote + expire)
        result = compiler.compile()

        # Get compiled context for system prompt
        context = compiler.get_compiled_context(max_tokens=4000)
    """

    def __init__(self, store: WikiStore | None = None) -> None:
        self._store = store or WikiStore()

    @property
    def store(self) -> WikiStore:
        return self._store

    def ingest(
        self,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        source: str = "",
        confidence: float = 0.5,
    ) -> WikiEntry:
        """Ingest a raw observation into the working tier."""
        entry_id = _entry_id(content)

        # Check if already exists (reinforce instead of duplicate)
        existing = self._store.get(entry_id)
        if existing:
            existing.reinforcement_count += 1
            existing.confidence = min(1.0, existing.confidence + 0.05)
            existing.updated_at = datetime.now(timezone.utc).isoformat()
            if source and source not in existing.sources:
                existing.sources.append(source)
            self._store.add(existing)
            return existing

        entry = WikiEntry(
            id=entry_id,
            tier="working",
            title=title or content[:60].strip(),
            content=content,
            confidence=confidence,
            sources=[source] if source else [],
            tags=tags or [],
        )
        self._store.add(entry)
        return entry

    def ingest_session(
        self,
        session_id: str,
        task: str,
        decisions: list[str],
        progress: list[str],
        todos: list[str],
    ) -> WikiEntry:
        """Ingest a session summary directly into the episodic tier."""
        parts = []
        if task:
            parts.append(f"Task: {task}")
        if decisions:
            parts.append("Decisions:\n" + "\n".join(f"- {d}" for d in decisions))
        if progress:
            parts.append("Progress:\n" + "\n".join(f"- {p}" for p in progress))
        if todos:
            parts.append("Open TODOs:\n" + "\n".join(f"- [ ] {t}" for t in todos))

        content = "\n\n".join(parts)
        entry = WikiEntry(
            id=_entry_id(f"session:{session_id}"),
            tier="episodic",
            title=f"Session: {task[:50]}" if task else f"Session {session_id[:8]}",
            content=content,
            confidence=0.7,
            sources=[f"session:{session_id}"],
            tags=["session"],
        )
        self._store.add(entry)
        return entry

    def compile(self) -> CompilationResult:
        """Run the full compilation loop."""
        start = time.monotonic()
        result = CompilationResult()

        # Phase 1: Promote entries that have enough reinforcements
        for tier_idx in range(len(TIERS) - 1):
            src_tier = TIERS[tier_idx]
            dst_tier = TIERS[tier_idx + 1]
            threshold = _PROMOTE_REINFORCEMENTS.get(src_tier, 999)

            entries = self._store.load_tier(src_tier)
            promoted = []
            remaining = []

            for entry in entries:
                if entry.reinforcement_count >= threshold:
                    entry.tier = dst_tier
                    entry.reinforcement_count = 0  # reset after promotion
                    entry.updated_at = datetime.now(timezone.utc).isoformat()
                    self._store.add(entry)
                    promoted.append(entry)
                    result.entries_promoted += 1
                else:
                    remaining.append(entry)

            if promoted:
                self._store.save_tier(src_tier, remaining)

        # Phase 2: Expire stale entries
        for tier in TIERS:
            config = TIER_CONFIG[tier]
            max_age = config["max_age_hours"]
            if max_age <= 0:
                continue  # no expiry for this tier

            entries = self._store.load_tier(tier)
            active = []
            for entry in entries:
                age = _hours_since(entry.updated_at)
                if age <= max_age:
                    active.append(entry)
                else:
                    result.entries_expired += 1

            if len(active) < len(entries):
                self._store.save_tier(tier, active)

        # Phase 3: Enforce max entries per tier (keep highest confidence)
        for tier in TIERS:
            max_count = _MAX_ENTRIES.get(tier, 500)
            entries = self._store.load_tier(tier)
            if len(entries) > max_count:
                entries.sort(key=lambda e: e.confidence, reverse=True)
                overflow = len(entries) - max_count
                entries = entries[:max_count]
                self._store.save_tier(tier, entries)
                result.entries_expired += overflow

        # Count total
        for tier in TIERS:
            result.entries_processed += len(self._store.load_tier(tier))

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    def get_compiled_context(self, max_tokens: int = 4000) -> str:
        """Generate compiled wiki context for system prompt injection.

        Returns a structured text block with the most important knowledge
        from all tiers, respecting token budgets.
        """
        sections: list[str] = []
        chars_used = 0
        # ~4 chars per token estimate
        max_chars = max_tokens * 4

        # Process tiers in priority order (procedural first)
        for tier in reversed(TIERS):
            budget_tokens = _TOKEN_BUDGETS.get(tier, 500)
            budget_chars = min(budget_tokens * 4, max_chars - chars_used)
            if budget_chars <= 0:
                break

            entries = self._store.load_tier(tier)
            if not entries:
                continue

            # Sort by confidence * access_count (most useful first)
            entries.sort(
                key=lambda e: e.confidence * (1 + e.access_count * 0.1),
                reverse=True,
            )

            tier_lines: list[str] = []
            tier_chars = 0
            for entry in entries:
                line = f"- [{entry.confidence:.1f}] {entry.title}: {entry.content[:200]}"
                if tier_chars + len(line) > budget_chars:
                    break
                tier_lines.append(line)
                tier_chars += len(line)

            if tier_lines:
                header = f"## {tier.title()} Knowledge ({len(tier_lines)}/{len(entries)})"
                section = header + "\n" + "\n".join(tier_lines)
                sections.append(section)
                chars_used += len(section)

        if not sections:
            return ""

        return "# Compiled Wiki\n\n" + "\n\n".join(sections)

    def reinforce(self, entry_id: str) -> bool:
        """Reinforce an entry (mark as useful/accessed)."""
        entry = self._store.get(entry_id)
        if not entry:
            return False
        entry.access_count += 1
        entry.reinforcement_count += 1
        entry.confidence = min(1.0, entry.confidence + 0.05)
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self._store.add(entry)
        return True

    def weaken(self, entry_id: str, amount: float = 0.1) -> bool:
        """Weaken an entry's confidence (contradicted or unhelpful)."""
        entry = self._store.get(entry_id)
        if not entry:
            return False
        entry.confidence = max(0.0, entry.confidence - amount)
        entry.updated_at = datetime.now(timezone.utc).isoformat()
        self._store.add(entry)
        return True

    def export_markdown(self, output_dir: Path | None = None) -> int:
        """Export all wiki entries as Obsidian-compatible markdown files."""
        out = output_dir or (self._store._dir / "pages")
        out.mkdir(parents=True, exist_ok=True)

        count = 0
        for tier in TIERS:
            tier_dir = out / tier
            tier_dir.mkdir(exist_ok=True)
            for entry in self._store.load_tier(tier):
                path = tier_dir / f"{entry.id}.md"
                path.write_text(entry.to_markdown(), encoding="utf-8")
                count += 1
        return count

    def generate_index(self) -> str:
        """Generate index.md — a catalog of all wiki pages."""
        lines = ["# Wiki Index\n"]
        for tier in reversed(TIERS):
            entries = self._store.load_tier(tier)
            if not entries:
                continue
            lines.append(f"\n## {tier.title()} ({len(entries)} entries)\n")
            entries.sort(key=lambda e: e.confidence, reverse=True)
            for e in entries[:50]:  # cap at 50 per tier in index
                tags = f" `{'` `'.join(e.tags)}`" if e.tags else ""
                lines.append(
                    f"- [[{e.id}]] **{e.title}** "
                    f"(confidence: {e.confidence:.2f}, sources: {len(e.sources)}){tags}"
                )
        return "\n".join(lines)
