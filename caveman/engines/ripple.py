"""Ripple Engine — knowledge propagation on memory write.

When new knowledge is written to Memory Store, Ripple:
  1. Searches for related existing entries (semantic + keyword)
  2. Detects contradictions (same topic, different claims)
  3. Marks stale entries as superseded
  4. Creates cross-references between related entries
  5. Emits conflict notifications for user confirmation

Design principle: Write quality >> Retrieve sophistication.
Every write triggers a "ripple" that keeps the knowledge base consistent.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class RippleEffect:
    """Result of a single ripple propagation."""
    new_entry_id: str
    stale_marked: list[str] = field(default_factory=list)
    cross_refs_added: list[tuple[str, str]] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    notifications: list[str] = field(default_factory=list)

    @property
    def had_impact(self) -> bool:
        return bool(self.stale_marked or self.conflicts or self.cross_refs_added)

    @property
    def summary(self) -> str:
        parts = []
        if self.stale_marked:
            parts.append(f"{len(self.stale_marked)} stale")
        if self.cross_refs_added:
            parts.append(f"{len(self.cross_refs_added)} cross-refs")
        if self.conflicts:
            parts.append(f"{len(self.conflicts)} conflicts")
        return f"Ripple: {', '.join(parts)}" if parts else "Ripple: no impact"


# Patterns for extracting key entities from memory content
_IP_RE = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_PATH_RE = re.compile(r'(?:/[\w./-]+|~/[\w./-]+)')
_VERSION_RE = re.compile(r'\bv?\d+\.\d+(?:\.\d+)?\b')
_PORT_RE = re.compile(r'\bport\s+(\d{2,5})\b', re.IGNORECASE)


class RippleEngine:
    """Knowledge propagation engine.

    Usage:
        ripple = RippleEngine(memory_manager)
        effect = await ripple.propagate(new_entry)
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        llm_fn: Callable[[str], Awaitable[str]] | None = None,
        similarity_threshold: float = 0.3,
    ):
        self.memory = memory_manager
        self.llm_fn = llm_fn
        self.similarity_threshold = similarity_threshold

    async def propagate(self, new_entry: MemoryEntry) -> RippleEffect:
        """Propagate a new memory write through the knowledge base."""
        effect = RippleEffect(new_entry_id=new_entry.id)

        # Step 1: Find related entries
        related = self._find_related(new_entry)
        if not related:
            return effect

        # Step 2: Check for contradictions and stale entries
        for existing in related:
            contradiction = self._detect_contradiction(new_entry, existing)
            if contradiction:
                contradiction["_existing_entry"] = existing
                effect.conflicts.append(contradiction)
                # Mark old entry as stale (newer wins by default)
                await self._mark_stale(existing, new_entry, effect)
            else:
                # Add cross-reference
                await self._add_cross_ref(new_entry, existing, effect)

        # Step 3: LLM verification for ambiguous cases (if available)
        if self.llm_fn and effect.conflicts:
            await self._verify_conflicts_llm(effect)

        # Note: metadata changes are now persisted immediately via update_metadata()
        # in _mark_stale() and _add_cross_ref(), so no need for a bulk save() call.

        # Step 4: Generate notifications
        for conflict in effect.conflicts:
            if conflict.get("dismissed"):
                continue
            effect.notifications.append(
                f"Conflict: '{conflict['new'][:60]}' vs '{conflict['existing'][:60]}' "
                f"— {conflict.get('reason', 'possible contradiction')}"
            )

        if effect.had_impact:
            logger.info("Ripple for %s: %s", new_entry.id, effect.summary)

        return effect

    def _find_related(self, entry: MemoryEntry) -> list[MemoryEntry]:
        """Find entries related to the new one."""
        # Keyword search
        related = self.memory.search_sync(entry.content, limit=10)

        # Also search by extracted entities
        entities = self._extract_entities(entry.content)
        for entity in entities:
            entity_matches = self.memory.search_sync(entity, limit=5)
            for m in entity_matches:
                if m.id != entry.id and m not in related:
                    related.append(m)

        # Filter out self
        return [r for r in related if r.id != entry.id]

    def _extract_entities(self, text: str) -> list[str]:
        """Extract key entities (IPs, paths, versions) from text."""
        entities = []
        entities.extend(_IP_RE.findall(text))
        entities.extend(_PATH_RE.findall(text))
        entities.extend(_PORT_RE.findall(text))
        return entities[:10]  # Cap to avoid excessive searches

    def _detect_contradiction(
        self, new: MemoryEntry, existing: MemoryEntry
    ) -> dict | None:
        """Detect if two entries contradict each other."""
        new_text = new.content.lower()
        existing_text = existing.content.lower()

        # Same IP mentioned but different context
        new_ips = set(_IP_RE.findall(new.content))
        existing_ips = set(_IP_RE.findall(existing.content))

        # Shared entities with different values
        new_entities = set(self._extract_entities(new.content))
        existing_entities = set(self._extract_entities(existing.content))
        shared = new_entities & existing_entities

        if shared:
            # Same entity mentioned — check if claims differ
            # Heuristic: shared keywords but different IPs/paths/versions
            new_only = new_entities - existing_entities
            existing_only = existing_entities - shared
            if new_only and existing_only:
                # Different values for same topic
                shared_words = set(new_text.split()) & set(existing_text.split())
                topic_words = shared_words - {"the", "is", "a", "to", "of", "in", "on", "at", "and", "or"}
                if len(topic_words) >= 2:
                    return {
                        "new": new.content[:200],
                        "new_id": new.id,
                        "existing": existing.content[:200],
                        "existing_id": existing.id,
                        "shared_entities": list(shared)[:5],
                        "reason": f"Same topic ({', '.join(list(topic_words)[:3])}) but different values",
                    }

        # Migration/update patterns
        migration_patterns = [
            (r"migrated?\s+(?:to|from)", "migration"),
            (r"(?:changed?|updated?|moved?)\s+(?:to|from)", "update"),
            (r"(?:new|replaced?|switched?)\s+(?:to|with)", "replacement"),
        ]
        for pattern, reason in migration_patterns:
            if re.search(pattern, new_text) and shared:
                return {
                    "new": new.content[:200],
                    "new_id": new.id,
                    "existing": existing.content[:200],
                    "existing_id": existing.id,
                    "shared_entities": list(shared)[:5],
                    "reason": reason,
                }

        return None

    async def _mark_stale(
        self, old: MemoryEntry, new: MemoryEntry, effect: RippleEffect
    ) -> None:
        """Mark an existing entry as superseded by the new one.

        Persists metadata via update_metadata() — works for both SQLite and JSON.
        Previous implementation modified in-memory objects and relied on save(),
        which is a no-op in SQLite mode (P0 #4 fix).
        """
        stale_meta = {
            "superseded_by": new.id,
            "superseded_at": datetime.now().isoformat(),
            "superseded_content": new.content[:100],
        }
        old.metadata.update(stale_meta)  # keep in-memory copy consistent
        await self.memory.update_metadata(old.id, stale_meta)
        effect.stale_marked.append(old.id)

    async def _add_cross_ref(
        self, new: MemoryEntry, existing: MemoryEntry, effect: RippleEffect
    ) -> None:
        """Add bidirectional cross-references. Persists via update_metadata()."""
        # Add ref to new entry
        refs = new.metadata.setdefault("related", [])
        if existing.id not in refs:
            refs.append(existing.id)
            await self.memory.update_metadata(new.id, {"related": refs})

        # Add ref to existing entry
        existing_refs = existing.metadata.setdefault("related", [])
        if new.id not in existing_refs:
            existing_refs.append(new.id)
            await self.memory.update_metadata(existing.id, {"related": existing_refs})

        effect.cross_refs_added.append((new.id, existing.id))

    async def _verify_conflicts_llm(self, effect: RippleEffect) -> None:
        """Use LLM to verify if detected conflicts are real."""
        for conflict in effect.conflicts:
            prompt = f"""Are these two statements contradictory?

Statement A: {conflict['new']}
Statement B: {conflict['existing']}

Respond with:
- "yes" if they contradict (say different things about the same topic)
- "no" if they are compatible (different topics or complementary info)
- "update" if A is clearly a newer version of B

One word answer:"""

            try:
                response = await self.llm_fn(prompt)
                answer = response.strip().lower().split()[0] if response.strip() else "unknown"
                conflict["llm_verdict"] = answer
                if answer == "no":
                    # Remove from stale list — not actually contradictory
                    if conflict["existing_id"] in effect.stale_marked:
                        effect.stale_marked.remove(conflict["existing_id"])
                    # Undo stale metadata in DB (P0 #4 fix: persist the undo)
                    existing_entry = conflict.get("_existing_entry")
                    if existing_entry is not None:
                        existing_entry.metadata.pop("superseded_by", None)
                        existing_entry.metadata.pop("superseded_at", None)
                        existing_entry.metadata.pop("superseded_content", None)
                        # Write cleaned metadata back to storage
                        await self.memory.update_metadata(
                            existing_entry.id, existing_entry.metadata,
                        )
                    conflict["dismissed"] = True
            except Exception as e:
                logger.warning("LLM conflict verification failed: %s", e)
                conflict["llm_verdict"] = "error"

    async def propagate_trust(self, entry_id: str, trust_delta: float, decay: float = 0.3) -> int:
        """Propagate trust change to related memories.

        When memory A's trust changes by +0.08, related memories B,C get +0.024.
        This is the 'ripple' — confidence spreads through the knowledge graph.

        Returns number of memories affected.
        """
        if abs(trust_delta) < 0.01:
            return 0

        # Find related memories
        try:
            entries = self.memory.recent(limit=500)
            entry = next((e for e in entries if e.id == entry_id), None)
            if not entry:
                return 0
        except Exception:
            return 0

        related_ids = entry.metadata.get("related", [])
        if not related_ids:
            return 0

        ripple_delta = trust_delta * decay
        affected = 0

        backend = getattr(self.memory, '_backend', None)
        if not backend:
            return 0

        for rid in related_ids:
            try:
                conn = backend._conn
                row = conn.execute(
                    "SELECT trust_score FROM memories WHERE id = ?", (rid,)
                ).fetchone()
                if row:
                    new_trust = max(0.0, min(1.0, row[0] + ripple_delta))
                    conn.execute(
                        "UPDATE memories SET trust_score = ? WHERE id = ?",
                        (new_trust, rid),
                    )
                    affected += 1
            except Exception as e:
                logger.debug("Ripple trust propagation failed for %s: %s", rid, e)

        if affected:
            logger.info(
                "Ripple trust: %s delta=%.3f → %d related (×%.1f decay)",
                entry_id[:8], trust_delta, affected, decay,
            )
        return affected
