"""Nudge Phase 2 — LLM-powered memory refinement.

Takes raw Phase 1 memories and refines them:
  - Deduplication: merge near-identical entries
  - Conflict detection: flag contradictions with existing memories
  - Quality upgrade: rewrite vague entries into precise, actionable ones
  - Merge: combine related fragments into coherent entries

Runs as a batch job after Phase 1 extraction, not on every turn.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable

from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    """Result of Phase 2 refinement."""
    input_count: int = 0
    output_count: int = 0
    merged: int = 0
    conflicts: list[dict] = field(default_factory=list)
    removed_duplicates: int = 0
    refined_entries: list[MemoryEntry] = field(default_factory=list)
    llm_cost_est: float = 0.0  # estimated cost in dollars

    @property
    def summary(self) -> str:
        return (
            f"Phase 2: {self.input_count} → {self.output_count} memories "
            f"({self.removed_duplicates} deduped, {self.merged} merged, "
            f"{len(self.conflicts)} conflicts, ~${self.llm_cost_est:.4f})"
        )


class NudgeRefiner:
    """Phase 2 memory refinement engine.

    Usage:
        refiner = NudgeRefiner(memory_manager, llm_fn)
        result = await refiner.refine(raw_memories)
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        llm_fn: Callable[[str], Awaitable[str]] | None = None,
    ):
        self.memory = memory_manager
        self.llm_fn = llm_fn

    async def refine(
        self, raw_memories: list[MemoryEntry]
    ) -> RefinementResult:
        """Refine a batch of raw Phase 1 memories."""
        result = RefinementResult(input_count=len(raw_memories))

        if not raw_memories:
            return result

        if self.llm_fn:
            return await self._refine_with_llm(raw_memories, result)
        return self._refine_heuristic(raw_memories, result)

    def _refine_heuristic(
        self, raw: list[MemoryEntry], result: RefinementResult
    ) -> RefinementResult:
        """Heuristic refinement (no LLM cost)."""
        # Step 1: Exact/near dedup
        seen: dict[str, MemoryEntry] = {}
        unique: list[MemoryEntry] = []
        for mem in raw:
            key = _normalize(mem.content)
            if key in seen:
                result.removed_duplicates += 1
            else:
                seen[key] = mem
                unique.append(mem)

        # Step 2: Check conflicts with existing memories
        for mem in unique:
            existing = self.memory.search_sync(mem.content, limit=3)
            for ex in existing:
                if _is_likely_conflict(mem.content, ex.content):
                    result.conflicts.append({
                        "new_id": mem.id,
                        "new_content": mem.content[:100],
                        "existing_id": ex.id,
                        "existing_content": ex.content[:100],
                        "status": "conflict",
                    })

        result.output_count = len(unique)
        result.refined_entries = unique
        return result

    async def _refine_with_llm(
        self, raw: list[MemoryEntry], result: RefinementResult
    ) -> RefinementResult:
        """LLM-powered refinement."""
        # Build input for LLM
        raw_text = "\n".join(
            f"[{i}] ({m.memory_type.value}) {m.content}"
            for i, m in enumerate(raw)
        )

        # Get existing memories for conflict check
        existing_sample = self.memory.recent(limit=20)
        existing_text = "\n".join(
            f"[E{i}] ({m.memory_type.value}) {m.content[:150]}"
            for i, m in enumerate(existing_sample)
        )

        prompt = f"""You are a memory refinement engine. Process these raw memories.

RAW MEMORIES (just extracted):
{raw_text}

EXISTING MEMORIES (already stored):
{existing_text}

Tasks:
1. DEDUP: Remove duplicates (same info, different wording)
2. MERGE: Combine related fragments into single coherent entries
3. CONFLICT: Flag entries that contradict existing memories
4. REFINE: Rewrite vague entries to be specific and actionable

Respond as JSON:
{{
  "refined": [
    {{"content": "...", "type": "episodic|semantic|procedural|user", "action": "keep|merge|refine"}}
  ],
  "conflicts": [
    {{"new_index": 0, "existing_index": 0, "reason": "..."}}
  ],
  "removed_indices": [1, 3]
}}"""

        try:
            response = await self.llm_fn(prompt)
            # Estimate cost: ~input_tokens/1M * $3 + output_tokens/1M * $15
            input_tokens = len(prompt) // 4
            output_tokens = len(response) // 4
            result.llm_cost_est = (input_tokens * 3 + output_tokens * 15) / 1_000_000

            parsed = _parse_json(response)
            if not parsed:
                logger.warning("LLM refinement returned unparseable response, falling back")
                return self._refine_heuristic(raw, result)

            # Process refined entries
            refined_list = parsed.get("refined", [])
            for entry in refined_list:
                if "content" in entry and "type" in entry:
                    from caveman.memory.types import resolve_memory_type
                    mem_type = resolve_memory_type(entry["type"])
                    result.refined_entries.append(MemoryEntry(
                        id=f"refined_{len(result.refined_entries)}",
                        content=entry["content"],
                        memory_type=mem_type,
                        created_at=datetime.now(),
                    ))

            # Process conflicts
            for conflict in parsed.get("conflicts", []):
                result.conflicts.append({
                    "new_index": conflict.get("new_index"),
                    "existing_index": conflict.get("existing_index"),
                    "reason": conflict.get("reason", ""),
                    "status": "conflict",
                })

            # Count removals
            removed = parsed.get("removed_indices", [])
            result.removed_duplicates = len(removed)
            result.merged = sum(
                1 for e in refined_list if e.get("action") == "merge"
            )
            result.output_count = len(result.refined_entries)

        except Exception as e:
            logger.warning("LLM refinement failed: %s, falling back to heuristic", e)
            return self._refine_heuristic(raw, result)

        return result


def _normalize(text: str) -> str:
    """Normalize for dedup comparison."""
    import re
    # Lowercase, collapse whitespace, remove hyphens and punctuation
    text = text.strip().lower()
    text = re.sub(r'[-_]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text)


def _is_likely_conflict(new: str, existing: str) -> bool:
    """Heuristic: check if two memories likely contradict."""
    new_lower = new.lower()
    existing_lower = existing.lower()

    # Same topic but different values (IPs, versions, paths)
    import re
    new_ips = set(re.findall(r'\d+\.\d+\.\d+\.\d+', new_lower))
    existing_ips = set(re.findall(r'\d+\.\d+\.\d+\.\d+', existing_lower))

    # If both mention IPs but different ones, and share keywords
    if new_ips and existing_ips and new_ips != existing_ips:
        # Check for shared context words
        new_words = set(new_lower.split())
        existing_words = set(existing_lower.split())
        shared = new_words & existing_words - {"the", "is", "a", "to", "of", "in", "on", "at"}
        if len(shared) >= 3:
            return True

    return False


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None
