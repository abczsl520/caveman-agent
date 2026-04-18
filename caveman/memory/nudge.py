"""Memory nudge — background extraction of memories from conversations.

Every N turns, a background "nudge" reviews the conversation and extracts
valuable information into long-term memory. Inspired by Hermes's
_spawn_background_review and Claude Code's 4-type memory system.

Principles:
  - Non-blocking: runs as background task, never delays the main conversation
  - PRD §5.2 source types: user/feedback/project/reference
  - Cognitive storage types: episodic/semantic/procedural/working
  - "Also confirm": don't just record discoveries, also reinforce known truths
  - Memory Drift aware: check for conflicts before writing
  - Dedup-aware: Jaccard similarity check before writing (FR-103)
"""
from __future__ import annotations
import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from .types import MemoryType, MemoryEntry, MemorySource, SOURCE_TO_TYPE, TYPE_MAP, resolve_memory_type, get_turn_text
from .drift import DriftDetector
from .retrieval import tokenize, jaccard_similarity

logger = logging.getLogger(__name__)

# Jaccard threshold for considering a candidate as duplicate of existing memory
_DEDUP_THRESHOLD = 0.6


class MemoryNudge:
    """Background memory extractor.

    Usage:
        nudge = MemoryNudge(memory_manager, llm_fn)
        # Call periodically during conversation:
        asyncio.create_task(nudge.run(conversation_turns))
    """

    def __init__(
        self,
        memory_manager,  # MemoryManager instance
        llm_fn=None,     # async fn(prompt: str) -> str
        interval: int = 10,  # nudge every N turns
        first_nudge: int = 3,  # first nudge after N turns (cold start)
    ):
        self.memory = memory_manager
        self.llm_fn = llm_fn
        self.interval = interval
        self.first_nudge = first_nudge
        self.drift_detector = DriftDetector(memory_manager)
        self._last_nudge_turn = 0
        self._nudge_count = 0
        self._lock = asyncio.Lock()

    def should_nudge(self, turn_count: int) -> bool:
        """Check if it's time for a nudge."""
        if turn_count < self.first_nudge:
            return False
        if self._nudge_count == 0 and turn_count >= self.first_nudge:
            return True  # First nudge: earlier for cold start
        return (turn_count - self._last_nudge_turn) >= self.interval

    async def run(self, turns: list[dict[str, Any]], task: str = "") -> list[MemoryEntry]:
        """Run nudge: analyze conversation → dedup → extract memories."""
        async with self._lock:
            self._nudge_count += 1
            self._last_nudge_turn = len(turns)

            candidates = await self._extract_candidates(turns, task)
            candidates = await self._dedup_candidates(candidates)

            created: list[MemoryEntry] = []
            for candidate in candidates:
                entry = await self._process_candidate(candidate)
                if entry:
                    created.append(entry)

            logger.info("Nudge #%d: extracted %d memories", self._nudge_count, len(created))
            return created

    async def _dedup_candidates(
        self, candidates: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Remove candidates that duplicate existing memories (Jaccard > threshold).

        When a candidate is a near-duplicate of an existing memory, instead of
        just skipping it, we boost the existing memory's trust (confirmation signal).
        """
        if not candidates:
            return []

        # Search for similar existing memories using FTS5 (not just recent 50)
        # This prevents re-storing knowledge from months ago
        existing: list[MemoryEntry] = []
        seen_ids: set[str] = set()
        for candidate in candidates:
            try:
                results = await self.memory.recall(candidate["content"], top_k=5)
                for r in results:
                    if r.id not in seen_ids:
                        existing.append(r)
                        seen_ids.add(r.id)
            except Exception as e:
                logger.debug('Nudge recall failed: %s', e)
        # Also include recent as fallback (catches non-FTS-matchable content)
        for r in self.memory.recent(limit=20):
            if r.id not in seen_ids:
                existing.append(r)
                seen_ids.add(r.id)
        existing_tokens = [tokenize(e.content) for e in existing]

        unique: list[dict[str, str]] = []
        unique_tokens: list[set[str]] = []

        for candidate in candidates:
            c_tokens = tokenize(candidate["content"])

            # Check against existing memories
            is_dup = False
            for i, et in enumerate(existing_tokens):
                if jaccard_similarity(c_tokens, et) >= _DEDUP_THRESHOLD:
                    logger.debug(
                        "Nudge dedup: '%s' confirms existing memory, boosting trust",
                        candidate["content"][:60],
                    )
                    try:
                        await self.memory.update_metadata(existing[i].id, {
                            "confirmed_count": existing[i].metadata.get("confirmed_count", 0) + 1,
                        })
                        backend = getattr(self.memory, '_backend', None)
                        if backend and hasattr(backend, 'mark_helpful'):
                            await backend.mark_helpful(existing[i].id, helpful=True)
                    except Exception as e:
                        logger.debug("Suppressed in nudge: %s", e)
                    is_dup = True
                    break

            if is_dup:
                continue

            # Check against other candidates in this batch
            for ut in unique_tokens:
                if jaccard_similarity(c_tokens, ut) >= _DEDUP_THRESHOLD:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(candidate)
                unique_tokens.append(c_tokens)

        if len(candidates) != len(unique):
            logger.info(
                "Nudge dedup: %d → %d candidates (removed %d duplicates)",
                len(candidates), len(unique), len(candidates) - len(unique),
            )
        return unique

    async def _process_candidate(self, candidate: dict[str, str]) -> MemoryEntry | None:
        """Process a single candidate: drift check → store."""
        drift_result = await self.drift_detector.check(candidate["content"])
        if drift_result:
            return await self._handle_drift(candidate, drift_result)

        mem_type = resolve_memory_type(candidate["type"])
        source_label = candidate["type"]  # Preserve PRD source label
        mid = await self.memory.store(
            candidate["content"], mem_type,
            metadata={
                "source": "nudge",
                "source_type": source_label,  # PRD §5.2 traceability
                "nudge_number": self._nudge_count,
            },
        )
        return MemoryEntry(
            id=mid, content=candidate["content"],
            memory_type=mem_type, created_at=datetime.now(),
        )

    async def _handle_drift(
        self, candidate: dict[str, str], drift_result: dict,
    ) -> None:
        """Handle drift detection result (contradiction or supersession)."""
        conflict = drift_result["entry"]
        drift_type = drift_result["drift_type"]

        if drift_type == "contradiction":
            logger.warning(
                "TRUE CONTRADICTION: new='%s' vs old='%s'",
                candidate["content"][:50], conflict.content[:50],
            )
            metadata = {
                "source": "nudge", "supersedes": conflict.id,
                "drift_type": "contradiction",
                "conflict_with": conflict.content[:100],
                "nudge_number": self._nudge_count, "needs_review": True,
            }
        else:  # supersession
            logger.info(
                "Knowledge update: new='%s' supersedes old='%s'",
                candidate["content"][:50], conflict.content[:50],
            )
            metadata = {
                "source": "nudge", "supersedes": conflict.id,
                "drift_type": "supersession",
                "nudge_number": self._nudge_count,
            }

        mem_type = resolve_memory_type(candidate["type"])
        await self.memory.store(candidate["content"], mem_type, metadata=metadata)
        stale_meta = {
            "superseded_by": candidate["content"][:100],
            "superseded_at": datetime.now().isoformat(),
        }
        await self.memory.update_metadata(conflict.id, stale_meta)
        return None

    async def _extract_candidates(
        self, turns: list[dict], task: str
    ) -> list[dict[str, str]]:
        """Extract memory candidates from conversation."""
        if self.llm_fn:
            return await self._extract_with_llm(turns, task)
        return self._extract_heuristic(turns, task)

    async def _extract_with_llm(
        self, turns: list[dict], task: str
    ) -> list[dict[str, str]]:
        """LLM-powered memory extraction using PRD §5.2 source types."""
        recent = turns[-20:] if len(turns) > 20 else turns
        conv_text = "\n".join(
            f"[{t.get('role', '?')}]: {get_turn_text(t)[:300]}" for t in recent
        )

        # Include existing memory summary to avoid duplicate extraction
        existing_summary = ""
        try:
            recent_mems = self.memory.recent(limit=10)
            if recent_mems:
                existing_summary = "\n\nAlready stored memories (DO NOT re-extract these):\n"
                existing_summary += "\n".join(
                    f"- [{m.memory_type.value}] {m.content[:100]}" for m in recent_mems
                )
        except Exception as e:
            logger.debug('Recent memory fetch failed: %s', e)

        prompt = f"""Analyze this conversation and extract valuable memories.

Task: {task}

Conversation:
{conv_text}
{existing_summary}

Extract memories in these categories (PRD source types):
- user: User preferences, working style, likes/dislikes
- feedback: Task outcomes, corrections, what worked/didn't
- project: Project structure, dependencies, architecture decisions
- reference: How-to knowledge, patterns, API details, procedures

Also accept cognitive types for backward compatibility:
- episodic: What happened (maps to feedback)
- semantic: Facts learned (maps to project)
- procedural: How-to knowledge (maps to reference)

Rules:
1. Be specific and actionable, not vague
2. Include BOTH new discoveries AND confirmations of known things
3. Each memory should be self-contained (understandable without context)
4. Skip trivial/obvious information
5. Do NOT extract anything already in the "Already stored memories" list

Respond as JSON array:
[{{"type": "user|feedback|project|reference|episodic|semantic|procedural", "content": "..."}}]

If nothing worth remembering, respond: []"""

        try:
            response = await self.llm_fn(prompt)
            from caveman.utils import parse_json_from_llm
            candidates = parse_json_from_llm(response, expect="array")
            if candidates and isinstance(candidates, list):
                valid = []
                for c in candidates:
                    if isinstance(c, dict) and "type" in c and "content" in c:
                        if c["type"] in TYPE_MAP:
                            valid.append(c)
                return valid
        except Exception as e:
            logger.warning("LLM nudge extraction failed: %s", e)

        return self._extract_heuristic(turns, task)

    def _extract_heuristic(
        self, turns: list[dict], task: str
    ) -> list[dict[str, str]]:
        """Rule-based memory extraction (no LLM needed)."""
        scan = self._scan_turns(turns)
        return self._build_candidates(scan, task)

    @staticmethod
    def _scan_turns(turns: list[dict]) -> dict:
        """Scan conversation turns for extractable patterns."""
        import re
        paths: set[str] = set()
        errors: list[tuple[str, str]] = []
        decisions: list[str] = []
        facts: list[str] = []
        prefs: list[str] = []
        last_assistant = ""

        _PREF_PATTERNS = (
            r"(?:I (?:prefer|like|want|always|usually|never|don't like|hate))\s+(.{5,120})",
            r"(?:(?:please |pls )?(?:always|never|don't)\s+)(.{5,120})",
            r"(?:(?:use|write|code) (?:in |with ))(.{3,60})",
            r"(?:我(?:喜欢|偏好|习惯|总是|从不|不喜欢|讨厌|要求|希望))(.{3,80})",
            r"(?:(?:请|麻烦)?(?:总是|永远|一定要|不要|别|千万别))(.{3,80})",
        )
        _DECISION_PATTERNS = (
            r"(?:decided|chose|will use|going with|let's use)\s+(.{10,120})",
            r"(?:the (?:best|right|correct) (?:approach|way|solution) is)\s+(.{10,120})",
            # Chinese decision patterns
            r"(?:决定|选择|采用|使用)(.{5,100})",
            r"(?:最好的|正确的|合适的)(?:方案|方法|做法)(?:是)(.{5,100})",
        )
        _FACT_PATTERNS = (
            r"(?:this (?:means|indicates|shows|confirms))\s+(.{10,120})",
            r"(?:the (?:issue|problem|root cause|reason) (?:is|was))\s+(.{10,120})",
            r"(?:(?:it|this) (?:requires|needs|expects))\s+(.{10,120})",
            r"(?:(?:fixed|solved|resolved) (?:by|with|using))\s+(.{10,120})",
            r"(?:the (?:fix|solution|workaround) (?:is|was))\s+(.{10,120})",
            r"(?:(?:I|we) (?:added|created|implemented|wrote))\s+(.{10,120})",
            r"(?:(?:doesn't|does not|can't|cannot) (?:work|support|handle))\s+(.{10,120})",
            # Chinese fact patterns
            r"(?:根因|根本原因|问题|原因)(?:是|在于)(.{5,100})",
            r"(?:解决方案|修复方法|方案)(?:是|为)(.{5,100})",
            r"(?:发现|确认|证实)(.{5,100})",
            r"(?:需要|必须|要求)(.{5,100})",
            r"(?:不能|无法|不支持)(.{5,100})",
        )

        for turn in turns:
            text = get_turn_text(turn)
            role = turn.get("role", turn.get("from", ""))

            if role in ("user", "human"):
                for pat in _PREF_PATTERNS:
                    for m in re.finditer(pat, text, re.IGNORECASE):
                        pref = m.group(0).strip()[:150]
                        if not any(pref.lower() in e.lower() or e.lower() in pref.lower() for e in prefs):
                            prefs.append(pref)

            if role in ("assistant", "gpt", "function_call"):
                last_assistant = text
                for m in re.finditer(r'(?:/[\w./-]+){2,}', text):
                    p = m.group()
                    if len(p) > 5:
                        paths.add(p)
                for pat in _DECISION_PATTERNS:
                    for m in re.finditer(pat, text, re.IGNORECASE):
                        decisions.append(m.group(0).strip()[:150])
                for pat in _FACT_PATTERNS:
                    for m in re.finditer(pat, text, re.IGNORECASE):
                        facts.append(m.group(0).strip()[:150])

            if ("error" in text.lower() or "failed" in text.lower()) and last_assistant:
                errors.append((text[:150].strip(), last_assistant[:100].strip()))

        return {"paths": paths, "errors": errors, "decisions": decisions,
                "facts": facts, "prefs": prefs}

    @staticmethod
    def _build_candidates(scan: dict, task: str) -> list[dict[str, str]]:
        """Build memory candidates from scan results."""
        candidates: list[dict[str, str]] = []
        for pref in scan["prefs"][:5]:
            candidates.append({"type": "user", "content": f"User preference: {pref}"})
        if scan["paths"] and task:
            paths = sorted(scan["paths"])[:8]
            candidates.append({"type": "project",
                "content": f"Project files in '{task[:60]}': {', '.join(paths)}"})
        for d in scan["decisions"][:3]:
            candidates.append({"type": "reference", "content": d})
        for f in scan["facts"][:3]:
            candidates.append({"type": "project", "content": f})
        for error, ctx in scan["errors"][:2]:
            candidates.append({"type": "feedback",
                "content": f"When trying: {ctx}... Got error: {error}"})
        return candidates

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "nudge_count": self._nudge_count,
            "last_nudge_turn": self._last_nudge_turn,
            "interval": self.interval,
            "has_llm": self.llm_fn is not None,
        }
