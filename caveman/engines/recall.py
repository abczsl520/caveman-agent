"""Recall Engine — restore context from previous sessions.

On new session startup, reads the most recent session_essence + relevant memories
and injects them into the system prompt via PromptBuilder layers.

The Recall Engine is the complement to the Shield:
  Shield: saves state during/after a session
  Recall: restores state at the start of a new session

Upgraded in Round 14: priority-based injection with token budgets.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from caveman.engines.shield import CompactionShield, SessionEssence
from caveman.engines.project_identity import ProjectIdentity, ProjectIdentityStore
from caveman.paths import SESSIONS_DIR

logger = logging.getLogger(__name__)


def _format_age(created_at: datetime) -> str:
    """Format memory age as human-readable string."""
    now = datetime.now()
    if created_at.tzinfo:
        from datetime import timezone
        now = datetime.now(timezone.utc)
    delta = now - created_at
    days = delta.days
    if days == 0:
        hours = delta.seconds // 3600
        return f"{hours}h ago" if hours > 0 else "just now"
    if days == 1:
        return "1d ago"
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    return f"{months}mo ago"


@dataclass
class RecallResult:
    """What Recall restored."""
    essences_loaded: int = 0
    memories_loaded: int = 0
    projects_loaded: int = 0
    total_tokens_est: int = 0
    essence_text: str = ""
    memory_text: str = ""
    project_text: str = ""

    @property
    def has_context(self) -> bool:
        return bool(self.essence_text or self.memory_text or self.project_text)

    def as_prompt_text(self) -> str:
        """Combined text for prompt injection."""
        parts = []
        if self.project_text:
            parts.append(self.project_text)  # Project identity first (highest priority)
        if self.essence_text:
            parts.append(self.essence_text)
        if self.memory_text:
            parts.append(self.memory_text)
        return "\n\n".join(parts)


class RecallEngine:
    """Restore context from previous sessions into the current prompt."""

    def __init__(
        self,
        sessions_dir: Path | None = None,
        memory_manager: Any | None = None,
        retrieval_log: Any | None = None,
        project_store: ProjectIdentityStore | None = None,
        max_essences: int = 3,
        max_memories: int = 10,
        essence_budget: int = 4_000,
        memory_budget: int = 3_000,
    ) -> None:
        self._sessions_dir = sessions_dir or SESSIONS_DIR
        self._memory_manager = memory_manager
        self._retrieval_log = retrieval_log
        # If custom sessions_dir but no project_store, use a sibling "projects" dir
        if project_store is not None:
            self._project_store = project_store
        elif sessions_dir is not None:
            self._project_store = ProjectIdentityStore(sessions_dir.parent / "projects")
        else:
            self._project_store = ProjectIdentityStore()
        self._max_essences = max_essences
        self._max_memories = max_memories
        self._essence_budget = essence_budget
        self._memory_budget = memory_budget

    async def restore(self, task: str = "") -> str:
        """Build a context restoration block for prompt injection.

        Returns a formatted string containing:
        1. Latest session essence(s) — decisions, progress, todos
        2. Relevant memories from memory store (if available)

        Backward-compatible: returns string. Use restore_structured()
        for RecallResult with metadata.
        """
        result = await self.restore_structured(task)
        return result.as_prompt_text()

    def _format_essences(self, essences) -> tuple[str, int]:
        """Format session essences within token budget. Returns (text, count)."""
        from caveman.utils import estimate_tokens
        parts = ["## Previous Session Context"]
        tokens_used = 0
        count = 0

        for i, essence in enumerate(essences):
            text = essence.narrative_summary
            text_tokens = estimate_tokens(text)
            if tokens_used + text_tokens > self._essence_budget:
                remaining = self._essence_budget - tokens_used
                if remaining > 50:
                    text = text[:remaining * 3] + "\n... (truncated)"
                else:
                    break
            label = "Most Recent Session" if i == 0 else f"Earlier Session ({i})"
            parts.append(f"### {label}")
            parts.append(text)
            tokens_used += text_tokens
            count += 1

        return "\n\n".join(parts), count

    @staticmethod
    def _synthesize_query(task: str, essences) -> str:
        """Build recall query — from task or synthesized from essences."""
        if task:
            return task
        if not essences:
            return ""
        latest = essences[0]
        parts = []
        if latest.task:
            parts.append(latest.task)
        if latest.decisions:
            parts.extend(latest.decisions[:3])
        if latest.open_todos:
            parts.extend(latest.open_todos[:3])
        return " ".join(parts)

    async def _recall_memories(self, query: str) -> tuple[str, int]:
        """Recall and format memories within budget. Returns (text, count)."""
        from caveman.utils import estimate_tokens
        try:
            memories = await self._memory_manager.recall(query, top_k=self._max_memories)
        except Exception as e:
            logger.warning("Recall memory search failed: %s", e)
            return "", 0

        if not memories:
            return "", 0

        lines = ["## Relevant Memories"]
        tokens_used = 0
        count = 0

        for mem in memories:
            age = _format_age(mem.created_at)
            trust = mem.metadata.get("trust_score", 0.5)
            trust_label = "★" if trust >= 0.7 else "☆" if trust >= 0.4 else "○"
            line = f"- [{mem.memory_type.value}] {trust_label} {mem.content} ({age})"
            line_tokens = estimate_tokens(line)
            if tokens_used + line_tokens > self._memory_budget:
                remaining = self._memory_budget - tokens_used
                if remaining > 30:
                    lines.append(line[:remaining * 3] + "...")
                    count += 1
                break
            lines.append(line)
            tokens_used += line_tokens
            count += 1

        if self._retrieval_log:
            try:
                self._retrieval_log.log_search(
                    query=query, results=[(1.0, m) for m in memories], source="recall")
            except Exception as e:
                logger.debug("Suppressed in recall: %s", e)

        return "\n".join(lines), count

    async def restore_structured(self, task: str = "") -> RecallResult:
        """Structured restore with metadata for PromptBuilder integration."""
        result = RecallResult()

        # Load essences first (needed for project detection)
        essences = await self._load_recent_essences()

        # 1. Load project identity (highest priority — never lost)
        project = self._detect_active_project(essences, task)
        if project:
            result.project_text = project.prompt_text
            result.projects_loaded = 1

        # 2. Load recent session essences
        if essences:
            result.essence_text, result.essences_loaded = self._format_essences(essences)

        # 2. Recall relevant memories
        if self._memory_manager:
            query = self._synthesize_query(task, essences)
            if query:
                result.memory_text, result.memories_loaded = await self._recall_memories(query)

        from caveman.utils import estimate_tokens
        result.total_tokens_est = estimate_tokens(
            result.project_text + result.essence_text + result.memory_text
        )

        return result

    def _detect_active_project(
        self,
        essences: list[SessionEssence],
        task: str = "",
    ) -> ProjectIdentity | None:
        """Detect active project from essences, task, or stored projects."""
        # Strategy 1: Check if task mentions a known project
        if task:
            for proj in self._project_store.list_projects():
                if proj.name.lower() in task.lower():
                    return proj

        # Strategy 2: Check essence key_data for project paths
        for essence in essences:
            for key, value in essence.key_data.items():
                if "path" in key and isinstance(value, str):
                    proj = self._project_store.load_by_path(value)
                    if proj:
                        return proj

        # Strategy 3: Check essence task field (word-boundary matching)
        for essence in essences:
            if essence.task:
                for proj in self._project_store.list_projects():
                    if re.search(rf'\b{re.escape(proj.name.lower())}\b', essence.task.lower()):
                        return proj

        # Strategy 4: Return most recently updated project (only if we have session context)
        if essences:
            projects = self._project_store.list_projects()
            if projects:
                projects.sort(key=lambda p: p.updated_at or "", reverse=True)
                if projects[0].session_count > 0:
                    return projects[0]

        return None

    async def _load_recent_essences(self) -> list[SessionEssence]:
        """Load the N most recent session essences."""
        if not self._sessions_dir.exists():
            return []

        try:
            import yaml
        except ImportError:
            return []

        # P0 #3 fix: sort by updated_at field, not filesystem mtime
        # mtime changes on rsync/git/cp, updated_at is the real session time
        essences_with_time: list[tuple[str, SessionEssence]] = []
        for path in self._sessions_dir.glob("*.yaml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if data:
                    essence = SessionEssence.from_dict(data)
                    sort_key = essence.updated_at or ""
                    essences_with_time.append((sort_key, essence))
            except Exception as e:
                logger.warning("Failed to load essence %s: %s", path, e)

        essences_with_time.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in essences_with_time[:self._max_essences]]

    async def has_previous_sessions(self) -> bool:
        """Check if there are any previous session essences."""
        if not self._sessions_dir.exists():
            return False
        return any(self._sessions_dir.glob("*.yaml"))
