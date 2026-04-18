"""Compaction Shield — the agent's "continuous self".

Maintains a structured session_essence that survives compaction.
After each turn, analyzes conversation to update decisions/progress/stances/key_data/todos.
Heuristic extraction by default; LLM-powered when available.

This is the HEART of the Agent OS kernel.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

from caveman.errors import CavemanError
from caveman.paths import SESSIONS_DIR
from caveman.engines.project_identity import (
    ProjectIdentity, ProjectIdentityStore, detect_project_from_messages,
)

logger = logging.getLogger(__name__)

try:
    import yaml
except ImportError:
    yaml = None


class ShieldError(CavemanError):
    """Shield-related error."""
    pass


@dataclass
class SessionEssence:
    """Structured session state that survives compaction."""
    session_id: str
    decisions: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    stances: list[str] = field(default_factory=list)
    key_data: dict[str, Any] = field(default_factory=dict)
    open_todos: list[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.now)
    turn_count: int = 0
    task: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["updated_at"] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionEssence:
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge(self, other: SessionEssence) -> None:
        """Merge another essence into this one (append + dedup)."""
        for item in other.decisions:
            if item not in self.decisions:
                self.decisions.append(item)
        for item in other.progress:
            if item not in self.progress:
                self.progress.append(item)
        for item in other.stances:
            if item not in self.stances:
                self.stances.append(item)
        self.key_data.update(other.key_data)
        for item in other.open_todos:
            if item not in self.open_todos:
                self.open_todos.append(item)

    @property
    def summary(self) -> str:
        """Human-readable summary for prompt injection."""
        parts = [f"Session: {self.session_id} (turn {self.turn_count})"]
        if self.task:
            parts.append(f"Task: {self.task}")
        if self.decisions:
            parts.append("Decisions:\n" + "\n".join(f"  - {d}" for d in self.decisions[-10:]))
        if self.progress:
            parts.append("Progress:\n" + "\n".join(f"  - {p}" for p in self.progress[-10:]))
        if self.stances:
            parts.append("Stances:\n" + "\n".join(f"  - {s}" for s in self.stances[-5:]))
        if self.key_data:
            parts.append("Key Data:\n" + "\n".join(f"  {k}: {v}" for k, v in list(self.key_data.items())[-10:]))
        if self.open_todos:
            parts.append("Open TODOs:\n" + "\n".join(f"  - {t}" for t in self.open_todos[-10:]))
        return "\n".join(parts)

    @property
    def narrative_summary(self) -> str:
        """Human-readable narrative summary for Recall prompt injection."""
        parts = []
        if self.task:
            parts.append(f"You were working on: {self.task}.")
        if self.decisions:
            joined = "; ".join(self.decisions[-5:])
            parts.append(f"You decided: {joined}.")
        if self.progress:
            joined = "; ".join(self.progress[-5:])
            parts.append(f"Completed: {joined}.")
        if self.open_todos:
            joined = "; ".join(self.open_todos[-5:])
            parts.append(f"Still TODO: {joined}.")
        if self.key_data:
            items = [f"{k}={v}" for k, v in list(self.key_data.items())[-5:]]
            parts.append(f"Key context: {', '.join(items)}.")
        if self.stances:
            joined = "; ".join(self.stances[-3:])
            parts.append(f"Stances: {joined}.")
        if not parts:
            return f"Session {self.session_id} (turn {self.turn_count}), no details captured."
        header = f"Session {self.session_id} (turn {self.turn_count})"
        return f"{header}: {' '.join(parts)}"


# ── Heuristic patterns ──

_DECISION_PATTERNS = [
    re.compile(r"(?:decided|chose|choosing|will use|going with|picked|selected|opted for)\s+(.{10,80})", re.I),
    re.compile(r"(?:decision|choice):\s*(.{10,80})", re.I),
    re.compile(r"(?:let'?s|we'?ll|I'?ll)\s+(use|go with|pick|choose)\s+(.{10,60})", re.I),
]

_PROGRESS_PATTERNS = [
    re.compile(r"(?:done|completed|finished|created|built|implemented|fixed|resolved|wrote|added)\s+(.{10,80})", re.I),
    re.compile(r"(?:✅|✓|☑)\s*(.{10,80})", re.I),
]

_TODO_PATTERNS = [
    re.compile(r"(?:TODO|FIXME|HACK|XXX)[\s:]+(.{10,80})", re.I),
    re.compile(r"(?:need to|still need|should|must|have to|next step)\s+(.{10,80})", re.I),
    re.compile(r"(?:⬜|☐|❌)\s*(.{10,80})", re.I),
]

_KEY_DATA_PATTERNS = [
    (re.compile(r"(?:path|file|dir(?:ectory)?)\s*[:=]\s*([~/\w.\-/]+)", re.I), "path"),
    (re.compile(r"(?:created|modified|edited|in)\s+(/[\w.\-/]+\.(?:py|js|ts|yaml|yml|json|toml|md|sql|sh))", re.I), "file"),
    (re.compile(r"(https?://\S{10,80})", re.I), "url"),
    (re.compile(r"(?:port|PORT)\s*[:=]\s*(\d{2,5})", re.I), "port"),
    (re.compile(r"(?:version|v)\s*[:=]?\s*(\d+\.\d+(?:\.\d+)?)", re.I), "version"),
    (re.compile(r"commit\s+([0-9a-f]{7,40})", re.I), "commit"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:ms|seconds?|s)\s+(?:for|per|at)", re.I), "perf"),
    (re.compile(r"(?:all\s+)?(\d+)\s+tests?\s+pass", re.I), "tests_passed"),
]


class CompactionShield:
    """Maintains session essence — the agent's continuous self."""

    def __init__(
        self,
        session_id: str | None = None,
        store_dir: Path | None = None,
        llm_fn: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._store_dir = Path(store_dir) if store_dir else SESSIONS_DIR
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._llm_fn = llm_fn
        self._essence = SessionEssence(session_id=self._session_id)
        self._project_store = ProjectIdentityStore()
        self._active_project: ProjectIdentity | None = None
        self._last_processed: int = 0  # incremental update offset

    @property
    def essence(self) -> SessionEssence:
        return self._essence

    @property
    def active_project(self) -> ProjectIdentity | None:
        return self._active_project

    async def update(self, messages: list[dict[str, Any]], task: str = "") -> SessionEssence:
        """Update essence from conversation. LLM if available, else heuristic.

        Incremental: only processes messages added since last update.
        """
        self._essence.turn_count = len([m for m in messages if m.get("role") == "assistant"])
        if task:
            self._essence.task = task

        # Incremental: only analyze new messages
        new_messages = messages[self._last_processed:]
        self._last_processed = len(messages)

        if not new_messages:
            return self._essence

        if self._llm_fn:
            try:
                extracted = await self._extract_with_llm(new_messages, task)
                self._essence.merge(extracted)
            except Exception as e:
                logger.warning("Shield LLM extraction failed, falling back to heuristic: %s", e)
                extracted = self._extract_heuristic(new_messages)
                self._essence.merge(extracted)
        else:
            extracted = self._extract_heuristic(new_messages)
            self._essence.merge(extracted)

        self._essence.updated_at = datetime.now()

        # Auto-detect and update project identity
        self._update_project_identity(messages)

        return self._essence

    async def save(self) -> Path:
        """Persist essence to disk as YAML (atomic write). Also saves active project identity."""
        if yaml is None:
            raise ShieldError("pyyaml required for Shield persistence")
        path = self._store_dir / f"{self._session_id}.yaml"
        tmp_path = path.with_suffix('.yaml.tmp')
        tmp_path.write_text(
            yaml.dump(self._essence.to_dict(), allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        tmp_path.rename(path)  # Atomic on POSIX
        logger.debug("Shield saved essence to %s", path)

        # Auto-save project identity if detected
        if self._active_project:
            self._active_project.session_count += 1
            self._project_store.save(self._active_project)
            logger.debug("Shield saved project identity: %s", self._active_project.name)

        return path

    @classmethod
    async def load(cls, session_id: str, store_dir: Path | None = None) -> SessionEssence | None:
        """Load essence from disk. Returns None if not found."""
        if yaml is None:
            return None
        d = Path(store_dir) if store_dir else SESSIONS_DIR
        path = d / f"{session_id}.yaml"
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return SessionEssence.from_dict(data) if data else None
        except Exception as e:
            logger.warning("Failed to load essence %s: %s", path, e)
            return None

    @classmethod
    async def load_latest(cls, store_dir: Path | None = None) -> SessionEssence | None:
        """Load the most recent session essence."""
        d = Path(store_dir) if store_dir else SESSIONS_DIR
        if not d.exists():
            return None
        yamls = sorted(d.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not yamls:
            return None
        try:
            if yaml is None:
                return None
            data = yaml.safe_load(yamls[0].read_text(encoding="utf-8"))
            return SessionEssence.from_dict(data) if data else None
        except Exception as e:
            logger.warning("Failed to load latest essence: %s", e)
            return None

    # ── Project Identity ──

    def _update_project_identity(self, messages: list[dict[str, Any]]) -> None:
        """Detect and update project identity from conversation."""
        detected = detect_project_from_messages(messages)
        if not detected:
            return

        # Try to load existing identity
        existing = self._project_store.load(detected.name)
        if existing:
            existing.merge_update(detected)
            self._active_project = existing
        else:
            self._active_project = detected

    def set_project(self, identity: ProjectIdentity) -> None:
        """Explicitly set the active project (e.g., from CLI or config)."""
        self._active_project = identity

    # ── Extraction methods ──

    def _extract_heuristic(self, messages: list[dict[str, Any]]) -> SessionEssence:
        """Rule-based extraction from conversation messages."""
        extracted = SessionEssence(session_id=self._session_id)

        for msg in messages:
            text = _get_message_text(msg)
            if not text or len(text) < 10:
                continue

            role = msg.get("role", "")

            # Extract decisions (from assistant messages)
            if role == "assistant":
                for pattern in _DECISION_PATTERNS:
                    for match in pattern.finditer(text):
                        decision = match.group(1).strip()[:120]
                        if decision and decision not in extracted.decisions:
                            extracted.decisions.append(decision)

            # Extract progress (from assistant messages)
            if role == "assistant":
                for pattern in _PROGRESS_PATTERNS:
                    for match in pattern.finditer(text):
                        progress = match.group(1).strip()[:120]
                        if progress and progress not in extracted.progress:
                            extracted.progress.append(progress)

            # Extract TODOs (from any message)
            for pattern in _TODO_PATTERNS:
                for match in pattern.finditer(text):
                    todo = match.group(1).strip()[:120]
                    if todo and todo not in extracted.open_todos:
                        extracted.open_todos.append(todo)

            # Extract key data (from any message)
            for pattern, label in _KEY_DATA_PATTERNS:
                for match in pattern.finditer(text):
                    value = match.group(1).strip()
                    key = f"{label}_{len(extracted.key_data)}"
                    extracted.key_data[key] = value

        # Cap lists to prevent unbounded growth
        extracted.decisions = extracted.decisions[-20:]
        extracted.progress = extracted.progress[-20:]
        extracted.open_todos = extracted.open_todos[-20:]
        extracted.stances = extracted.stances[-10:]

        return extracted

    async def _extract_with_llm(self, messages: list[dict[str, Any]], task: str) -> SessionEssence:
        """LLM-powered structured extraction."""
        assert self._llm_fn is not None

        recent = messages[-20:] if len(messages) > 20 else messages
        conv_text = "\n".join(
            f"[{m.get('role', '?')}]: {_get_message_text(m)[:300]}" for m in recent
        )

        prompt = f"""Analyze this conversation and extract structured session state.

Task: {task}

Conversation:
{conv_text}

Extract into these categories:
- decisions: Key decisions made (what was chosen and why)
- progress: What has been completed
- stances: Current positions or opinions held
- key_data: Important data points (paths, IDs, URLs, versions, names)
- open_todos: Unfinished items or next steps

Rules:
1. Be specific and actionable
2. Each item should be self-contained
3. Skip trivial information
4. key_data should be a flat dict of name:value pairs

Respond as JSON:
{{"decisions": ["..."], "progress": ["..."], "stances": ["..."], "key_data": {{"name": "value"}}, "open_todos": ["..."]}}"""

        response = await self._llm_fn(prompt)

        # Parse JSON (handles code fences, preamble, nested braces)
        from caveman.utils import parse_json_from_llm
        data = parse_json_from_llm(response, expect="object")
        if data and isinstance(data, dict):
            return SessionEssence(
                session_id=self._session_id,
                decisions=data.get("decisions", [])[:20],
                progress=data.get("progress", [])[:20],
                stances=data.get("stances", [])[:10],
                key_data=data.get("key_data", {}),
                open_todos=data.get("open_todos", [])[:20],
            )

        raise ValueError("Could not parse LLM response as JSON")


def _get_message_text(msg: dict[str, Any]) -> str:
    """Extract text content from a message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    return str(content)
