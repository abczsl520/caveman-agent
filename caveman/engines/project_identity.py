"""Project Identity — persistent project-level context that survives compaction.

Unlike SessionEssence (per-session, volatile), ProjectIdentity is:
  - Per-project, not per-session
  - Rarely changes (mission, principles are stable)
  - Auto-detected from conversation context
  - Auto-loaded by Recall, auto-saved by Shield
  - Injected into Wiki as procedural tier (never expires)

This is what makes Caveman better than OpenClaw at context preservation:
OpenClaw relies on rules telling the agent to read files.
Caveman's memory system automatically protects and restores project identity.
"""
from __future__ import annotations

__all__ = ["ProjectIdentity", "ProjectIdentityStore"]

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.paths import PROJECTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class ProjectIdentity:
    """Persistent project-level identity — the "who am I" that never gets lost."""

    name: str  # "caveman"
    path: str = ""  # "~/projects/caveman"
    mission: str = ""  # "做出比 OpenClaw/Hermes 更好用的 Agent"
    principles: list[str] = field(default_factory=list)
    current_phase: str = ""  # "Round 20 — 从巨头学平台能力"
    tech_stack: list[str] = field(default_factory=list)
    key_identifiers: dict[str, str] = field(default_factory=dict)
    updated_at: str = ""
    session_count: int = 0  # how many sessions have touched this project

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "mission": self.mission,
            "principles": self.principles,
            "current_phase": self.current_phase,
            "tech_stack": self.tech_stack,
            "key_identifiers": self.key_identifiers,
            "updated_at": self.updated_at,
            "session_count": self.session_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectIdentity:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def merge_update(self, other: ProjectIdentity) -> None:
        """Merge updates from another identity (newer wins for scalars)."""
        if other.mission and other.mission != self.mission:
            self.mission = other.mission
        if other.current_phase:
            self.current_phase = other.current_phase
        for p in other.principles:
            if p not in self.principles:
                self.principles.append(p)
        for t in other.tech_stack:
            if t not in self.tech_stack:
                self.tech_stack.append(t)
        self.key_identifiers.update(other.key_identifiers)
        self.updated_at = datetime.now().isoformat()

    @property
    def prompt_text(self) -> str:
        """Formatted text for system prompt injection."""
        parts = [f"## Active Project: {self.name}"]
        if self.path:
            parts.append(f"Path: {self.path}")
        if self.mission:
            parts.append(f"Mission: {self.mission}")
        if self.principles:
            parts.append("Principles:\n" + "\n".join(f"  - {p}" for p in self.principles))
        if self.current_phase:
            parts.append(f"Current Phase: {self.current_phase}")
        if self.tech_stack:
            parts.append(f"Tech: {', '.join(self.tech_stack)}")
        if self.key_identifiers:
            parts.append("Key IDs:\n" + "\n".join(
                f"  {k}: {v}" for k, v in self.key_identifiers.items()
            ))
        return "\n".join(parts)


class ProjectIdentityStore:
    """Persistent storage for project identities.

    Layout:
        ~/.caveman/projects/
        ├── caveman.json
        ├── my-app.json
        └── ...
    """

    def __init__(self, store_dir: Path | None = None) -> None:
        self._dir = store_dir or PROJECTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, identity: ProjectIdentity) -> Path:
        """Save a project identity."""
        identity.updated_at = datetime.now().isoformat()
        path = self._dir / f"{_safe_filename(identity.name)}.json"
        path.write_text(
            json.dumps(identity.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Saved project identity: %s", identity.name)
        return path

    def load(self, name: str) -> ProjectIdentity | None:
        """Load a project identity by name."""
        path = self._dir / f"{_safe_filename(name)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ProjectIdentity.from_dict(data)
        except Exception as e:
            logger.warning("Failed to load project identity %s: %s", name, e)
            return None

    def list_projects(self) -> list[ProjectIdentity]:
        """List all known project identities."""
        projects = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                projects.append(ProjectIdentity.from_dict(data))
            except Exception:
                continue
        return projects

    def load_by_path(self, project_path: str) -> ProjectIdentity | None:
        """Find a project identity by its path."""
        normalized = str(Path(project_path).expanduser().resolve())
        for proj in self.list_projects():
            if proj.path:
                proj_normalized = str(Path(proj.path).expanduser().resolve())
                if proj_normalized == normalized:
                    return proj
        return None

    def delete(self, name: str) -> bool:
        """Delete a project identity."""
        path = self._dir / f"{_safe_filename(name)}.json"
        if path.exists():
            path.unlink()
            return True
        return False


# --- Project detection heuristics ---

_PROJECT_NAME_PATTERNS = [
    r"(?:project|项目)[:\s]+[\"']?(\w[\w\-]{1,40})[\"']?",
    r"(?:working on|做|开发)\s+[\"']?(\w[\w\-]{1,40})\b",
]

_PROJECT_PATH_PATTERNS = [
    r"(?:path|路径|目录)[:\s=]+([~/\w.\-/]{5,80})",
    r"(?:cd|chdir)\s+([~/\w.\-/]{5,80})",
    r"(?:repo|repository)\s+(?:at|in)\s+([~/\w.\-/]{5,80})",
]

_MISSION_PATTERNS = [
    r"(?:goal|mission|目标|愿景|核心目标)[:\s]+([^.!?\n]{10,200})",
    r"(?:aim(?:ing)? to|want to|trying to)\s+([^.!?\n]{10,150})",
]

_PHASE_PATTERNS = [
    r"(?:phase|round|阶段|轮次)\s*(\d+\w*)[:\s]*(.{5,100})?",
    r"(?:currently|正在|当前)\s+(.{10,100})",
]


def detect_project_from_messages(
    messages: list[dict[str, Any]],
) -> ProjectIdentity | None:
    """Try to detect project identity from conversation messages.

    Returns None if no clear project context is found.
    """
    name = ""
    path = ""
    mission = ""
    phase = ""
    tech: list[str] = []

    for msg in messages:
        text = msg.get("content", "")
        if not isinstance(text, str):
            continue

        # Detect project name
        if not name:
            for pattern in _PROJECT_NAME_PATTERNS:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    name = m.group(1).strip()
                    break

        # Detect path
        if not path:
            for pattern in _PROJECT_PATH_PATTERNS:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    path = m.group(1).strip()
                    break

        # Detect mission
        if not mission:
            for pattern in _MISSION_PATTERNS:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    mission = m.group(1).strip()[:200]
                    break

        # Detect phase
        if not phase:
            for pattern in _PHASE_PATTERNS:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    groups = [g for g in m.groups() if g]
                    phase = " ".join(groups).strip()[:100]
                    break

    # Need at least a name to create an identity
    if not name:
        # Try to infer from path
        if path:
            name = Path(path).name
        else:
            return None

    return ProjectIdentity(
        name=name,
        path=path,
        mission=mission,
        current_phase=phase,
        tech_stack=tech,
    )


def _safe_filename(name: str) -> str:
    """Convert project name to safe filename."""
    safe = re.sub(r"[^\w\-.]", "_", name.lower().strip())
    return safe[:50] or "unnamed"
