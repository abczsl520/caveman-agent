"""Workspace file reader — load SOUL.md, USER.md, etc. into prompt layers.

Scans workspace directories for persona/context files and injects them
into the system prompt with proper layering (SOUL > USER > MEMORY > AGENTS > rest).
"""
from __future__ import annotations

import logging
from pathlib import Path

from caveman.paths import CAVEMAN_HOME
from caveman.agent.prompt_contract import PromptTopic, detect_topics

__all__ = [
    "WORKSPACE_FILES",
    "WORKSPACE_PATHS",
    "WorkspaceLoader",
]


logger = logging.getLogger(__name__)

WORKSPACE_FILES = (
    "SOUL.md", "USER.md", "MEMORY.md", "AGENTS.md",
    "HEARTBEAT.md", "TOOLS.md",
)

WORKSPACE_PATHS = [
    CAVEMAN_HOME / "workspace",
    Path("~/.openclaw/workspace").expanduser(),
]

# Layer ordering — lower index = higher priority in prompt
_LAYER_ORDER = {
    "SOUL.md": 0,
    "USER.md": 1,
    "MEMORY.md": 2,
    "AGENTS.md": 3,
}


class WorkspaceLoader:
    """Load workspace files and inject into system prompt layers."""

    def __init__(self, paths: list[Path] | None = None) -> None:
        self.paths = paths or list(WORKSPACE_PATHS)

    @staticmethod
    def _strip_workspace_contract_violations(content: str, filename: str) -> str:
        """Remove lines owned by non-workspace prompt authorities.

        Workspace files are user-controlled and may contain useful notes like
        group-chat formatting preferences.  Those rules must not be injected
        through the workspace layer because response_style is the sole
        authority for format/layout instructions.  Drop only the offending
        lines instead of suppressing the contract warning globally.
        """
        kept: list[str] = []
        for line in content.splitlines():
            topics = detect_topics(line)
            if PromptTopic.FORMAT_LAYOUT in topics:
                logger.debug("Workspace: dropped format/layout line from %s: %s", filename, line[:120])
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def load(self) -> dict[str, str]:
        """Scan workspace dirs, return {filename: content}.

        First path found wins (caveman > openclaw).
        Only loads memory/*.md from the primary (caveman) workspace,
        not from external workspaces like OpenClaw.
        """
        found: dict[str, str] = {}
        # Track provenance: which file came from where
        self._provenance: dict[str, str] = {}
        primary_ws = None

        for ws_dir in self.paths:
            if not ws_dir.is_dir():
                continue
            if primary_ws is None:
                primary_ws = ws_dir

            for name in WORKSPACE_FILES:
                if name in found:
                    continue
                fp = ws_dir / name
                if fp.is_file():
                    try:
                        found[name] = self._strip_workspace_contract_violations(
                            fp.read_text(encoding="utf-8"), name
                        )
                        self._provenance[name] = str(fp)
                        logger.info("Workspace: %s → %s", name, fp)
                    except Exception:
                        logger.warning("Failed to read %s", fp, exc_info=True)

        # Only load memory/*.md from primary workspace (avoid loading
        # huge external memory dirs like OpenClaw's daily logs)
        if primary_ws:
            mem_dir = primary_ws / "memory"
            if mem_dir.is_dir():
                for md in sorted(mem_dir.glob("*.md")):
                    # Skip large files (>10KB) to avoid prompt bloat
                    if md.stat().st_size > 10240:
                        logger.debug("Skipping large memory file: %s (%d bytes)", md, md.stat().st_size)
                        continue
                    key = f"memory/{md.name}"
                    if key not in found:
                        try:
                            found[key] = self._strip_workspace_contract_violations(
                                md.read_text(encoding="utf-8"), key
                            )
                        except Exception:
                            logger.warning("Failed to read %s", md, exc_info=True)
        return found

    def build_prompt_layers(self) -> str:
        """Build prompt injection from workspace files.

        Layer 0: SOUL.md (persona)
        Layer 1: USER.md (user context)
        Layer 2: MEMORY.md (long-term memory)
        Layer 3: AGENTS.md (rules)
        Layer 4: Other files
        """
        files = self.load()
        if not files:
            return ""

        layers: list[tuple[int, str, str]] = []
        for name, content in files.items():
            order = _LAYER_ORDER.get(name, 99)
            layers.append((order, name, content.strip()))

        layers.sort(key=lambda t: t[0])

        parts: list[str] = []
        for _, name, content in layers:
            if content:
                parts.append(f"<!-- {name} -->\n{content}")

        return "\n\n".join(parts)
