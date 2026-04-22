"""Codex importer — MEMORY.md + rollout_summaries/."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from .base import (
    BaseImporter, ImportItem, ImportManifest,
    split_markdown_sections,
)

logger = logging.getLogger(__name__)


class CodexImporter(BaseImporter):
    """Import from ~/.codex/ — MEMORY.md and rollout summaries."""

    def __init__(self, caveman_home: Path, dry_run: bool = True, include_secrets: bool = False) -> None:
        super().__init__(caveman_home, dry_run, include_secrets)
        self.root = Path("~/.codex").expanduser()

    @property
    def source_name(self) -> str:
        return "Codex"

    def detect(self) -> bool:
        return self.root.is_dir()

    def scan(self) -> ImportManifest:
        manifest = ImportManifest(source="codex")
        if not self.detect():
            return manifest

        memory_md = self.root / "MEMORY.md"
        if memory_md.is_file():
            content = self._read_safe(memory_md)
            if content:
                for section in split_markdown_sections(content):
                    manifest.items.append(ImportItem(
                        source_path=memory_md, target_type="memory",
                        memory_type=MemoryType.SEMANTIC, content=section,
                    ))

        rollout_dir = self.root / "rollout_summaries"
        if rollout_dir.is_dir():
            for md in sorted(rollout_dir.glob("*.md")):
                content = self._read_safe(md)
                if not content:
                    continue
                for section in split_markdown_sections(content):
                    manifest.items.append(ImportItem(
                        source_path=md, target_type="memory",
                        memory_type=MemoryType.EPISODIC, content=section,
                    ))

        return manifest


    def _read_safe(self, path: Path) -> str:
        try:
            content = path.read_text(encoding="utf-8")
            return content if content.strip() else ""
        except Exception as e:
            logger.debug("Failed to read %s: %s", path, e)
            return ""
