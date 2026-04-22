"""Claude Code importer — settings.json + plans/*.md."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from .base import (
    BaseImporter, ImportItem, ImportManifest, ImportResult,
    split_markdown_sections,
)

logger = logging.getLogger(__name__)


class ClaudeCodeImporter(BaseImporter):
    """Import from ~/.claude/ — settings and plans."""

    def __init__(self, caveman_home: Path, dry_run: bool = True, include_secrets: bool = False) -> None:
        super().__init__(caveman_home, dry_run, include_secrets)
        self.root = Path("~/.claude").expanduser()

    @property
    def source_name(self) -> str:
        return "Claude Code"

    def detect(self) -> bool:
        return self.root.is_dir()

    def scan(self) -> ImportManifest:
        manifest = ImportManifest(source="claude-code")
        if not self.detect():
            return manifest

        settings = self.root / "settings.json"
        if settings.is_file():
            content = self._read_safe(settings)
            if content:
                manifest.items.append(ImportItem(
                    source_path=settings, target_type="config", content=content,
                ))

        plans_dir = self.root / "plans"
        if plans_dir.is_dir():
            for md in sorted(plans_dir.glob("*.md")):
                content = self._read_safe(md)
                if not content:
                    continue
                for section in split_markdown_sections(content):
                    manifest.items.append(ImportItem(
                        source_path=md, target_type="memory",
                        memory_type=MemoryType.PROCEDURAL, content=section,
                    ))

        return manifest

    async def _handle_item(
        self, item: ImportItem, result: "ImportResult", memory_manager: Any,
    ) -> None:
        """Handle config imports."""
        from .config_merger import ConfigMerger

        if item.target_type == "config":
            if not self.dry_run:
                merger = ConfigMerger(self.caveman_home)
                merger.merge_claude_settings(item.content)
            result.imported += 1
        else:
            result.imported += 1

    def _read_safe(self, path: Path) -> str:
        try:
            content = path.read_text(encoding="utf-8")
            return content if content.strip() else ""
        except Exception as e:
            logger.debug("Failed to read %s: %s", path, e)
            return ""
