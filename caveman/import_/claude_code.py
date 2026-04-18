"""Claude Code importer — settings.json + plans/*.md."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from .base import (
    BaseImporter, ImportItem, ImportManifest, ImportResult,
    split_markdown_sections, write_import_log,
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

    async def execute(self, manifest: ImportManifest, memory_manager: Any) -> ImportResult:
        """Execute Claude Code import."""
        from .dedup import ImportDedup
        from .config_merger import ConfigMerger

        result = ImportResult()
        dedup = ImportDedup(memory_manager)

        for item in manifest.items:
            if item.skip_reason:
                result.skipped += 1
                continue
            try:
                if item.target_type == "memory":
                    if dedup.is_duplicate(item.content):
                        result.duplicates += 1
                        continue
                    if not self.dry_run:
                        await memory_manager.store(
                            item.content, item.memory_type,
                            metadata={
                                "source": "import:claude-code",
                                "source_file": str(item.source_path),
                                "imported_at": datetime.now().isoformat(),
                            },
                            trusted=self.include_secrets,
                        )
                    result.imported += 1
                elif item.target_type == "config":
                    if not self.dry_run:
                        merger = ConfigMerger(self.caveman_home)
                        merger.merge_claude_settings(item.content)
                    result.imported += 1
            except Exception as e:
                result.failed += 1
                logger.warning("Claude Code import failed: %s", e)
            result.files_processed += 1

        if not self.dry_run:
            write_import_log(self.caveman_home, {
                "source": "claude-code", "imported": result.imported,
                "duplicates": result.duplicates,
            })
        return result

    def _read_safe(self, path: Path) -> str:
        try:
            content = path.read_text(encoding="utf-8")
            return content if content.strip() else ""
        except Exception as e:
            logger.debug("Failed to read %s: %s", path, e)
            return ""
