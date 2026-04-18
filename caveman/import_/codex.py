"""Codex importer — MEMORY.md + rollout_summaries/."""
from __future__ import annotations

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

    async def execute(self, manifest: ImportManifest, memory_manager: Any) -> ImportResult:
        """Execute Codex import."""
        from .dedup import ImportDedup

        result = ImportResult()
        dedup = ImportDedup(memory_manager)

        for item in manifest.items:
            if item.skip_reason:
                result.skipped += 1
                continue
            try:
                if dedup.is_duplicate(item.content):
                    result.duplicates += 1
                    continue
                if not self.dry_run:
                    await memory_manager.store(
                        item.content, item.memory_type,
                        metadata={
                            "source": "import:codex",
                            "source_file": str(item.source_path),
                            "imported_at": datetime.now().isoformat(),
                        },
                        trusted=self.include_secrets,
                    )
                result.imported += 1
            except Exception as e:
                result.failed += 1
                logger.warning("Codex import failed: %s", e)
            result.files_processed += 1

        if not self.dry_run:
            write_import_log(self.caveman_home, {
                "source": "codex", "imported": result.imported,
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
