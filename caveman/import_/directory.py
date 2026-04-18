"""Directory importer — generic import from any directory of .md files."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import (
    BaseImporter, ImportItem, ImportManifest, ImportResult,
    infer_type, split_markdown_sections, write_import_log,
)

logger = logging.getLogger(__name__)


class DirectoryImporter(BaseImporter):
    """Import .md files from an arbitrary directory."""

    def __init__(self, caveman_home: Path, dry_run: bool = True,
                 directory: Path | None = None, include_secrets: bool = False) -> None:
        super().__init__(caveman_home, dry_run, include_secrets)
        self.directory = directory

    @property
    def source_name(self) -> str:
        return "Directory"

    def detect(self) -> bool:
        return self.directory is not None and self.directory.is_dir()

    def scan(self) -> ImportManifest:
        manifest = ImportManifest(source="directory")
        if not self.detect():
            return manifest

        for md in sorted(self.directory.rglob("*.md")):
            content = self._read_safe(md)
            if not content:
                continue
            for section in split_markdown_sections(content):
                secret_warn = self._scan_secrets(section)
                mem_type = infer_type(section, md)
                item = ImportItem(
                    source_path=md, target_type="memory",
                    memory_type=mem_type, content=section,
                )
                if secret_warn:
                    item.skip_reason = secret_warn
                    item.content = ""
                manifest.items.append(item)

        return manifest

    async def execute(self, manifest: ImportManifest, memory_manager: Any) -> ImportResult:
        """Execute directory import."""
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
                            "source": "import:directory",
                            "source_file": str(item.source_path),
                            "imported_at": datetime.now().isoformat(),
                        },
                        trusted=self.include_secrets,
                    )
                result.imported += 1
            except Exception as e:
                result.failed += 1
                logger.warning("Directory import failed: %s", e)
            result.files_processed += 1

        if not self.dry_run:
            write_import_log(self.caveman_home, {
                "source": "directory", "imported": result.imported,
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
