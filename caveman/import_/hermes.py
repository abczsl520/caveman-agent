"""Hermes importer — § (section sign) delimited memories."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from .base import (
    BaseImporter, ImportItem, ImportManifest, ImportResult,
    infer_type, write_import_log,
)

logger = logging.getLogger(__name__)


def split_by_section_sign(content: str) -> list[str]:
    """Split content by §  (section sign) delimiter, not ## headers."""
    parts = content.split("\n§\n")
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 10]


class HermesImporter(BaseImporter):
    """Import from ~/.hermes/ — § delimited memories + config."""

    def __init__(self, caveman_home: Path, dry_run: bool = True, include_secrets: bool = False) -> None:
        super().__init__(caveman_home, dry_run, include_secrets)
        self.root = Path("~/.hermes").expanduser()

    @property
    def source_name(self) -> str:
        return "Hermes"

    def detect(self) -> bool:
        return self.root.is_dir()

    def scan(self) -> ImportManifest:
        manifest = ImportManifest(source="hermes")
        if not self.detect():
            return manifest

        memories_dir = self.root / "memories"
        if memories_dir.is_dir():
            self._scan_memory_file(memories_dir / "MEMORY.md", manifest, default_type=None)
            self._scan_memory_file(memories_dir / "USER.md", manifest, default_type=MemoryType.WORKING)

        # Skills
        skills_dir = self.root / "skills"
        if skills_dir.is_dir():
            for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
                content = self._read_safe(skill_md)
                if content:
                    manifest.items.append(ImportItem(
                        source_path=skill_md, target_type="skill", content=content,
                    ))

        # Config
        config_path = self.root / "config.yaml"
        if config_path.is_file():
            content = self._read_safe(config_path)
            if content:
                manifest.items.append(ImportItem(
                    source_path=config_path, target_type="config", content=content,
                ))

        return manifest

    def _scan_memory_file(
        self, path: Path, manifest: ImportManifest,
        default_type: MemoryType | None,
    ) -> None:
        if not path.is_file():
            return
        content = self._read_safe(path)
        if not content:
            return

        entries = split_by_section_sign(content)
        for entry_text in entries:
            mem_type = default_type or infer_type(entry_text, path)
            secret_warn = self._scan_secrets(entry_text)
            item = ImportItem(
                source_path=path, target_type="memory",
                memory_type=mem_type, content=entry_text,
            )
            if secret_warn:
                item.skip_reason = secret_warn
                item.content = ""
            manifest.items.append(item)

    async def execute(self, manifest: ImportManifest, memory_manager: Any) -> ImportResult:
        """Execute Hermes import."""
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
                                "source": "import:hermes",
                                "source_file": str(item.source_path),
                                "imported_at": datetime.now().isoformat(),
                            },
                            trusted=self.include_secrets,
                        )
                    result.imported += 1
                elif item.target_type == "skill":
                    skill_name = item.source_path.parent.name
                    target = self.caveman_home / "skills" / skill_name / "SKILL.md"
                    if not self.dry_run:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            target.write_text(item.content, encoding="utf-8")
                    result.imported += 1
                elif item.target_type == "config":
                    if not self.dry_run:
                        merger = ConfigMerger(self.caveman_home)
                        merger.merge_hermes_yaml(item.content)
                    result.imported += 1
            except Exception as e:
                result.failed += 1
                logger.warning("Hermes import failed: %s", e)

            result.files_processed += 1

        if not self.dry_run:
            write_import_log(self.caveman_home, {
                "source": "hermes", "imported": result.imported,
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
