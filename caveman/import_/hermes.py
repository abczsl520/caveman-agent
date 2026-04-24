"""Hermes importer — § (section sign) delimited memories + nested skills."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from caveman.aio import aio_exists, aio_mkdir, aio_write_text
from .base import (
    BaseImporter, ImportItem, ImportManifest, ImportResult,
    infer_type, write_import_log,
)

logger = logging.getLogger(__name__)


def split_by_section_sign(content: str) -> list[str]:
    """Split content by §  (section sign) delimiter, not ## headers."""
    parts = content.split("\n§\n")
    return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 10]


def _skill_relative_path(skills_dir: Path, skill_md: Path) -> str:
    """Compute the skill's relative directory path under skills_dir.

    Hermes skills can be nested (e.g. software-development/systematic-debugging/SKILL.md).
    We preserve the full relative path so nested skills don't collide.

    Examples:
        skills_dir=/home/user/.hermes/skills
        skill_md=  /home/user/.hermes/skills/game-quality-gates/SKILL.md
        → "game-quality-gates"

        skill_md=  /home/user/.hermes/skills/software-development/systematic-debugging/SKILL.md
        → "software-development/systematic-debugging"
    """
    rel = skill_md.parent.relative_to(skills_dir)
    return str(rel)


class HermesImporter(BaseImporter):
    """Import from ~/.hermes/ — § delimited memories, nested skills, config."""

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

        # Skills — rglob to catch nested structures like
        # software-development/systematic-debugging/SKILL.md
        skills_dir = self.root / "skills"
        if skills_dir.is_dir():
            for skill_md in sorted(skills_dir.rglob("SKILL.md")):
                content = self._read_safe(skill_md)
                if content:
                    manifest.items.append(ImportItem(
                        source_path=skill_md, target_type="skill", content=content,
                    ))

        # Skill references — also import references/*.md alongside SKILL.md
        # These contain detailed knowledge (API endpoints, checklists, etc.)
        if skills_dir.is_dir():
            for ref_md in sorted(skills_dir.rglob("references/*.md")):
                content = self._read_safe(ref_md)
                if content:
                    manifest.items.append(ImportItem(
                        source_path=ref_md, target_type="skill_reference",
                        memory_type=MemoryType.PROCEDURAL, content=content,
                    ))

        # OpenClaw synced memory (if Hermes has openclaw-memory/ from cross-sync)
        oc_mem = self.root / "openclaw-memory"
        if oc_mem.is_dir():
            self._scan_openclaw_synced_memory(oc_mem, manifest)

        # Config
        config_path = self.root / "config.yaml"
        if config_path.is_file():
            content = self._read_safe(config_path)
            if content:
                manifest.items.append(ImportItem(
                    source_path=config_path, target_type="config", content=content,
                ))

        return manifest

    def _scan_openclaw_synced_memory(self, oc_mem: Path, manifest: ImportManifest) -> None:
        """Scan openclaw-memory/ synced into Hermes (methodology files, lessons, etc.)."""
        from .base import split_markdown_sections

        for md in sorted(oc_mem.rglob("*.md")):
            content = self._read_safe(md)
            if not content:
                continue
            sections = split_markdown_sections(content)
            for section in sections:
                mem_type = infer_type(section, md)
                secret_warn = self._scan_secrets(section)
                item = ImportItem(
                    source_path=md, target_type="memory",
                    memory_type=mem_type, content=section,
                )
                if secret_warn:
                    item.skip_reason = secret_warn
                    item.content = ""
                manifest.items.append(item)

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

    async def _handle_item(
        self, item: ImportItem, result: "ImportResult", memory_manager: Any,
    ) -> None:
        """Handle skill, skill_reference, and config imports."""
        from .config_merger import ConfigMerger

        if item.target_type == "skill":
            skills_dir = self.root / "skills"
            rel_path = _skill_relative_path(skills_dir, item.source_path)
            target = self.caveman_home / "skills" / rel_path / "SKILL.md"
            if not self.dry_run:
                await aio_mkdir(target.parent, parents=True, exist_ok=True)
                if not await aio_exists(target):
                    await aio_write_text(target, item.content, encoding="utf-8")
            result.imported += 1
        elif item.target_type == "skill_reference":
            # Store skill references as procedural memory for vector search
            if memory_manager and not self.dry_run:
                skill_name = item.source_path.parent.parent.name
                ref_name = item.source_path.stem
                await memory_manager.store(
                    content=item.content,
                    memory_type=MemoryType.PROCEDURAL,
                    metadata={
                        "source": "import:hermes-skill-ref",
                        "skill": skill_name,
                        "reference": ref_name,
                        "path": str(item.source_path),
                    },
                    trusted=self.include_secrets,
                )
            result.imported += 1
        elif item.target_type == "config":
            if not self.dry_run:
                merger = ConfigMerger(self.caveman_home)
                merger.merge_hermes_yaml(item.content)
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
