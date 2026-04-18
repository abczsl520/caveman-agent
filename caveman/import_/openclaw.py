"""OpenClaw importer — full recursive scan of ~/.openclaw/."""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from .base import (
    BaseImporter, ImportItem, ImportManifest, ImportResult,
    infer_type, split_markdown_sections, write_import_log,
)

logger = logging.getLogger(__name__)

_WORKSPACE_FILES = (
    "SOUL.md", "USER.md", "MEMORY.md", "AGENTS.md",
    "HEARTBEAT.md", "TOOLS.md", "IDENTITY.md",
)

# Subdirectory → memory type mapping
_DIR_TYPE_MAP: dict[str, MemoryType] = {
    "projects": MemoryType.SEMANTIC,
    "seo": MemoryType.SEMANTIC,
    "studies": MemoryType.SEMANTIC,
    "agent-profiles": MemoryType.SEMANTIC,
    "lessons": MemoryType.PROCEDURAL,
    "sop-references": MemoryType.PROCEDURAL,
    "archive": MemoryType.EPISODIC,
}


class OpenClawImporter(BaseImporter):
    """Import from ~/.openclaw/ — workspace, memory, agents, learnings."""

    def __init__(self, caveman_home: Path, dry_run: bool = True, include_secrets: bool = False) -> None:
        super().__init__(caveman_home, dry_run, include_secrets)
        self.root = Path("~/.openclaw").expanduser()

    @property
    def source_name(self) -> str:
        return "OpenClaw"

    def detect(self) -> bool:
        return self.root.is_dir()

    def scan(self) -> ImportManifest:
        """Scan all OpenClaw data: workspace files, memory tree, agents, learnings."""
        manifest = ImportManifest(source="openclaw")
        if not self.detect():
            return manifest

        ws = self.root / "workspace"
        if ws.is_dir():
            self._scan_workspace_files(ws, manifest)
            self._scan_agents(ws / "agents", manifest)
            self._scan_skills(ws / "skills", manifest)
            self._scan_scripts(ws / "scripts", manifest)
            self._scan_memory_tree(ws / "memory", manifest)

        self._scan_learnings(self.root / "workspace" / ".learnings", manifest)
        self._scan_agent_state(self.root / "workspace" / ".agent-state" / "done", manifest)
        self._scan_config(self.root / "openclaw.json", manifest)
        self._scan_cron(self.root / "cron" / "jobs.json", manifest)

        return manifest

    def _scan_workspace_files(self, ws: Path, manifest: ImportManifest) -> None:
        for name in _WORKSPACE_FILES:
            fp = ws / name
            if fp.is_file():
                content = self._read_safe(fp)
                if content:
                    manifest.items.append(ImportItem(
                        source_path=fp, target_type="workspace", content=content,
                    ))

    def _scan_agents(self, agents_dir: Path, manifest: ImportManifest) -> None:
        if not agents_dir.is_dir():
            return
        for md in sorted(agents_dir.glob("*.md")):
            content = self._read_safe(md)
            if content:
                manifest.items.append(ImportItem(
                    source_path=md, target_type="workspace", content=content,
                ))

    def _scan_skills(self, skills_dir: Path, manifest: ImportManifest) -> None:
        if not skills_dir.is_dir():
            return
        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            content = self._read_safe(skill_md)
            if content:
                manifest.items.append(ImportItem(
                    source_path=skill_md, target_type="skill", content=content,
                ))

    def _scan_scripts(self, scripts_dir: Path, manifest: ImportManifest) -> None:
        if not scripts_dir.is_dir():
            return
        for sh in sorted(scripts_dir.glob("*.sh")):
            content = self._read_safe(sh)
            if content:
                manifest.items.append(ImportItem(
                    source_path=sh, target_type="workspace", content=content,
                ))

    def _scan_memory_tree(self, mem_dir: Path, manifest: ImportManifest) -> None:
        """Recursively scan memory/ and all subdirectories."""
        if not mem_dir.is_dir():
            return
        for md in sorted(mem_dir.rglob("*.md")):
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
                    item.content = ""  # Don't carry secret content
                manifest.items.append(item)

    def _scan_learnings(self, learnings_dir: Path, manifest: ImportManifest) -> None:
        if not learnings_dir.is_dir():
            return
        for md in sorted(learnings_dir.glob("*.md")):
            content = self._read_safe(md)
            if not content:
                continue
            name_lower = md.name.lower()
            mem_type = MemoryType.EPISODIC if "error" in name_lower else MemoryType.PROCEDURAL
            for section in split_markdown_sections(content):
                manifest.items.append(ImportItem(
                    source_path=md, target_type="memory",
                    memory_type=mem_type, content=section,
                ))

    def _scan_agent_state(self, done_dir: Path, manifest: ImportManifest) -> None:
        if not done_dir.is_dir():
            return
        for jf in sorted(done_dir.glob("*.json")):
            content = self._read_safe(jf)
            if content:
                manifest.items.append(ImportItem(
                    source_path=jf, target_type="memory",
                    memory_type=MemoryType.EPISODIC, content=content,
                ))

    def _scan_config(self, config_path: Path, manifest: ImportManifest) -> None:
        if config_path.is_file():
            content = self._read_safe(config_path)
            if content:
                manifest.items.append(ImportItem(
                    source_path=config_path, target_type="config", content=content,
                ))

    def _scan_cron(self, cron_path: Path, manifest: ImportManifest) -> None:
        if cron_path.is_file():
            content = self._read_safe(cron_path)
            if content:
                manifest.items.append(ImportItem(
                    source_path=cron_path, target_type="cron", content=content,
                ))

    async def execute(self, manifest: ImportManifest, memory_manager: Any) -> ImportResult:
        """Execute import: store memories, copy workspace files, merge config."""
        from .dedup import ImportDedup
        from .config_merger import ConfigMerger

        result = ImportResult()
        dedup = ImportDedup(memory_manager)

        for item in manifest.items:
            if item.skip_reason:
                result.skipped += 1
                result.details.append(f"Skipped: {item.source_path.name} ({item.skip_reason})")
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
                                "source": "import:openclaw",
                                "source_file": str(item.source_path),
                                "imported_at": datetime.now().isoformat(),
                            },
                            trusted=self.include_secrets,
                        )
                    result.imported += 1

                elif item.target_type == "workspace":
                    self._copy_workspace_file(item, result)

                elif item.target_type == "skill":
                    self._copy_skill(item, result)

                elif item.target_type == "config":
                    if not self.dry_run:
                        merger = ConfigMerger(self.caveman_home)
                        merger.merge_openclaw_json(item.content)
                    result.imported += 1
                    result.details.append("Merged OpenClaw config")

                elif item.target_type == "cron":
                    self._copy_cron(item, result)

                else:
                    result.imported += 1

            except Exception as e:
                result.failed += 1
                result.details.append(f"Failed: {item.source_path.name}: {e}")
                logger.warning("Import failed for %s: %s", item.source_path, e)

            result.files_processed += 1

        if not self.dry_run:
            write_import_log(self.caveman_home, {
                "source": "openclaw", "imported": result.imported,
                "duplicates": result.duplicates, "skipped": result.skipped,
            })

        return result

    def _copy_workspace_file(self, item: ImportItem, result: ImportResult) -> None:
        ws_dir = self.caveman_home / "workspace"
        # Determine target name
        if "agents/" in str(item.source_path):
            target = ws_dir / "agents" / item.source_path.name
        elif "scripts/" in str(item.source_path):
            target = ws_dir / "scripts" / item.source_path.name
        else:
            target = ws_dir / item.source_path.name

        if not self.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_suffix(f".imported-from-openclaw{target.suffix}")
                shutil.copy2(item.source_path, backup)
                result.details.append(f"Backup: {target.name} → {backup.name}")
            else:
                target.write_text(item.content, encoding="utf-8")
        result.imported += 1

    def _copy_skill(self, item: ImportItem, result: ImportResult) -> None:
        skill_name = item.source_path.parent.name
        target = self.caveman_home / "skills" / skill_name / "SKILL.md"
        if not self.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(item.content, encoding="utf-8")
        result.imported += 1

    def _copy_cron(self, item: ImportItem, result: ImportResult) -> None:
        target = self.caveman_home / "cron" / "imported-jobs.json"
        if not self.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.content, encoding="utf-8")
        result.imported += 1

    def _read_safe(self, path: Path) -> str:
        """Read file safely, return empty string on failure."""
        try:
            content = path.read_text(encoding="utf-8")
            return content if content.strip() else ""
        except Exception as e:
            logger.debug("Failed to read %s: %s", path, e)
            return ""
