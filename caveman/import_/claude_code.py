"""Claude Code importer — settings.json + plans/*.md + project CLAUDE.md files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

from .base import (
    BaseImporter, ImportItem, ImportManifest, ImportResult,
    infer_type, split_markdown_sections,
)

logger = logging.getLogger(__name__)

# Common project directories where CLAUDE.md files live
_PROJECT_SEARCH_PATHS = (
    Path("~/projects").expanduser(),
    Path("~/code").expanduser(),
    Path("~/dev").expanduser(),
    Path("~/work").expanduser(),
    Path.cwd(),
)

# Max depth to search for CLAUDE.md (avoid scanning entire filesystem)
_MAX_DEPTH = 3


class ClaudeCodeImporter(BaseImporter):
    """Import from ~/.claude/ — settings, plans, and project CLAUDE.md files.

    PRD §8.8.4: "Claude Code CLAUDE.md → 解析 → 提取项目知识 → 写入"
    """

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

        # 1. Global settings
        settings = self.root / "settings.json"
        if settings.is_file():
            content = self._read_safe(settings)
            if content:
                manifest.items.append(ImportItem(
                    source_path=settings, target_type="config", content=content,
                ))

        # 2. Plans
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

        # 3. Project-level CLAUDE.md files (PRD §8.8.4)
        self._scan_project_claude_files(manifest)

        return manifest

    def _scan_project_claude_files(self, manifest: ImportManifest) -> None:
        """Find and parse CLAUDE.md files in project directories.

        These contain project-specific knowledge: conventions, architecture
        decisions, coding standards — high-value SEMANTIC memories.
        """
        seen_paths: set[Path] = set()

        for search_root in _PROJECT_SEARCH_PATHS:
            if not search_root.is_dir():
                continue
            # Bounded depth search to avoid scanning huge trees
            for claude_md in self._find_claude_md(search_root, depth=0):
                resolved = claude_md.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)

                content = self._read_safe(claude_md)
                if not content:
                    continue

                # Derive project name from parent directory
                project_name = claude_md.parent.name

                for section in split_markdown_sections(content):
                    # Use infer_type for nuanced classification, but default
                    # to SEMANTIC since CLAUDE.md is mostly project knowledge
                    mem_type = infer_type(section, claude_md)
                    secret_warn = self._scan_secrets(section)
                    item = ImportItem(
                        source_path=claude_md, target_type="memory",
                        memory_type=mem_type,
                        content=f"[Project: {project_name}] {section}",
                    )
                    if secret_warn:
                        item.skip_reason = secret_warn
                        item.content = ""
                    manifest.items.append(item)

        if seen_paths:
            logger.info("Found %d project CLAUDE.md files", len(seen_paths))

    def _find_claude_md(self, root: Path, depth: int) -> list[Path]:
        """Recursively find CLAUDE.md files up to _MAX_DEPTH."""
        results: list[Path] = []
        if depth > _MAX_DEPTH:
            return results

        claude_md = root / "CLAUDE.md"
        if claude_md.is_file():
            results.append(claude_md)

        if depth < _MAX_DEPTH:
            try:
                for child in sorted(root.iterdir()):
                    if child.is_dir() and not child.name.startswith("."):
                        # Skip common non-project dirs
                        if child.name in ("node_modules", "__pycache__", ".venv",
                                          "venv", "dist", "build", ".git"):
                            continue
                        results.extend(self._find_claude_md(child, depth + 1))
            except PermissionError:
                pass

        return results

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
