"""Tests for the import system — all importers, dedup, type inference, reports."""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from caveman.memory.types import MemoryType
from caveman.import_.base import (
    ImportItem, ImportManifest, ImportResult,
    infer_type, split_markdown_sections,
)
from caveman.import_.dedup import ImportDedup, content_hash
from caveman.import_.hermes import split_by_section_sign
from caveman.import_.report import (
    format_manifest_report, format_result_report, format_detect_report,
)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

class TestTypeInference:
    def test_lessons_path(self):
        p = Path("/home/user/.openclaw/workspace/memory/lessons/git.md")
        assert infer_type("some content", p) == MemoryType.PROCEDURAL

    def test_projects_path(self):
        p = Path("/home/user/.openclaw/workspace/memory/projects/foo.md")
        assert infer_type("some content", p) == MemoryType.SEMANTIC

    def test_date_filename(self):
        p = Path("/home/user/.openclaw/workspace/memory/2024-01-15.md")
        assert infer_type("some content", p) == MemoryType.EPISODIC

    def test_archive_path(self):
        p = Path("/home/user/.openclaw/workspace/memory/archive/old.md")
        assert infer_type("some content", p) == MemoryType.EPISODIC

    def test_sop_path(self):
        p = Path("/home/user/.openclaw/workspace/memory/sop-references/deploy.md")
        assert infer_type("some content", p) == MemoryType.PROCEDURAL

    def test_content_fallback_procedural(self):
        p = Path("/tmp/generic.md")
        assert infer_type("step 1: install the package", p) == MemoryType.PROCEDURAL

    def test_content_fallback_semantic(self):
        p = Path("/tmp/generic.md")
        assert infer_type("The API uses REST endpoints for data access", p) == MemoryType.SEMANTIC

    def test_seo_path(self):
        p = Path("/home/user/.openclaw/workspace/memory/seo/keywords.md")
        assert infer_type("some content", p) == MemoryType.SEMANTIC

    def test_studies_path(self):
        p = Path("/home/user/.openclaw/workspace/memory/studies/research.md")
        assert infer_type("some content", p) == MemoryType.SEMANTIC


# ---------------------------------------------------------------------------
# Markdown splitting
# ---------------------------------------------------------------------------

class TestMarkdownSplitting:
    def test_split_by_headers(self):
        content = "## Section A\nContent A\n\n## Section B\nContent B"
        sections = split_markdown_sections(content)
        assert len(sections) == 2
        assert "Section A" in sections[0]
        assert "Section B" in sections[1]

    def test_skip_short_sections(self):
        content = "## Hi\nX\n\n## Real Section\nThis is a real section with enough content to pass the threshold."
        sections = split_markdown_sections(content)
        assert len(sections) == 1
        assert "Real Section" in sections[0]

    def test_split_oversized_section(self):
        big = "## Big Section\n" + ("A" * 2000 + "\n\n") * 3
        sections = split_markdown_sections(big, max_chars=2500)
        assert len(sections) >= 2

    def test_no_headers(self):
        content = "Just a plain text file with enough content to be imported as a single entry."
        sections = split_markdown_sections(content)
        assert len(sections) == 1


# ---------------------------------------------------------------------------
# § (section sign) splitting — Hermes format
# ---------------------------------------------------------------------------

class TestSectionSignSplitting:
    def test_basic_split(self):
        content = "Memory entry one about Python.\n§\nMemory entry two about Rust."
        parts = split_by_section_sign(content)
        assert len(parts) == 2
        assert "Python" in parts[0]
        assert "Rust" in parts[1]

    def test_multiple_sections(self):
        content = "First entry here.\n§\nSecond entry here.\n§\nThird entry here."
        parts = split_by_section_sign(content)
        assert len(parts) == 3

    def test_skip_short_entries(self):
        content = "OK\n§\nThis is a real memory entry with enough content."
        parts = split_by_section_sign(content)
        assert len(parts) == 1

    def test_no_section_sign(self):
        content = "This is a single memory with no section signs at all."
        parts = split_by_section_sign(content)
        assert len(parts) == 1

    def test_preserves_content(self):
        entry = "User prefers dark mode and vim keybindings."
        content = f"{entry}\n§\nAnother entry about deployment steps."
        parts = split_by_section_sign(content)
        assert parts[0] == entry


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_first_seen_not_duplicate(self):
        dedup = ImportDedup(memory_manager=None)
        assert dedup.is_duplicate("unique content") is False

    def test_second_seen_is_duplicate(self):
        dedup = ImportDedup(memory_manager=None)
        dedup.is_duplicate("same content")
        assert dedup.is_duplicate("same content") is True

    def test_different_content_not_duplicate(self):
        dedup = ImportDedup(memory_manager=None)
        dedup.is_duplicate("content A")
        assert dedup.is_duplicate("content B") is False

    def test_content_hash_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_content_hash_strips_whitespace(self):
        h1 = content_hash("  hello  ")
        h2 = content_hash("hello")
        assert h1 == h2


# ---------------------------------------------------------------------------
# ImportItem / ImportManifest
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_import_item_auto_hash(self):
        item = ImportItem(source_path=Path("/tmp/test.md"), target_type="memory", content="hello world")
        assert item.content_hash
        assert len(item.content_hash) == 16

    def test_import_item_auto_size(self):
        item = ImportItem(source_path=Path("/tmp/test.md"), target_type="memory", content="hello")
        assert item.size_bytes == 5

    def test_import_item_preview(self):
        item = ImportItem(source_path=Path("/tmp/test.md"), target_type="memory", content="A" * 200)
        assert len(item.preview) == 100

    def test_manifest_summary(self):
        manifest = ImportManifest(source="test", items=[
            ImportItem(source_path=Path("/a.md"), target_type="memory", content="x" * 100),
            ImportItem(source_path=Path("/b.md"), target_type="memory", content="y" * 50, skip_reason="dup"),
        ])
        assert "test" in manifest.summary
        assert len(manifest.actionable) == 1
        assert len(manifest.skipped) == 1


# ---------------------------------------------------------------------------
# OpenClaw importer — detect, scan, deep recursion
# ---------------------------------------------------------------------------

class TestOpenClawImporter:
    def test_detect_missing(self, tmp_path):
        from caveman.import_.openclaw import OpenClawImporter
        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = tmp_path / "nonexistent"
        assert imp.detect() is False

    def test_detect_present(self, tmp_path):
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        root.mkdir()
        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        assert imp.detect() is True

    def test_scan_workspace_files(self, tmp_path):
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        ws = root / "workspace"
        ws.mkdir(parents=True)
        (ws / "SOUL.md").write_text("I am the soul of the agent, defining its core personality.", encoding="utf-8")
        (ws / "USER.md").write_text("User prefers dark mode and concise responses always.", encoding="utf-8")

        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        ws_items = [i for i in manifest.items if i.target_type == "workspace"]
        assert len(ws_items) == 2

    def test_scan_deep_memory_recursion(self, tmp_path):
        """memory/projects/seo-matrix/*.md should be found."""
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        deep = root / "workspace" / "memory" / "projects" / "seo-matrix"
        deep.mkdir(parents=True)
        (deep / "keywords.md").write_text("## Keywords\nSEO keyword research for the main product landing page.", encoding="utf-8")

        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) >= 1
        assert mem_items[0].memory_type == MemoryType.SEMANTIC

    def test_scan_learnings(self, tmp_path):
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        learn = root / "workspace" / ".learnings"
        learn.mkdir(parents=True)
        (learn / "LEARNINGS.md").write_text("## Git Rebase\nAlways rebase before merging to keep history clean.", encoding="utf-8")

        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) >= 1
        assert mem_items[0].memory_type == MemoryType.PROCEDURAL

    @pytest.mark.asyncio
    async def test_dry_run_no_writes(self, tmp_path):
        """dry-run should not create any files."""
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        ws = root / "workspace"
        ws.mkdir(parents=True)
        (ws / "SOUL.md").write_text("Soul content with enough text to pass the threshold check.", encoding="utf-8")

        caveman_home = tmp_path / "caveman"
        caveman_home.mkdir()
        imp = OpenClawImporter(caveman_home=caveman_home, dry_run=True)
        imp.root = root
        manifest = imp.scan()
        result = await imp.execute(manifest, memory_manager=None)
        # No workspace dir should be created
        assert not (caveman_home / "workspace" / "SOUL.md").exists()

    @pytest.mark.asyncio
    async def test_workspace_conflict_backup(self, tmp_path):
        """Existing workspace file should get .imported-from-openclaw backup."""
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        ws = root / "workspace"
        ws.mkdir(parents=True)
        (ws / "SOUL.md").write_text("New soul content from OpenClaw with enough text.", encoding="utf-8")

        caveman_home = tmp_path / "caveman"
        cws = caveman_home / "workspace"
        cws.mkdir(parents=True)
        (cws / "SOUL.md").write_text("Existing soul content that should not be overwritten.", encoding="utf-8")

        imp = OpenClawImporter(caveman_home=caveman_home, dry_run=False)
        imp.root = root
        manifest = imp.scan()
        result = await imp.execute(manifest, memory_manager=None)
        # Original should be untouched
        assert "Existing" in (cws / "SOUL.md").read_text()
        # Backup should exist
        backup = cws / "SOUL.imported-from-openclaw.md"
        assert backup.exists()


# ---------------------------------------------------------------------------
# Hermes importer
# ---------------------------------------------------------------------------

class TestHermesImporter:
    def test_detect_missing(self, tmp_path):
        from caveman.import_.hermes import HermesImporter
        imp = HermesImporter(caveman_home=tmp_path)
        imp.root = tmp_path / "nonexistent"
        assert imp.detect() is False

    def test_scan_section_sign_memories(self, tmp_path):
        from caveman.import_.hermes import HermesImporter
        root = tmp_path / "hermes"
        mem = root / "memories"
        mem.mkdir(parents=True)
        (mem / "MEMORY.md").write_text(
            "First memory about Python programming.\n§\n"
            "Second memory about Rust language features.",
            encoding="utf-8",
        )
        imp = HermesImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) == 2

    def test_user_md_gets_working_type(self, tmp_path):
        from caveman.import_.hermes import HermesImporter
        root = tmp_path / "hermes"
        mem = root / "memories"
        mem.mkdir(parents=True)
        (mem / "USER.md").write_text(
            "User prefers dark mode and vim keybindings always.",
            encoding="utf-8",
        )
        imp = HermesImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) == 1
        assert mem_items[0].memory_type == MemoryType.WORKING


# ---------------------------------------------------------------------------
# Claude Code importer
# ---------------------------------------------------------------------------

class TestClaudeCodeImporter:
    def test_scan_plans(self, tmp_path):
        from caveman.import_.claude_code import ClaudeCodeImporter
        root = tmp_path / "claude"
        plans = root / "plans"
        plans.mkdir(parents=True)
        (plans / "refactor.md").write_text(
            "## Plan\nRefactor the authentication module to use JWT tokens instead of sessions.",
            encoding="utf-8",
        )
        imp = ClaudeCodeImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) >= 1
        assert mem_items[0].memory_type == MemoryType.PROCEDURAL

    def test_scan_settings(self, tmp_path):
        from caveman.import_.claude_code import ClaudeCodeImporter
        root = tmp_path / "claude"
        root.mkdir(parents=True)
        (root / "settings.json").write_text('{"model": "claude-opus-4-6"}', encoding="utf-8")
        imp = ClaudeCodeImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        cfg_items = [i for i in manifest.items if i.target_type == "config"]
        assert len(cfg_items) == 1


# ---------------------------------------------------------------------------
# Codex importer
# ---------------------------------------------------------------------------

class TestCodexImporter:
    def test_scan_memory_and_rollouts(self, tmp_path):
        from caveman.import_.codex import CodexImporter
        root = tmp_path / "codex"
        root.mkdir(parents=True)
        (root / "MEMORY.md").write_text(
            "## Project Setup\nThe project uses Python 3.12 with Poetry for dependency management.",
            encoding="utf-8",
        )
        rollouts = root / "rollout_summaries"
        rollouts.mkdir()
        (rollouts / "2024-01-15.md").write_text(
            "## Rollout\nDeployed v2.1 to production with zero downtime migration.",
            encoding="utf-8",
        )
        imp = CodexImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) == 2
        types = {i.memory_type for i in mem_items}
        assert MemoryType.SEMANTIC in types
        assert MemoryType.EPISODIC in types


# ---------------------------------------------------------------------------
# Directory importer
# ---------------------------------------------------------------------------

class TestDirectoryImporter:
    def test_scan_recursive(self, tmp_path):
        from caveman.import_.directory import DirectoryImporter
        sub = tmp_path / "notes" / "deep"
        sub.mkdir(parents=True)
        (sub / "note.md").write_text(
            "## Deep Note\nThis is a deeply nested note that should be found by recursive scan.",
            encoding="utf-8",
        )
        imp = DirectoryImporter(caveman_home=tmp_path, directory=tmp_path / "notes")
        manifest = imp.scan()
        assert len(manifest.items) >= 1


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

class TestReports:
    def test_manifest_report(self):
        manifest = ImportManifest(source="test", items=[
            ImportItem(source_path=Path("/a.md"), target_type="memory",
                       memory_type=MemoryType.SEMANTIC, content="x" * 100),
        ])
        report = format_manifest_report(manifest)
        assert "test" in report
        assert "Memory" in report

    def test_result_report(self):
        result = ImportResult(imported=10, duplicates=2, skipped=1, failed=0)
        report = format_result_report(result)
        assert "10" in report
        assert "2" in report

    def test_detect_report(self):
        report = format_detect_report({"openclaw": True, "hermes": False})
        assert "openclaw" in report
        assert "hermes" in report
