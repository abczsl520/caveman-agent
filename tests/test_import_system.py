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

    def test_scan_root_level_skills(self, tmp_path):
        """OpenClaw stores user-created skills at ~/.openclaw/skills/ (separate from workspace/skills/).
        The importer must scan both locations."""
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"

        # workspace/skills (already tested implicitly)
        ws_skill = root / "workspace" / "skills" / "taobao-native"
        ws_skill.mkdir(parents=True)
        (ws_skill / "SKILL.md").write_text("---\nname: taobao-native\n---\n# Taobao Native", encoding="utf-8")

        # Root-level skills (the bug: these were not scanned)
        root_skill = root / "skills" / "bug-audit"
        root_skill.mkdir(parents=True)
        (root_skill / "SKILL.md").write_text("---\nname: bug-audit\n---\n# Bug Audit\nDissect then verify.", encoding="utf-8")

        root_skill2 = root / "skills" / "codex-review"
        root_skill2.mkdir(parents=True)
        (root_skill2 / "SKILL.md").write_text("---\nname: codex-review\n---\n# Codex Review\nThree-tier defense.", encoding="utf-8")

        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        skill_items = [i for i in manifest.items if i.target_type == "skill"]
        assert len(skill_items) == 3  # 1 workspace + 2 root-level
        contents = " ".join(i.content for i in skill_items)
        assert "Taobao Native" in contents
        assert "Bug Audit" in contents
        assert "Codex Review" in contents


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

    def test_scan_nested_skills(self, tmp_path):
        """Hermes skills can be nested (e.g. software-development/systematic-debugging/).
        The importer must use rglob to find all SKILL.md files at any depth."""
        from caveman.import_.hermes import HermesImporter
        root = tmp_path / "hermes"
        skills = root / "skills"

        # One-level skill
        s1 = skills / "game-quality-gates"
        s1.mkdir(parents=True)
        (s1 / "SKILL.md").write_text("---\nname: game-quality-gates\n---\n# Game Quality Gates\nFull lifecycle.", encoding="utf-8")

        # Two-level nested skills
        s2 = skills / "software-development" / "systematic-debugging"
        s2.mkdir(parents=True)
        (s2 / "SKILL.md").write_text("---\nname: systematic-debugging\n---\n# Systematic Debugging\nFind root cause first.", encoding="utf-8")

        s3 = skills / "software-development" / "test-driven-development"
        s3.mkdir(parents=True)
        (s3 / "SKILL.md").write_text("---\nname: test-driven-development\n---\n# TDD\nWrite the test first.", encoding="utf-8")

        imp = HermesImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        skill_items = [i for i in manifest.items if i.target_type == "skill"]
        assert len(skill_items) == 3
        # Verify all three are found
        contents = {i.content for i in skill_items}
        assert any("Game Quality Gates" in c for c in contents)
        assert any("Systematic Debugging" in c for c in contents)
        assert any("TDD" in c for c in contents)

    def test_scan_skill_references(self, tmp_path):
        """Skill references (references/*.md) should be imported as procedural memory."""
        from caveman.import_.hermes import HermesImporter
        root = tmp_path / "hermes"
        skill = root / "skills" / "research" / "polymarket"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: polymarket\n---\n# Polymarket", encoding="utf-8")
        refs = skill / "references"
        refs.mkdir()
        (refs / "api-endpoints.md").write_text("# API Endpoints\nGET /markets returns all active markets.", encoding="utf-8")

        imp = HermesImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        ref_items = [i for i in manifest.items if i.target_type == "skill_reference"]
        assert len(ref_items) == 1
        assert ref_items[0].memory_type == MemoryType.PROCEDURAL
        assert "GET /markets" in ref_items[0].content

    def test_nested_skill_target_path(self, tmp_path):
        """Nested skills should preserve directory structure in target path."""
        from caveman.import_.hermes import HermesImporter, _skill_relative_path
        root = tmp_path / "hermes"
        skills_dir = root / "skills"

        # One-level
        s1 = skills_dir / "dogfood" / "SKILL.md"
        s1.parent.mkdir(parents=True)
        s1.write_text("test", encoding="utf-8")

        # Two-level
        s2 = skills_dir / "creative" / "ascii-art" / "SKILL.md"
        s2.parent.mkdir(parents=True)
        s2.write_text("test", encoding="utf-8")

        assert _skill_relative_path(skills_dir, s1) == "dogfood"
        assert _skill_relative_path(skills_dir, s2) == "creative/ascii-art"

    def test_scan_openclaw_synced_memory(self, tmp_path):
        """If Hermes has openclaw-memory/ from cross-sync, it should be scanned."""
        from caveman.import_.hermes import HermesImporter
        root = tmp_path / "hermes"
        oc = root / "openclaw-memory"
        oc.mkdir(parents=True)
        (oc / "validated-approaches.md").write_text(
            "## Approach 1\nUse SQLite FTS5 for full-text search with hybrid retrieval.",
            encoding="utf-8",
        )
        lessons = oc / "memory" / "lessons"
        lessons.mkdir(parents=True)
        (lessons / "game-state.md").write_text(
            "## Game State Management\nSeparate logic from rendering. Always use state machines.",
            encoding="utf-8",
        )

        imp = HermesImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()
        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) >= 2
        contents = " ".join(i.content for i in mem_items)
        assert "SQLite FTS5" in contents
        assert "state machines" in contents


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


# ---------------------------------------------------------------------------
# OpenClaw MEMORY.md dual import (PRD §8.8.1)
# ---------------------------------------------------------------------------

class TestOpenClawMemoryMdDualImport:
    """MEMORY.md should produce BOTH workspace AND memory items."""

    def test_memory_md_produces_workspace_and_memory_items(self, tmp_path):
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        ws = root / "workspace"
        ws.mkdir(parents=True)
        (ws / "MEMORY.md").write_text(
            "## P0 — 核心记忆\n\n### 服务器\n"
            "- **阿里云** `198.51.100.10` Windows（业务主力）\n"
            "- **新服务器** `198.51.100.20` Ubuntu\n\n"
            "## P1 — 按需查\n\n"
            "高频入口已在 AGENTS.md 触发器表。其余知识通过 memory_search 语义检索自动发现。",
            encoding="utf-8",
        )

        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()

        ws_items = [i for i in manifest.items if i.target_type == "workspace"]
        mem_items = [i for i in manifest.items if i.target_type == "memory"]

        # Should have 1 workspace item (the file copy)
        assert len(ws_items) == 1
        assert ws_items[0].source_path.name == "MEMORY.md"

        # Should have 2+ memory items (sections parsed from MEMORY.md)
        assert len(mem_items) >= 2
        # Memory items should contain the actual content
        all_content = " ".join(i.content for i in mem_items)
        assert "198.51.100.10" in all_content
        assert "AGENTS.md" in all_content

    def test_soul_md_does_not_produce_memory_items(self, tmp_path):
        """Only MEMORY.md should get dual treatment, not SOUL.md."""
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        ws = root / "workspace"
        ws.mkdir(parents=True)
        (ws / "SOUL.md").write_text(
            "## Identity\nI am Caveman, a self-evolving AI agent system.\n\n"
            "## Style\nFriendly and natural conversation style always.",
            encoding="utf-8",
        )

        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()

        ws_items = [i for i in manifest.items if i.target_type == "workspace"]
        mem_items = [i for i in manifest.items if i.target_type == "memory"]

        assert len(ws_items) == 1
        assert len(mem_items) == 0  # SOUL.md should NOT be in memory


# ---------------------------------------------------------------------------
# OpenClaw HEARTBEAT.md cron registration (PRD §8.8.1)
# ---------------------------------------------------------------------------

class TestOpenClawHeartbeatCron:
    def test_heartbeat_produces_cron_register(self, tmp_path):
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        ws = root / "workspace"
        ws.mkdir(parents=True)
        (ws / "HEARTBEAT.md").write_text(
            "# HEARTBEAT.md\n\nRun health checks every 5 minutes.\n"
            "Check session health and sub-agent health.",
            encoding="utf-8",
        )

        imp = OpenClawImporter(caveman_home=tmp_path)
        imp.root = root
        manifest = imp.scan()

        ws_items = [i for i in manifest.items if i.target_type == "workspace"]
        cron_items = [i for i in manifest.items if i.target_type == "cron_register"]

        assert len(ws_items) == 1  # workspace copy
        assert len(cron_items) == 1  # cron registration

    @pytest.mark.asyncio
    async def test_heartbeat_cron_writes_file(self, tmp_path):
        from caveman.import_.openclaw import OpenClawImporter
        root = tmp_path / "openclaw"
        ws = root / "workspace"
        ws.mkdir(parents=True)
        (ws / "HEARTBEAT.md").write_text(
            "# HEARTBEAT.md\nRun health checks periodically for system monitoring.",
            encoding="utf-8",
        )

        caveman_home = tmp_path / "caveman"
        caveman_home.mkdir()
        imp = OpenClawImporter(caveman_home=caveman_home, dry_run=False)
        imp.root = root
        manifest = imp.scan()
        result = await imp.execute(manifest, memory_manager=None)

        cron_file = caveman_home / "cron" / "heartbeat-cron.json"
        assert cron_file.exists()
        data = json.loads(cron_file.read_text())
        assert data["name"] == "heartbeat"
        assert "schedule" in data


# ---------------------------------------------------------------------------
# Claude Code project CLAUDE.md scanning
# ---------------------------------------------------------------------------

class TestClaudeCodeProjectScan:
    def test_finds_project_claude_md(self, tmp_path):
        from caveman.import_.claude_code import ClaudeCodeImporter, _PROJECT_SEARCH_PATHS
        root = tmp_path / "claude"
        root.mkdir(parents=True)
        (root / "settings.json").write_text('{}', encoding="utf-8")

        # Create a fake project with CLAUDE.md
        project = tmp_path / "myproject"
        project.mkdir()
        (project / "CLAUDE.md").write_text(
            "## Conventions\nUse snake_case for all Python function names and variables.\n\n"
            "## Architecture\nThe project follows a hexagonal architecture pattern.",
            encoding="utf-8",
        )

        imp = ClaudeCodeImporter(caveman_home=tmp_path)
        imp.root = root
        # Override search paths to use our tmp_path
        import caveman.import_.claude_code as cc_mod
        original_paths = cc_mod._PROJECT_SEARCH_PATHS
        cc_mod._PROJECT_SEARCH_PATHS = (tmp_path,)
        try:
            manifest = imp.scan()
        finally:
            cc_mod._PROJECT_SEARCH_PATHS = original_paths

        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        assert len(mem_items) >= 2
        # Should contain project name prefix
        assert any("[Project: myproject]" in i.content for i in mem_items)

    def test_skips_node_modules(self, tmp_path):
        from caveman.import_.claude_code import ClaudeCodeImporter
        root = tmp_path / "claude"
        root.mkdir(parents=True)
        (root / "settings.json").write_text('{}', encoding="utf-8")

        # CLAUDE.md inside node_modules should be skipped
        nm = tmp_path / "project" / "node_modules" / "some-pkg"
        nm.mkdir(parents=True)
        (nm / "CLAUDE.md").write_text(
            "## Pkg\nThis should not be imported from node_modules directory.",
            encoding="utf-8",
        )

        imp = ClaudeCodeImporter(caveman_home=tmp_path)
        imp.root = root
        import caveman.import_.claude_code as cc_mod
        original_paths = cc_mod._PROJECT_SEARCH_PATHS
        cc_mod._PROJECT_SEARCH_PATHS = (tmp_path,)
        try:
            manifest = imp.scan()
        finally:
            cc_mod._PROJECT_SEARCH_PATHS = original_paths

        mem_items = [i for i in manifest.items if i.target_type == "memory"]
        # node_modules CLAUDE.md should not appear
        assert not any("node_modules" in str(i.source_path) for i in mem_items)


# ---------------------------------------------------------------------------
# WorkspaceMemorySync
# ---------------------------------------------------------------------------

class TestWorkspaceMemorySync:
    """Test the runtime workspace → Memory Store sync."""

    @pytest.mark.asyncio
    async def test_sync_inserts_sections(self, tmp_path):
        from caveman.agent.workspace_memory_sync import WorkspaceMemorySync

        # Create workspace with MEMORY.md
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "MEMORY.md").write_text(
            "## Servers\n- Aliyun `198.51.100.10` Windows\n\n"
            "## Rules\nAlways pull before editing code on remote servers.",
            encoding="utf-8",
        )

        # Mock memory manager
        stored = []
        class MockMM:
            async def store(self, content, mem_type, metadata=None):
                stored.append({"content": content, "type": mem_type, "meta": metadata})
                return f"id-{len(stored)}"

        sync = WorkspaceMemorySync(tmp_path, MockMM())
        sync._workspace_paths = [ws]
        stats = await sync.sync()

        assert stats["inserted"] == 2
        assert stats["unchanged"] == 0
        assert len(stored) == 2
        # Check source tagging
        assert all("workspace-sync:MEMORY.md" in s["meta"]["source"] for s in stored)

    @pytest.mark.asyncio
    async def test_sync_idempotent(self, tmp_path):
        from caveman.agent.workspace_memory_sync import WorkspaceMemorySync

        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "MEMORY.md").write_text(
            "## Facts\nThe server IP is 10.0.0.1 and runs Ubuntu 22.04 LTS.",
            encoding="utf-8",
        )

        stored = []
        class MockMM:
            async def store(self, content, mem_type, metadata=None):
                stored.append(content)
                return f"id-{len(stored)}"

        sync = WorkspaceMemorySync(tmp_path, MockMM())
        sync._workspace_paths = [ws]

        # First sync
        stats1 = await sync.sync()
        assert stats1["inserted"] == 1

        # Second sync — no changes
        stats2 = await sync.sync()
        assert stats2["unchanged"] == 1
        assert stats2["inserted"] == 0
        assert len(stored) == 1  # No new stores

    @pytest.mark.asyncio
    async def test_sync_detects_changes(self, tmp_path):
        from caveman.agent.workspace_memory_sync import WorkspaceMemorySync

        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "MEMORY.md").write_text(
            "## V1\nOriginal content about the project architecture and design.",
            encoding="utf-8",
        )

        stored = []
        deleted = []
        class MockMM:
            async def store(self, content, mem_type, metadata=None):
                stored.append(content)
                return f"id-{len(stored)}"
            async def delete(self, mid):
                deleted.append(mid)

        sync = WorkspaceMemorySync(tmp_path, MockMM())
        sync._workspace_paths = [ws]

        # First sync
        await sync.sync()
        assert len(stored) == 1

        # Modify file
        (ws / "MEMORY.md").write_text(
            "## V2\nUpdated content with new server configuration details.",
            encoding="utf-8",
        )

        # Second sync — should detect change
        stats2 = await sync.sync()
        assert stats2["deleted"] == 1  # old section removed
        assert stats2["inserted"] == 1  # new section added

    @pytest.mark.asyncio
    async def test_force_resync(self, tmp_path):
        from caveman.agent.workspace_memory_sync import WorkspaceMemorySync

        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "MEMORY.md").write_text(
            "## Data\nImportant data about the production environment setup.",
            encoding="utf-8",
        )

        stored = []
        deleted = []
        class MockMM:
            async def store(self, content, mem_type, metadata=None):
                stored.append(content)
                return f"id-{len(stored)}"
            async def delete(self, mid):
                deleted.append(mid)

        sync = WorkspaceMemorySync(tmp_path, MockMM())
        sync._workspace_paths = [ws]

        # Initial sync
        await sync.sync()
        assert len(stored) == 1

        # Force resync — should delete old and re-insert
        stats = await sync.force_resync()
        assert len(deleted) == 1
        assert stats["inserted"] == 1
