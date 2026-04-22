"""Tests for Obsidian Vault Bridge."""
import json
from pathlib import Path

import pytest

from caveman.import_.obsidian_bridge import ObsidianBridge
from caveman.import_.obsidian_models import (
    ObsidianNote,
    ObsidianVault,
    SyncResult,
    VaultSyncState,
    _split_note_for_memory,
    parse_note,
)
from caveman.memory.types import MemoryType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vault(tmp_path: Path, notes: dict[str, str]) -> Path:
    """Create a fake Obsidian vault with given notes."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / ".obsidian").mkdir()
    for rel_path, content in notes.items():
        p = vault / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return vault


# ---------------------------------------------------------------------------
# Note Parsing
# ---------------------------------------------------------------------------

class TestNoteParsing:
    def test_basic_note(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "test.md": "# Hello\n\nSome content here.",
        })
        note = parse_note(vault / "test.md", vault)
        assert note is not None
        assert note.title == "Hello"
        assert note.body == "# Hello\n\nSome content here."
        assert note.file_hash

    def test_front_matter_parsing(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "note.md": "---\ntags:\n  - 架构\n  - AI-Agent\ndate: 2026-04-11\nsource: Discord\n---\n\n# My Note\n\nContent.",
        })
        note = parse_note(vault / "note.md", vault)
        assert note is not None
        assert "架构" in note.tags
        assert "AI-Agent" in note.tags
        assert note.date == "2026-04-11"
        assert note.front_matter["source"] == "Discord"

    def test_comment_tags(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "note.md": "<!-- tags: SQLite, DEFAULT, node -->\n# SQLite 踩坑\n\nContent.",
        })
        note = parse_note(vault / "note.md", vault)
        assert note is not None
        assert "SQLite" in note.tags
        assert "DEFAULT" in note.tags

    def test_wikilinks(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "note.md": "# Note\n\nSee [[Other Note]] and [[API密钥速查|密钥]].",
        })
        note = parse_note(vault / "note.md", vault)
        assert note is not None
        assert "Other Note" in note.wikilinks
        assert "API密钥速查" in note.wikilinks

    def test_date_from_filename(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "2026-04-14-obsidian学习.md": "# Learning\n\nContent.",
        })
        note = parse_note(vault / "2026-04-14-obsidian学习.md", vault)
        assert note is not None
        assert note.date == "2026-04-14"

    def test_empty_file_returns_none(self, tmp_path):
        vault = _make_vault(tmp_path, {"empty.md": ""})
        assert parse_note(vault / "empty.md", vault) is None

    def test_folder_inference(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "经验教训/git.md": "# Git 踩坑\n\nContent.",
        })
        note = parse_note(vault / "经验教训/git.md", vault)
        assert note is not None
        assert note.folder == "经验教训"
        assert note.inferred_memory_type == MemoryType.PROCEDURAL

    def test_architecture_folder(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "架构/hermes.md": "# Hermes\n\nAnalysis.",
        })
        note = parse_note(vault / "架构/hermes.md", vault)
        assert note is not None
        assert note.inferred_memory_type == MemoryType.SEMANTIC

    def test_daily_folder(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "日常/2026-04-14.md": "# Today\n\nDid stuff.",
        })
        note = parse_note(vault / "日常/2026-04-14.md", vault)
        assert note is not None
        assert note.inferred_memory_type == MemoryType.EPISODIC


# ---------------------------------------------------------------------------
# Vault Scanner
# ---------------------------------------------------------------------------

class TestVaultScanner:
    def test_scan_basic(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "note1.md": "# Note 1\n\nContent.",
            "sub/note2.md": "# Note 2\n\nContent.",
        })
        vault = ObsidianVault(root=vault_path)
        vault.scan()
        assert vault.note_count == 2

    def test_scan_skips_obsidian_dir(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "note.md": "# Note\n\nContent.",
        })
        # .obsidian already created by _make_vault
        (vault_path / ".obsidian" / "config.md").write_text("internal")
        vault = ObsidianVault(root=vault_path)
        vault.scan()
        assert vault.note_count == 1  # Only note.md, not .obsidian/config.md

    def test_scan_skips_hidden(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "note.md": "# Note\n\nContent.",
            ".hidden/secret.md": "# Secret\n\nHidden.",
        })
        vault = ObsidianVault(root=vault_path)
        vault.scan()
        assert vault.note_count == 1

    def test_resolve_wikilink(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "架构/hermes.md": "# Hermes\n\nAnalysis.",
            "项目/caveman.md": "# Caveman\n\nSee [[hermes]].",
        })
        vault = ObsidianVault(root=vault_path)
        vault.scan()
        resolved = vault.resolve_wikilink("hermes")
        assert resolved is not None
        assert resolved.title == "Hermes"


# ---------------------------------------------------------------------------
# Note Splitting
# ---------------------------------------------------------------------------

class TestNoteSplitting:
    def test_small_note_single_chunk(self, tmp_path):
        vault = _make_vault(tmp_path, {"note.md": "# Small\n\nShort content."})
        note = parse_note(vault / "note.md", vault)
        chunks = _split_note_for_memory(note)
        assert len(chunks) == 1
        assert "[Obsidian] Small" in chunks[0]

    def test_large_note_split_by_headers(self, tmp_path):
        content = "# Big Note\n\n" + "## Section A\n\n" + "A" * 2000 + "\n\n## Section B\n\n" + "B" * 2000
        vault = _make_vault(tmp_path, {"big.md": content})
        note = parse_note(vault / "big.md", vault)
        chunks = _split_note_for_memory(note, max_chars=2500)
        assert len(chunks) >= 2
        assert all("[Obsidian] Big Note" in c for c in chunks)

    def test_empty_body_no_chunks(self, tmp_path):
        vault = _make_vault(tmp_path, {"empty.md": "---\ntags: [test]\n---\n\n"})
        note = parse_note(vault / "empty.md", vault)
        chunks = _split_note_for_memory(note)
        assert len(chunks) == 0

    def test_tags_in_header(self, tmp_path):
        vault = _make_vault(tmp_path, {
            "note.md": "---\ntags:\n  - 架构\n  - Hermes\n---\n\n# Analysis\n\nContent here.",
        })
        note = parse_note(vault / "note.md", vault)
        chunks = _split_note_for_memory(note)
        assert len(chunks) == 1
        assert "架构" in chunks[0]
        assert "Hermes" in chunks[0]


# ---------------------------------------------------------------------------
# Sync State
# ---------------------------------------------------------------------------

class TestSyncState:
    def test_save_and_load(self, tmp_path):
        state = VaultSyncState(vault_path="/tmp/vault", last_sync="2026-04-18")
        state.save(tmp_path)
        loaded = VaultSyncState.load(tmp_path)
        assert loaded.vault_path == "/tmp/vault"
        assert loaded.last_sync == "2026-04-18"

    def test_needs_sync_new_file(self):
        state = VaultSyncState()
        note = ObsidianNote(
            path=Path("/tmp/note.md"), relative_path="note.md",
            content="x", file_hash="abc123",
        )
        assert state.needs_sync(note)

    def test_needs_sync_unchanged(self):
        state = VaultSyncState()
        note = ObsidianNote(
            path=Path("/tmp/note.md"), relative_path="note.md",
            content="x", file_hash="abc123",
        )
        state.mark_synced(note, ["mem1"])
        assert not state.needs_sync(note)

    def test_needs_sync_changed(self):
        state = VaultSyncState()
        note = ObsidianNote(
            path=Path("/tmp/note.md"), relative_path="note.md",
            content="x", file_hash="abc123",
        )
        state.mark_synced(note, ["mem1"])
        note.file_hash = "def456"  # Changed
        assert state.needs_sync(note)

    def test_get_deleted(self):
        state = VaultSyncState()
        note = ObsidianNote(
            path=Path("/tmp/note.md"), relative_path="note.md",
            content="x", file_hash="abc123",
        )
        state.mark_synced(note, ["mem1"])
        deleted = state.get_deleted(set())  # No current files
        assert len(deleted) == 1
        assert deleted[0].relative_path == "note.md"


# ---------------------------------------------------------------------------
# Bridge Integration
# ---------------------------------------------------------------------------

class TestObsidianBridge:
    def test_link_vault(self, tmp_path):
        vault_path = _make_vault(tmp_path, {"note.md": "# Test\n\nContent."})
        bridge = ObsidianBridge(caveman_home=tmp_path)
        result = bridge.link(vault_path)
        assert "Linked" in result
        assert bridge.is_linked

    def test_link_nonexistent(self, tmp_path):
        bridge = ObsidianBridge(caveman_home=tmp_path)
        result = bridge.link(tmp_path / "nonexistent")
        assert "Error" in result

    def test_unlink(self, tmp_path):
        vault_path = _make_vault(tmp_path, {"note.md": "# Test\n\nContent."})
        bridge = ObsidianBridge(caveman_home=tmp_path)
        bridge.link(vault_path)
        bridge.unlink()
        assert not bridge.is_linked

    def test_status_unlinked(self, tmp_path):
        bridge = ObsidianBridge(caveman_home=tmp_path)
        status = bridge.status()
        assert status["linked"] is False

    def test_status_linked(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "note1.md": "# Note 1\n\nContent.",
            "note2.md": "# Note 2\n\nContent.",
        })
        bridge = ObsidianBridge(caveman_home=tmp_path)
        bridge.link(vault_path)
        status = bridge.status()
        assert status["linked"] is True
        assert status["total_notes"] == 2
        assert status["new_files"] == 2

    @pytest.mark.asyncio
    async def test_sync_dry_run(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "架构/hermes.md": "# Hermes Analysis\n\nDeep dive into Hermes architecture.",
        })
        bridge = ObsidianBridge(caveman_home=tmp_path)
        bridge.link(vault_path)

        # Mock memory manager
        class MockMM:
            async def store(self, content, memory_type, metadata=None):
                return "mock-id"
            async def delete(self, mid):
                pass
            def all_entries(self):
                return []

        result = await bridge.sync(MockMM(), dry_run=True)
        assert result.added == 1
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_sync_actual(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "经验教训/sqlite.md": "# SQLite 踩坑\n\nDEFAULT 表达式不支持函数。",
        })
        bridge = ObsidianBridge(caveman_home=tmp_path)
        bridge.link(vault_path)

        stored = []
        class MockMM:
            async def store(self, content, memory_type, metadata=None):
                stored.append({"content": content, "type": memory_type, "meta": metadata})
                return f"id-{len(stored)}"
            async def delete(self, mid):
                pass
            def all_entries(self):
                return []

        result = await bridge.sync(MockMM())
        assert result.added == 1
        assert len(stored) == 1
        assert stored[0]["type"] == MemoryType.PROCEDURAL
        assert stored[0]["meta"]["source"] == "obsidian"
        assert "sqlite" in stored[0]["meta"]["vault_file"].lower()

    @pytest.mark.asyncio
    async def test_incremental_sync(self, tmp_path):
        vault_path = _make_vault(tmp_path, {
            "note.md": "# Note\n\nOriginal content.",
        })
        bridge = ObsidianBridge(caveman_home=tmp_path)
        bridge.link(vault_path)

        class MockMM:
            async def store(self, content, memory_type, metadata=None):
                return "id-1"
            async def delete(self, mid):
                pass
            def all_entries(self):
                return []

        # First sync
        r1 = await bridge.sync(MockMM())
        assert r1.added == 1

        # Second sync — no changes
        r2 = await bridge.sync(MockMM())
        assert r2.unchanged == 1
        assert r2.added == 0

        # Modify file
        (vault_path / "note.md").write_text("# Note\n\nUpdated content!")
        r3 = await bridge.sync(MockMM())
        assert r3.updated == 1

    @pytest.mark.asyncio
    async def test_write_note(self, tmp_path):
        vault_path = _make_vault(tmp_path, {})
        bridge = ObsidianBridge(caveman_home=tmp_path)
        bridge.link(vault_path)

        result = await bridge.write_note(
            title="Hermes 研究总结",
            content="# 关键发现\n\nHermes 的核心是学习飞轮。",
            folder="架构",
            tags=["Hermes", "竞品研究"],
        )
        assert result is not None
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Hermes" in content
        assert "caveman" in content  # source: caveman in front matter
