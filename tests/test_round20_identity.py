"""Tests for Round 20 Phase 4 — ProjectIdentity."""
from __future__ import annotations

import pytest
from caveman.engines.project_identity import (
    ProjectIdentity, ProjectIdentityStore, detect_project_from_messages,
    _safe_filename,
)
from caveman.engines.shield import CompactionShield
from caveman.engines.recall import RecallEngine, RecallResult


class TestProjectIdentity:
    def test_basic_creation(self):
        pi = ProjectIdentity(name="caveman", path="~/projects/caveman")
        assert pi.name == "caveman"
        assert pi.path == "~/projects/caveman"

    def test_to_dict_roundtrip(self):
        pi = ProjectIdentity(
            name="caveman",
            path="~/projects/caveman",
            mission="Beat OpenClaw and Hermes",
            principles=["Stand on giants' shoulders"],
            current_phase="Round 20",
            tech_stack=["Python", "SQLite"],
        )
        d = pi.to_dict()
        restored = ProjectIdentity.from_dict(d)
        assert restored.name == "caveman"
        assert restored.mission == "Beat OpenClaw and Hermes"
        assert len(restored.principles) == 1

    def test_merge_update(self):
        pi1 = ProjectIdentity(name="caveman", mission="Old mission")
        pi2 = ProjectIdentity(
            name="caveman",
            mission="New mission",
            principles=["New principle"],
            current_phase="Round 21",
        )
        pi1.merge_update(pi2)
        assert pi1.mission == "New mission"
        assert pi1.current_phase == "Round 21"
        assert "New principle" in pi1.principles

    def test_prompt_text(self):
        pi = ProjectIdentity(
            name="caveman",
            path="~/projects/caveman",
            mission="Beat them all",
            current_phase="Round 20",
        )
        text = pi.prompt_text
        assert "caveman" in text
        assert "Beat them all" in text
        assert "Round 20" in text

    def test_merge_preserves_existing(self):
        pi1 = ProjectIdentity(
            name="caveman",
            principles=["A", "B"],
            tech_stack=["Python"],
        )
        pi2 = ProjectIdentity(
            name="caveman",
            principles=["B", "C"],
            tech_stack=["Python", "SQLite"],
        )
        pi1.merge_update(pi2)
        assert pi1.principles == ["A", "B", "C"]
        assert pi1.tech_stack == ["Python", "SQLite"]


class TestProjectIdentityStore:
    @pytest.fixture
    def store(self, tmp_path):
        return ProjectIdentityStore(tmp_path / "projects")

    def test_save_load(self, store):
        pi = ProjectIdentity(name="caveman", mission="Test")
        store.save(pi)
        loaded = store.load("caveman")
        assert loaded is not None
        assert loaded.mission == "Test"
        assert loaded.updated_at  # auto-set

    def test_load_nonexistent(self, store):
        assert store.load("nonexistent") is None

    def test_list_projects(self, store):
        store.save(ProjectIdentity(name="proj-a"))
        store.save(ProjectIdentity(name="proj-b"))
        projects = store.list_projects()
        assert len(projects) == 2

    def test_load_by_path(self, store):
        store.save(ProjectIdentity(name="caveman", path="/tmp/test-proj"))
        found = store.load_by_path("/tmp/test-proj")
        assert found is not None
        assert found.name == "caveman"

    def test_load_by_path_not_found(self, store):
        assert store.load_by_path("/nonexistent") is None

    def test_delete(self, store):
        store.save(ProjectIdentity(name="caveman"))
        assert store.delete("caveman")
        assert store.load("caveman") is None
        assert not store.delete("caveman")

    def test_unicode_content(self, store):
        pi = ProjectIdentity(
            name="caveman",
            mission="做出比 OpenClaw 更好用的 Agent 🦴",
        )
        store.save(pi)
        loaded = store.load("caveman")
        assert loaded is not None
        assert "🦴" in loaded.mission


class TestDetectProject:
    def test_detect_from_project_mention(self):
        messages = [
            {"role": "user", "content": "Working on project: caveman"},
        ]
        pi = detect_project_from_messages(messages)
        assert pi is not None
        assert pi.name == "caveman"

    def test_detect_from_path(self):
        messages = [
            {"role": "user", "content": "cd ~/projects/my-app"},
        ]
        pi = detect_project_from_messages(messages)
        assert pi is not None
        assert pi.path == "~/projects/my-app"
        assert pi.name == "my-app"

    def test_detect_mission(self):
        messages = [
            {"role": "user", "content": "project: caveman"},
            {"role": "user", "content": "goal: Build the best AI agent framework"},
        ]
        pi = detect_project_from_messages(messages)
        assert pi is not None
        assert "best AI agent" in pi.mission

    def test_detect_phase(self):
        messages = [
            {"role": "user", "content": "project: caveman"},
            {"role": "assistant", "content": "Currently working on Round 20 platform capabilities"},
        ]
        pi = detect_project_from_messages(messages)
        assert pi is not None

    def test_no_project_detected(self):
        messages = [
            {"role": "user", "content": "What's the weather today?"},
        ]
        pi = detect_project_from_messages(messages)
        assert pi is None

    def test_chinese_project_mention(self):
        messages = [
            {"role": "user", "content": "我们在做 caveman 项目"},
        ]
        pi = detect_project_from_messages(messages)
        assert pi is not None
        assert pi.name == "caveman"


class TestSafeFilename:
    def test_normal(self):
        assert _safe_filename("caveman") == "caveman"

    def test_spaces(self):
        assert _safe_filename("my project") == "my_project"

    def test_special_chars(self):
        assert _safe_filename("my/project@v2") == "my_project_v2"


class TestShieldProjectIntegration:
    @pytest.mark.asyncio
    async def test_shield_detects_project(self, tmp_path):
        shield = CompactionShield(
            session_id="test",
            store_dir=tmp_path / "sessions",
        )
        shield._project_store = ProjectIdentityStore(tmp_path / "projects")

        messages = [
            {"role": "user", "content": "Working on project: caveman"},
            {"role": "assistant", "content": "Let's continue with caveman development"},
        ]
        await shield.update(messages, task="Build caveman")
        assert shield.active_project is not None
        assert shield.active_project.name == "caveman"

    @pytest.mark.asyncio
    async def test_shield_saves_project(self, tmp_path):
        shield = CompactionShield(
            session_id="test",
            store_dir=tmp_path / "sessions",
        )
        store = ProjectIdentityStore(tmp_path / "projects")
        shield._project_store = store

        messages = [
            {"role": "user", "content": "Working on project: myapp"},
            {"role": "assistant", "content": "Done with the task"},
        ]
        await shield.update(messages)
        await shield.save()

        loaded = store.load("myapp")
        assert loaded is not None
        assert loaded.session_count == 1

    @pytest.mark.asyncio
    async def test_shield_set_project_explicit(self, tmp_path):
        shield = CompactionShield(
            session_id="test",
            store_dir=tmp_path / "sessions",
        )
        pi = ProjectIdentity(name="caveman", mission="Beat them all")
        shield.set_project(pi)
        assert shield.active_project is not None
        assert shield.active_project.mission == "Beat them all"


class TestRecallProjectIntegration:
    @pytest.mark.asyncio
    async def test_recall_loads_project(self, tmp_path):
        # Setup: save a project identity
        store = ProjectIdentityStore(tmp_path / "projects")
        store.save(ProjectIdentity(
            name="caveman",
            mission="Beat OpenClaw",
            session_count=1,
        ))

        recall = RecallEngine(
            sessions_dir=tmp_path / "sessions",
            project_store=store,
        )
        result = await recall.restore_structured(task="Continue caveman work")
        assert result.projects_loaded == 1
        assert "caveman" in result.project_text
        assert "Beat OpenClaw" in result.project_text

    @pytest.mark.asyncio
    async def test_recall_project_first_in_prompt(self, tmp_path):
        store = ProjectIdentityStore(tmp_path / "projects")
        store.save(ProjectIdentity(
            name="caveman",
            mission="Beat them",
            session_count=1,
        ))

        recall = RecallEngine(
            sessions_dir=tmp_path / "sessions",
            project_store=store,
        )
        result = await recall.restore_structured(task="caveman stuff")
        text = result.as_prompt_text()
        # Project identity should come first
        assert text.startswith("## Active Project: caveman")

    @pytest.mark.asyncio
    async def test_recall_no_project(self, tmp_path):
        recall = RecallEngine(
            sessions_dir=tmp_path / "sessions",
            project_store=ProjectIdentityStore(tmp_path / "projects"),
        )
        result = await recall.restore_structured(task="random question")
        assert result.projects_loaded == 0
        assert result.project_text == ""
