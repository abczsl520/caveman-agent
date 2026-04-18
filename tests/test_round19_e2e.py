"""E2E Flywheel Verification — real-world multi-session test.

Verifies the complete flywheel cycle:
  Session 1: Task execution → Shield save → Wiki ingest → Nudge extract
  Session 2: Recall restore → Use Session 1 knowledge → Shield update
  Session 3: Knowledge promotion → Compiled context available

This test uses real components (no mocks) but doesn't require an LLM.
It exercises the heuristic paths of all engines.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from caveman.engines.shield import CompactionShield, SessionEssence
from caveman.engines.recall import RecallEngine
from caveman.engines.reflect import ReflectEngine
from caveman.wiki import WikiStore
from caveman.wiki.compiler import WikiCompiler
from caveman.agent.session_hooks import on_session_end
from caveman.memory.nudge import MemoryNudge
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.compression.safeguard import CompactionSafeguard, SafeguardPhase


class TestRealWorldFlywheel:
    """Multi-session flywheel verification with real components."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Create a temporary workspace with all required dirs."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        return {
            "root": tmp_path,
            "sessions": sessions_dir,
            "wiki": wiki_dir,
            "memory": memory_dir,
        }

    @pytest.mark.asyncio
    async def test_session_1_task_and_save(self, workspace):
        """Session 1: Execute task, save Shield essence, ingest to Wiki."""
        # Setup
        shield = CompactionShield(
            session_id="session-001",
            store_dir=workspace["sessions"],
        )
        wiki = WikiCompiler(WikiStore(workspace["wiki"]))

        # Simulate a task execution with messages
        messages = [
            {"role": "user", "content": "Build a REST API with FastAPI"},
            {"role": "assistant", "content": "I'll create a FastAPI project with:\n1. User authentication with JWT\n2. CRUD endpoints for items\n3. PostgreSQL database with SQLAlchemy"},
            {"role": "user", "content": "Good, also add rate limiting"},
            {"role": "assistant", "content": "Added rate limiting with slowapi. The API now has:\n- JWT auth (access + refresh tokens)\n- CRUD for items with pagination\n- Rate limiting: 100 req/min per IP\n- PostgreSQL with async SQLAlchemy"},
        ]

        # Shield extracts essence
        essence = await shield.update(messages, task="Build REST API")
        assert essence.turn_count >= 1

        # Save Shield
        path = await shield.save()
        assert path.exists()

        # Wiki ingest session
        wiki_entry = wiki.ingest_session(
            session_id="session-001",
            task="Build REST API with FastAPI",
            decisions=["Use JWT for auth", "Use PostgreSQL", "Add rate limiting"],
            progress=["Created project structure", "Implemented auth", "Added rate limiting"],
            todos=["Add tests", "Deploy to production"],
        )
        assert wiki_entry.tier == "episodic"

        # Verify Wiki state
        stats = wiki.store.stats()
        assert stats["episodic"] == 1

        return shield, wiki

    @pytest.mark.asyncio
    async def test_session_2_recall_and_continue(self, workspace):
        """Session 2: Recall Session 1 context, continue work."""
        # First, run Session 1 to create state
        shield1 = CompactionShield(
            session_id="session-001",
            store_dir=workspace["sessions"],
        )
        messages1 = [
            {"role": "user", "content": "Build a REST API"},
            {"role": "assistant", "content": "Created FastAPI project with JWT auth and PostgreSQL"},
        ]
        await shield1.update(messages1, task="Build REST API")
        await shield1.save()

        wiki = WikiCompiler(WikiStore(workspace["wiki"]))
        wiki.ingest_session(
            session_id="session-001",
            task="Build REST API",
            decisions=["Use JWT", "Use PostgreSQL"],
            progress=["Created project"],
            todos=["Add tests"],
        )

        # Session 2: Recall
        recall = RecallEngine(
            sessions_dir=workspace["sessions"],
            max_essences=3,
        )
        result = await recall.restore_structured("Continue REST API work")
        assert result.has_context
        assert result.essences_loaded >= 1

        # Wiki compiled context
        context = wiki.get_compiled_context(max_tokens=2000)
        assert "Episodic" in context or "Session" in context

        # Session 2 Shield
        shield2 = CompactionShield(
            session_id="session-002",
            store_dir=workspace["sessions"],
        )
        messages2 = [
            {"role": "user", "content": "Add tests for the API"},
            {"role": "assistant", "content": "Added pytest tests:\n- test_auth.py: JWT flow\n- test_items.py: CRUD operations\n- test_rate_limit.py: rate limiting\nAll 15 tests pass."},
        ]
        await shield2.update(messages2, task="Add API tests")
        await shield2.save()

        # Verify both sessions exist
        sessions = list(workspace["sessions"].glob("*.yaml"))
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_session_3_knowledge_promotion(self, workspace):
        """Session 3: Knowledge gets promoted through reinforcement."""
        wiki = WikiCompiler(WikiStore(workspace["wiki"]))

        # Ingest the same knowledge multiple times (simulating repeated use)
        entry = wiki.ingest("FastAPI is great for async REST APIs", tags=["fastapi"])
        assert entry.tier == "working"

        # Reinforce (simulating the knowledge being useful across sessions)
        for _ in range(3):
            wiki.reinforce(entry.id)

        # Compile — should promote to episodic
        result = wiki.compile()
        assert result.entries_promoted >= 1

        # Verify promotion
        episodic = wiki.store.load_tier("episodic")
        assert any(e.id == entry.id for e in episodic)

        # Continue reinforcing
        for _ in range(4):
            wiki.reinforce(entry.id)

        # Compile again — should promote to semantic
        result2 = wiki.compile()
        semantic = wiki.store.load_tier("semantic")
        assert any(e.id == entry.id for e in semantic)

        # Compiled context should include the promoted knowledge
        context = wiki.get_compiled_context()
        assert "FastAPI" in context

    @pytest.mark.asyncio
    async def test_full_flywheel_cycle(self, workspace):
        """Complete flywheel: Shield → Wiki → Compile → Recall → Repeat."""
        wiki = WikiCompiler(WikiStore(workspace["wiki"]))

        # === Session 1 ===
        shield1 = CompactionShield(
            session_id="fly-001",
            store_dir=workspace["sessions"],
        )
        msgs1 = [
            {"role": "user", "content": "Set up CI/CD pipeline"},
            {"role": "assistant", "content": "Created GitHub Actions workflow with:\n- Lint + type check\n- Tests on Python 3.12\n- Deploy to staging on PR merge\n- Deploy to prod on release tag"},
        ]
        await shield1.update(msgs1, task="CI/CD setup")
        await shield1.save()

        # Wiki ingest
        wiki.ingest_session(
            session_id="fly-001",
            task="CI/CD setup",
            decisions=["GitHub Actions", "Staging on PR merge", "Prod on release"],
            progress=["Workflow created", "Tests passing"],
            todos=["Add Docker build step"],
        )
        wiki.ingest("GitHub Actions is the CI/CD tool of choice", tags=["cicd"], source="session:fly-001")

        # Compile
        r1 = wiki.compile()
        assert r1.entries_processed >= 2

        # === Session 2 ===
        # Recall from Session 1
        recall = RecallEngine(sessions_dir=workspace["sessions"], max_essences=3)
        recalled = await recall.restore_structured("Add Docker to CI/CD")
        assert recalled.has_context

        # Get wiki context
        ctx = wiki.get_compiled_context()
        assert len(ctx) > 0

        shield2 = CompactionShield(
            session_id="fly-002",
            store_dir=workspace["sessions"],
        )
        msgs2 = [
            {"role": "user", "content": "Add Docker build to CI/CD"},
            {"role": "assistant", "content": "Added Docker build step:\n- Multi-stage Dockerfile\n- Build + push to GHCR\n- Cache layers for speed"},
        ]
        await shield2.update(msgs2, task="Docker CI/CD")
        await shield2.save()

        # Reinforce the CI/CD knowledge (it was useful)
        cicd_entries = wiki.store.search("GitHub Actions")
        for e in cicd_entries:
            wiki.reinforce(e.id)
            wiki.reinforce(e.id)

        # === Session 3 ===
        # Compile again — knowledge should be promoted
        r3 = wiki.compile()

        # Recall should now have 2 sessions
        recalled3 = await recall.restore_structured("Review CI/CD pipeline")
        assert recalled3.essences_loaded >= 2

        # Wiki should have promoted entries
        all_stats = wiki.store.stats()
        total = sum(all_stats.values())
        assert total >= 2  # At least the session + the fact

        # Final compiled context should be richer
        final_ctx = wiki.get_compiled_context()
        assert len(final_ctx) > len(ctx) or "GitHub Actions" in final_ctx

    @pytest.mark.asyncio
    async def test_safeguard_protects_during_critical_phase(self, workspace):
        """Safeguard delays compaction during tool execution."""
        safeguard = CompactionSafeguard()

        # Enter critical phase
        safeguard.enter_phase(SafeguardPhase.TOOL_EXECUTING)
        assert safeguard.state.is_critical

        # Request compaction — should be deferred
        result = safeguard.request_compaction()
        assert not result  # Can't compact during critical phase

    @pytest.mark.asyncio
    async def test_reflect_extracts_patterns(self, workspace):
        """Reflect engine extracts patterns from completed tasks."""
        reflect = ReflectEngine()

        # Simulate a trajectory
        trajectory = [
            {"role": "user", "content": "Fix the authentication bug"},
            {"role": "assistant", "content": "Found the issue: JWT token expiry was set to 0. Fixed to 3600s."},
            {"role": "user", "content": "That worked, thanks!"},
        ]

        reflection = reflect._reflect_heuristic("Fix auth bug", trajectory, "success")
        assert reflection is not None
        assert reflection.outcome in ("success", "partial", "failure", "unknown")

    @pytest.mark.asyncio
    async def test_on_session_end_with_wiki(self, workspace):
        """on_session_end hook integrates with Wiki Compiler."""
        shield = CompactionShield(
            session_id="hook-001",
            store_dir=workspace["sessions"],
        )
        msgs = [
            {"role": "user", "content": "Create a database schema"},
            {"role": "assistant", "content": "Created schema with users, posts, comments tables"},
        ]
        await shield.update(msgs, task="DB schema")

        trajectory = TrajectoryRecorder(base_dir=workspace["root"] / "trajectories")
        await trajectory.record_turn("user", "Create a database schema")
        await trajectory.record_turn("assistant", "Created schema with users, posts, comments tables")

        wiki = WikiCompiler(WikiStore(workspace["wiki"]))

        result = await on_session_end(
            shield=shield,
            nudge=None,
            trajectory=trajectory,
            task="DB schema",
            wiki_compiler=wiki,
        )

        assert result["shield_saved"]
        assert result["wiki_ingested"]

        # Verify wiki has the session
        stats = wiki.store.stats()
        assert stats["episodic"] >= 1
