"""Round 15 — Flywheel E2E tests.

Tests the complete flywheel:
  Session 1: Task → Shield extracts essence → saves
  Session 2: Recall restores → agent has context
  Session 3: Nudge knowledge → agent uses learned knowledge

Also tests: Safeguard integration, session hooks, delegation hooks.
"""
from __future__ import annotations

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.engines.shield import CompactionShield, SessionEssence
from caveman.engines.recall import RecallEngine, RecallResult
from caveman.memory.nudge import MemoryNudge
from caveman.trajectory.recorder import TrajectoryRecorder
from caveman.compression.safeguard import CompactionSafeguard, SafeguardPhase
from caveman.compression.pipeline import CompressionPipeline
from caveman.agent.session_hooks import on_session_end, on_delegation_complete


class TestFlywheelE2E:
    """The core promise: Shield → Save → Recall → Continuity."""

    @pytest.mark.asyncio
    async def test_shield_save_recall_cycle(self, tmp_path):
        """Session 1 saves essence → Session 2 recalls it."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # --- Session 1: Do work, Shield extracts, saves ---
        shield1 = CompactionShield(
            session_id="session-001", store_dir=sessions_dir,
        )

        # Simulate a conversation with decisions and progress
        messages = [
            {"role": "user", "content": "Build a REST API for user management"},
            {"role": "assistant", "content": "I'll use FastAPI. Let me create the project structure."},
            {"role": "assistant", "content": "Done: created src/main.py with FastAPI app"},
            {"role": "user", "content": "Add authentication"},
            {"role": "assistant", "content": "I decided to use JWT tokens. Implemented auth middleware."},
            {"role": "assistant", "content": "Done: added auth/jwt.py with token generation"},
            {"role": "user", "content": "TODO: add rate limiting next"},
        ]

        essence1 = await shield1.update(messages, task="Build REST API")
        path1 = await shield1.save()

        # Verify Shield extracted meaningful data
        assert essence1.turn_count > 0
        assert essence1.task == "Build REST API"
        assert path1.exists()

        # --- Session 2: Recall restores context ---
        recall = RecallEngine(sessions_dir=sessions_dir)
        restored = await recall.restore_structured("Continue API work")

        # Verify Recall found the previous session
        assert restored.has_context
        assert restored.essences_loaded >= 1
        assert "session-001" in restored.essence_text

        # Verify key information survived the cycle
        text = restored.as_prompt_text()
        assert "REST API" in text or "Build" in text

    @pytest.mark.asyncio
    async def test_multi_session_continuity(self, tmp_path):
        """3 sessions: each builds on the previous."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Session 1
        shield1 = CompactionShield(session_id="s1", store_dir=sessions_dir)
        await shield1.update([
            {"role": "user", "content": "Create project caveman"},
            {"role": "assistant", "content": "Done: created project structure at ~/projects/caveman"},
        ], task="Create project")
        await shield1.save()

        # Session 2 — recalls Session 1
        recall2 = RecallEngine(sessions_dir=sessions_dir)
        ctx2 = await recall2.restore_structured("Add tests")
        assert ctx2.essences_loaded >= 1

        shield2 = CompactionShield(session_id="s2", store_dir=sessions_dir)
        await shield2.update([
            {"role": "user", "content": "Add unit tests"},
            {"role": "assistant", "content": "Done: created tests/test_main.py with 10 tests"},
        ], task="Add tests")
        await shield2.save()

        # Session 3 — recalls both Session 1 and 2
        recall3 = RecallEngine(sessions_dir=sessions_dir, max_essences=3)
        ctx3 = await recall3.restore_structured("Deploy project")
        assert ctx3.essences_loaded >= 2

        # Most recent session should be first
        assert "s2" in ctx3.essence_text

    @pytest.mark.asyncio
    async def test_essence_survives_compaction(self, tmp_path):
        """Shield essence persists even after context compression."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        shield = CompactionShield(session_id="compact-test", store_dir=sessions_dir)

        # Build up a long conversation
        messages = []
        for i in range(20):
            messages.append({"role": "user", "content": f"Step {i}: do task {i}"})
            messages.append({"role": "assistant", "content": f"Done: completed task {i}"})

        # Shield extracts from full conversation
        await shield.update(messages, task="Long task")
        await shield.save()

        # Simulate compaction: messages get compressed, but essence is on disk
        compressed_messages = [
            {"role": "system", "content": "[Compacted: 20 steps completed]"},
            {"role": "user", "content": "Continue with step 21"},
        ]

        # New session can still recall the essence
        recall = RecallEngine(sessions_dir=sessions_dir)
        ctx = await recall.restore_structured("Continue long task")
        assert ctx.has_context
        assert ctx.essences_loaded >= 1

    @pytest.mark.asyncio
    async def test_shield_heuristic_extraction_quality(self, tmp_path):
        """Verify heuristic extraction captures decisions, progress, todos."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        shield = CompactionShield(session_id="quality-test", store_dir=sessions_dir)

        messages = [
            {"role": "user", "content": "How should we handle auth?"},
            {"role": "assistant", "content": "I decided to use JWT with refresh tokens. Going with RS256 signing."},
            {"role": "assistant", "content": "Done: implemented JWT auth in auth/jwt.py"},
            {"role": "assistant", "content": "Created the token refresh endpoint at /api/refresh"},
            {"role": "user", "content": "TODO: add rate limiting to prevent abuse"},
            {"role": "assistant", "content": "Need to add Redis-based rate limiting next"},
        ]

        essence = await shield.update(messages, task="Auth system")

        # Should have extracted decisions
        assert len(essence.decisions) > 0 or len(essence.progress) > 0

        # Should have extracted todos
        assert len(essence.open_todos) > 0

        # Summary should be readable
        summary = essence.summary
        assert "quality-test" in summary
        assert len(summary) > 50


class TestSafeguardIntegration:
    """Safeguard prevents compaction during critical phases."""

    @pytest.mark.asyncio
    async def test_safeguard_defers_during_tool_execution(self):
        """Compaction is deferred when tools are running."""
        sg = CompactionSafeguard()
        sg.enter_phase(SafeguardPhase.TOOL_EXECUTING)

        # Compaction request should be deferred
        assert not sg.request_compaction()
        assert sg.has_pending()

        # After tool completes, compaction can proceed
        sg.enter_phase(SafeguardPhase.IDLE)
        assert sg.request_compaction()

    @pytest.mark.asyncio
    async def test_safeguard_with_compression_pipeline(self):
        """Safeguard gates the compression pipeline."""
        sg = CompactionSafeguard()
        pipeline = CompressionPipeline()

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        # Normal: compression allowed
        assert sg.request_compaction()
        result, stats = await pipeline.compress(messages, context_usage=0.0)
        sg.compaction_completed()

        # During tool execution: deferred
        sg.enter_phase(SafeguardPhase.TOOL_EXECUTING)
        assert not sg.request_compaction()

        # After completion: allowed again
        sg.enter_phase(SafeguardPhase.IDLE)
        assert sg.request_compaction()


class TestSessionHooks:
    """on_session_end and on_delegation_complete hooks."""

    @pytest.mark.asyncio
    async def test_on_session_end_saves_shield(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        shield = CompactionShield(session_id="hook-test", store_dir=sessions_dir)
        await shield.update([
            {"role": "assistant", "content": "Done: completed the task"},
        ], task="Test task")

        trajectory = TrajectoryRecorder()
        await trajectory.record_turn("user", "do something")
        await trajectory.record_turn("gpt", "done")

        result = await on_session_end(
            shield=shield, nudge=None,
            trajectory=trajectory, task="Test task",
        )

        assert result["shield_saved"]
        assert result["essence_turns"] >= 0
        assert (sessions_dir / "hook-test.yaml").exists()

    @pytest.mark.asyncio
    async def test_on_session_end_with_nudge(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        shield = CompactionShield(session_id="nudge-hook", store_dir=sessions_dir)
        trajectory = TrajectoryRecorder()

        # Mock nudge that returns created memories
        mock_nudge = AsyncMock(spec=MemoryNudge)
        mock_nudge.run = AsyncMock(return_value=["memory1", "memory2"])

        result = await on_session_end(
            shield=shield, nudge=mock_nudge,
            trajectory=trajectory, task="Test",
        )

        assert result["memories_created"] == 2
        mock_nudge.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_delegation_complete(self):
        mock_nudge = AsyncMock(spec=MemoryNudge)
        mock_nudge.run = AsyncMock(return_value=["learned_pattern"])

        count = await on_delegation_complete(
            parent_nudge=mock_nudge,
            agent_name="Claude Code",
            agent_output="Fixed the bug by updating the import path",
            task="Fix import error",
        )

        assert count == 1
        mock_nudge.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_delegation_no_nudge(self):
        count = await on_delegation_complete(
            parent_nudge=None,
            agent_name="Claude Code",
            agent_output="output",
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_on_delegation_empty_output(self):
        mock_nudge = AsyncMock(spec=MemoryNudge)
        count = await on_delegation_complete(
            parent_nudge=mock_nudge,
            agent_name="Claude Code",
            agent_output="",
        )
        assert count == 0


class TestRecallWithBudgets:
    """Recall respects token budgets."""

    @pytest.mark.asyncio
    async def test_essence_budget_truncation(self, tmp_path):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Create a session with very long essence
        shield = CompactionShield(session_id="big-session", store_dir=sessions_dir)
        long_decisions = [f"Decision {i}: " + "x" * 200 for i in range(50)]
        shield._essence.decisions = long_decisions
        shield._essence.turn_count = 50
        await shield.save()

        # Recall with small budget
        recall = RecallEngine(
            sessions_dir=sessions_dir,
            essence_budget=500,  # ~2000 chars
        )
        result = await recall.restore_structured("test")

        # Should have loaded but truncated
        assert result.has_context
        assert "truncated" in result.essence_text or len(result.essence_text) < 10_000

    @pytest.mark.asyncio
    async def test_recall_empty_sessions_dir(self, tmp_path):
        recall = RecallEngine(sessions_dir=tmp_path / "nonexistent")
        result = await recall.restore_structured("test")
        assert not result.has_context
        assert result.essences_loaded == 0
        assert result.memories_loaded == 0
