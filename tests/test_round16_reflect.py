"""Tests for Round 16 — Reflect Engine + Memory Confidence."""
from __future__ import annotations

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from caveman.engines.reflect import (
    ReflectEngine,
    Reflection,
)
from caveman.memory.confidence import (
    ConfidenceTracker,
    ConfidenceRecord,
    DEFAULT_CONFIDENCE,
    RECALL_SUCCESS,
    RECALL_FAILURE,
    REFLECT_CONFIRM,
    LINT_STALE,
)
from caveman.skills.manager import SkillManager
from caveman.engines.flags import EngineFlags


# --- Reflect Engine ---

class TestReflection:
    def test_to_dict(self):
        r = Reflection(task="test", outcome="success")
        d = r.to_dict()
        assert d["task"] == "test"
        assert d["outcome"] == "success"
        assert "reflected_at" in d

    def test_defaults(self):
        r = Reflection(task="t", outcome="partial")
        assert r.effective_patterns == []
        assert r.anti_patterns == []
        assert r.confidence == 0.5


class TestReflectEngine:
    @pytest.mark.asyncio
    async def test_heuristic_success_detection(self):
        engine = ReflectEngine()
        trajectory = [
            {"from": "human", "value": "Build a REST API"},
            {"from": "gpt", "value": "I'll create the project. Running bash command..."},
            {"from": "gpt", "value": "Done: created src/main.py ✅"},
            {"from": "gpt", "value": "All 10 tests pass"},
        ]
        r = await engine.reflect("Build API", trajectory, "commit e59d0d0")
        assert r.outcome == "success"
        assert r.confidence >= 0.6

    @pytest.mark.asyncio
    async def test_heuristic_failure_detection(self):
        engine = ReflectEngine()
        trajectory = [
            {"from": "human", "value": "Fix the bug"},
            {"from": "gpt", "value": "Trying to fix... got TypeError"},
            {"from": "gpt", "value": "FAILED: still broken ❌"},
        ]
        r = await engine.reflect("Fix bug", trajectory, "error")
        assert r.outcome == "failure"
        assert r.confidence <= 0.5

    @pytest.mark.asyncio
    async def test_anti_pattern_detection(self):
        engine = ReflectEngine()
        trajectory = [
            {"from": "gpt", "value": "Tried 5 times but still failing"},
            {"from": "gpt", "value": "Applied a workaround to bypass the issue"},
        ]
        r = await engine.reflect("Fix issue", trajectory)
        assert len(r.anti_patterns) > 0

    @pytest.mark.asyncio
    async def test_effective_pattern_extraction(self):
        engine = ReflectEngine()
        trajectory = [
            {"from": "gpt", "value": "Ran bash command to check status"},
            {"from": "gpt", "value": "Created file src/main.py"},
            {"from": "gpt", "value": "Searched for documentation"},
        ]
        r = await engine.reflect("Setup project", trajectory, "Done ✅")
        assert len(r.effective_patterns) > 0

    @pytest.mark.asyncio
    async def test_skill_creation_on_success(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        sm = SkillManager(skills_dir=skills_dir)

        engine = ReflectEngine(skill_manager=sm)
        trajectory = [
            {"from": "human", "value": "Deploy the app"},
            {"from": "gpt", "value": "Running bash deploy script"},
            {"from": "gpt", "value": "Done: deployed successfully ✅"},
            {"from": "gpt", "value": "All health checks pass"},
        ]
        r = await engine.reflect("Deploy the app", trajectory, "success")

        assert r.outcome == "success"
        assert len(r.skill_updates) > 0

        # Skill should have been created
        skills = sm.list_all()
        assert len(skills) >= 1
        assert skills[0].source == "reflect"

    @pytest.mark.asyncio
    async def test_skill_update_on_repeat(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        sm = SkillManager(skills_dir=skills_dir)

        engine = ReflectEngine(skill_manager=sm)

        # Trajectory needs >= 4 turns for skill creation
        traj = [
            {"from": "human", "value": "Deploy the app"},
            {"from": "gpt", "value": "Running deploy script via bash"},
            {"from": "gpt", "value": "Created deployment config file"},
            {"from": "gpt", "value": "Done: deployed successfully ✅"},
            {"from": "gpt", "value": "All health checks pass"},
        ]

        # First reflection creates skill
        r1 = await engine.reflect("deploy app", traj, "success")
        assert len(r1.skill_updates) > 0

        # Second reflection updates it
        r2 = await engine.reflect("deploy app", traj, "success")

        skill = sm.get("deploy-app")
        assert skill is not None
        assert skill.success_count >= 2
        assert skill.version >= 2

    @pytest.mark.asyncio
    async def test_reflections_accumulate(self):
        engine = ReflectEngine()
        await engine.reflect("task1", [{"from": "gpt", "value": "done ✅"}])
        await engine.reflect("task2", [{"from": "gpt", "value": "done ✅"}])
        assert len(engine.reflections) == 2

    @pytest.mark.asyncio
    async def test_llm_reflection(self):
        async def mock_llm(prompt):
            return '{"outcome": "success", "effective_patterns": ["used tests"], "anti_patterns": [], "lessons": ["always test first"], "confidence": 0.9}'

        engine = ReflectEngine(llm_fn=mock_llm)
        r = await engine.reflect("test task", [{"from": "gpt", "value": "done"}])
        assert r.outcome == "success"
        assert r.confidence == 0.9
        assert "always test first" in r.lessons

    @pytest.mark.asyncio
    async def test_llm_fallback_to_heuristic(self):
        async def bad_llm(prompt):
            raise RuntimeError("LLM unavailable")

        engine = ReflectEngine(llm_fn=bad_llm)
        r = await engine.reflect("task", [{"from": "gpt", "value": "done ✅"}])
        # Should still work via heuristic
        assert r.outcome in ("success", "partial", "failure")

    @pytest.mark.asyncio
    async def test_task_to_skill_name(self):
        engine = ReflectEngine()
        assert engine._task_to_skill_name("Build a REST API") == "build-a-rest-api"
        assert engine._task_to_skill_name("Fix bug #123") == "fix-bug-123"
        assert engine._task_to_skill_name("") == "unnamed-skill"


# --- Memory Confidence ---

class TestConfidenceRecord:
    def test_defaults(self):
        r = ConfidenceRecord(memory_id="m1")
        assert r.score == DEFAULT_CONFIDENCE
        assert r.use_count == 0
        assert r.success_rate == 0.5

    def test_success_rate(self):
        r = ConfidenceRecord(memory_id="m1", success_count=3, failure_count=1)
        assert r.success_rate == 0.75


class TestConfidenceTracker:
    def test_initial_score(self):
        tracker = ConfidenceTracker()
        rec = tracker.get("m1")
        assert rec.score == DEFAULT_CONFIDENCE

    def test_recall_success_increases(self):
        tracker = ConfidenceTracker()
        score = tracker.record_use("m1", success=True)
        assert score > DEFAULT_CONFIDENCE
        assert score == DEFAULT_CONFIDENCE + RECALL_SUCCESS

    def test_recall_failure_decreases(self):
        tracker = ConfidenceTracker()
        score = tracker.record_use("m1", success=False)
        assert score < DEFAULT_CONFIDENCE

    def test_reflect_confirm_boost(self):
        tracker = ConfidenceTracker()
        score = tracker.record_reflect_confirm("m1")
        assert score == DEFAULT_CONFIDENCE + REFLECT_CONFIRM

    def test_lint_stale_drop(self):
        tracker = ConfidenceTracker()
        score = tracker.record_lint_stale("m1")
        assert score == DEFAULT_CONFIDENCE + LINT_STALE

    def test_clamped_to_bounds(self):
        tracker = ConfidenceTracker()
        # Many failures should not go below 0
        for _ in range(20):
            tracker.record_use("m1", success=False)
        assert tracker.get("m1").score >= 0.0

        # Many successes should not go above 1
        for _ in range(20):
            tracker.record_use("m2", success=True)
        assert tracker.get("m2").score <= 1.0

    def test_use_count_tracking(self):
        tracker = ConfidenceTracker()
        tracker.record_use("m1", success=True)
        tracker.record_use("m1", success=True)
        tracker.record_use("m1", success=False)
        rec = tracker.get("m1")
        assert rec.use_count == 3
        assert rec.success_count == 2
        assert rec.failure_count == 1

    def test_rank_memories(self):
        tracker = ConfidenceTracker()
        # m1 has high confidence
        tracker.record_use("m1", success=True)
        tracker.record_use("m1", success=True)
        # m2 has low confidence
        tracker.record_use("m2", success=False)
        tracker.record_use("m2", success=False)

        ranked = tracker.rank_memories(
            ["m1", "m2"],
            relevance_scores=[0.5, 0.8],  # m2 more relevant
            confidence_weight=0.5,
        )
        # m1 should rank higher due to confidence despite lower relevance
        assert ranked[0][0] == "m1"

    def test_rank_without_relevance(self):
        tracker = ConfidenceTracker()
        tracker.record_use("m1", success=True)
        ranked = tracker.rank_memories(["m1", "m2"])
        assert len(ranked) == 2

    def test_feedback(self):
        tracker = ConfidenceTracker()
        s1 = tracker.record_feedback("m1", helpful=True)
        assert s1 > DEFAULT_CONFIDENCE
        s2 = tracker.record_feedback("m2", helpful=False)
        assert s2 < DEFAULT_CONFIDENCE

    def test_stats_empty(self):
        tracker = ConfidenceTracker()
        s = tracker.stats()
        assert s["count"] == 0

    def test_stats_with_data(self):
        tracker = ConfidenceTracker()
        tracker.record_use("m1", success=True)
        tracker.record_use("m2", success=False)
        s = tracker.stats()
        assert s["count"] == 2
        assert s["total_uses"] == 2


# --- Engine Flags ---

class TestReflectInFlags:
    def test_reflect_in_engines(self):
        from caveman.engines.flags import ENGINES
        assert "reflect" in ENGINES

    def test_reflect_enabled_by_default(self):
        flags = EngineFlags()
        assert flags.is_enabled("reflect")

    def test_reflect_can_be_disabled(self):
        flags = EngineFlags(config={"engines": {"reflect": {"enabled": False}}})
        assert not flags.is_enabled("reflect")
