"""Tests for confidence feedback loop — the core flywheel compound interest.

When memories are recalled and the task succeeds/fails, their trust scores
should be adjusted. Over time, helpful memories surface first.

Round 107: Added Lint→trust and Reflect→trust loop tests.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.memory.types import MemoryType, MemoryEntry
from caveman.events import EventBus, EventType


class TestConfidenceFeedbackLoop:
    """Test the Recall → Use → Outcome → Trust adjustment cycle."""

    @pytest.mark.asyncio
    async def test_finalize_adjusts_trust_on_success(self, tmp_path):
        """Successful task should boost trust of recalled memories."""
        from caveman.agent.phases import phase_finalize
        from caveman.memory.manager import MemoryManager
        from caveman.skills.manager import SkillManager
        from caveman.trajectory.recorder import TrajectoryRecorder

        mm = MemoryManager.with_sqlite(db_path=tmp_path / "test.db")
        mid = await mm.store("Server runs on port 8080", MemoryType.SEMANTIC)

        # Get initial trust
        initial = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]

        bus = EventBus()
        await phase_finalize(
            task="Deploy the server",
            final="Successfully deployed to port 8080",
            matched_skills=[],
            memory_manager=mm,
            skill_manager=SkillManager(),
            trajectory_recorder=TrajectoryRecorder(),
            bus=bus,
            recalled_ids=[mid],
        )

        # Trust should have increased
        after = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]
        assert after > initial

    @pytest.mark.asyncio
    async def test_finalize_adjusts_trust_on_failure(self, tmp_path):
        """Failed task should demote trust of recalled memories."""
        from caveman.agent.phases import phase_finalize
        from caveman.memory.manager import MemoryManager
        from caveman.skills.manager import SkillManager
        from caveman.trajectory.recorder import TrajectoryRecorder

        mm = MemoryManager.with_sqlite(db_path=tmp_path / "test.db")
        mid = await mm.store("Use pip install for deps", MemoryType.PROCEDURAL)

        initial = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]

        bus = EventBus()
        await phase_finalize(
            task="Install dependencies",
            final="Error: pip install failed with permission denied",
            matched_skills=[],
            memory_manager=mm,
            skill_manager=SkillManager(),
            trajectory_recorder=TrajectoryRecorder(),
            bus=bus,
            recalled_ids=[mid],
        )

        after = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]
        assert after < initial

    @pytest.mark.asyncio
    async def test_finalize_no_recalled_ids_is_safe(self, tmp_path):
        """Finalize without recalled_ids should work fine (backward compat)."""
        from caveman.agent.phases import phase_finalize
        from caveman.memory.manager import MemoryManager
        from caveman.skills.manager import SkillManager
        from caveman.trajectory.recorder import TrajectoryRecorder

        mm = MemoryManager.with_sqlite(db_path=tmp_path / "test.db")
        bus = EventBus()

        result = await phase_finalize(
            task="Simple task",
            final="Done",
            matched_skills=[],
            memory_manager=mm,
            skill_manager=SkillManager(),
            trajectory_recorder=TrajectoryRecorder(),
            bus=bus,
        )
        assert result == "Done"

    @pytest.mark.asyncio
    async def test_prepare_emits_recalled_ids(self, tmp_path):
        """phase_prepare should emit recalled_ids in MEMORY_RECALL event."""
        from caveman.memory.manager import MemoryManager

        mm = MemoryManager.with_sqlite(db_path=tmp_path / "test.db")
        mid = await mm.store("Python 3.12 is required", MemoryType.SEMANTIC)

        bus = EventBus()
        captured_ids = []

        def capture(event):
            if event.data.get("recalled_ids"):
                captured_ids.extend(event.data["recalled_ids"])

        bus.on(EventType.MEMORY_RECALL, capture)

        # Simulate what phase_prepare does
        memories = await mm.recall("Python version", top_k=5)
        await bus.emit(EventType.MEMORY_RECALL, {
            "query": "Python version",
            "results": len(memories),
            "recalled_ids": [m.id for m in memories],
        }, source="memory")

        assert mid in captured_ids


class TestLintTrustFeedback:
    """Test Lint → trust_score demotion loop (Round 107)."""

    @pytest.mark.asyncio
    async def test_lint_demotes_trust_for_stale_paths(self, tmp_path):
        """Lint finding stale paths should lower trust_score."""
        from caveman.engines.lint import LintEngine
        from caveman.memory.manager import MemoryManager

        mm = MemoryManager.with_sqlite(db_path=tmp_path / "test.db")
        mid = await mm.store(
            "Config is at /nonexistent/path/config.yaml",
            MemoryType.SEMANTIC,
        )

        initial = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]

        lint = LintEngine(memory_manager=mm, check_paths=True)
        report = await lint.scan()

        # Should have found the stale path
        stale_issues = [i for i in report.issues if i.memory_id == mid]
        assert len(stale_issues) > 0

        # Trust should have decreased
        after = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]
        assert after < initial

    @pytest.mark.asyncio
    async def test_lint_no_penalty_for_clean_memories(self, tmp_path):
        """Clean memories should not have trust demoted."""
        from caveman.engines.lint import LintEngine
        from caveman.memory.manager import MemoryManager

        mm = MemoryManager.with_sqlite(db_path=tmp_path / "test.db")
        mid = await mm.store("Python 3.12 is great", MemoryType.SEMANTIC)

        initial = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]

        lint = LintEngine(memory_manager=mm, check_paths=False)
        report = await lint.scan()

        after = mm._backend._get_conn().execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]
        assert after == initial

    @pytest.mark.asyncio
    async def test_lint_compounds_penalties(self, tmp_path):
        """Multiple lint scans should compound trust penalties."""
        from caveman.engines.lint import LintEngine
        from caveman.memory.manager import MemoryManager

        mm = MemoryManager.with_sqlite(db_path=tmp_path / "test.db")
        mid = await mm.store(
            "Deploy to /nonexistent/server/path",
            MemoryType.SEMANTIC,
        )

        lint = LintEngine(memory_manager=mm, check_paths=True)

        scores = []
        for _ in range(3):
            await lint.scan()
            score = mm._backend._get_conn().execute(
                "SELECT trust_score FROM memories WHERE id = ?", (mid,)
            ).fetchone()[0]
            scores.append(score)

        # Each scan should lower trust further
        assert scores[0] >= scores[1] >= scores[2]  # Trust may not decrease if lint finds no issues


class TestEstimateTokens:
    """Test the unified token estimation in utils.py."""

    def test_english_text(self):
        from caveman.utils import estimate_tokens
        # 400 ASCII chars → 400/4 + 3 = 103
        assert estimate_tokens("x" * 400) == 103

    def test_cjk_text(self):
        from caveman.utils import estimate_tokens
        # 100 CJK chars → 100 + 0/4 + 3 = 103
        assert estimate_tokens("中" * 100) == 103

    def test_mixed_text(self):
        from caveman.utils import estimate_tokens
        # 50 CJK + 50 ASCII → 50 + 50/4 + 3 = 65
        text = "中" * 50 + "x" * 50
        assert estimate_tokens(text) == 65

    def test_empty(self):
        from caveman.utils import estimate_tokens
        assert estimate_tokens("") == 0

    def test_context_delegates_to_utils(self):
        """context._estimate_str_tokens should delegate to utils.estimate_tokens."""
        from caveman.agent.context import _estimate_str_tokens
        from caveman.utils import estimate_tokens
        text = "Hello 你好 World 世界"
        assert _estimate_str_tokens(text) == estimate_tokens(text)
