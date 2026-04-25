"""Regression tests for memory subsystem self-audit fixes."""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from caveman.memory.manager import MemoryManager
from caveman.memory.nudge import MemoryNudge
from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.memory.types import MemoryEntry, MemoryType
from caveman.mcp import server as mcp_server


def test_mcp_memory_manager_uses_sqlite_backend():
    """MCP tools must share the production memory graph, not legacy JSON files."""
    manager = mcp_server._memory_manager()

    try:
        assert isinstance(manager.backend, SQLiteMemoryStore)
    finally:
        if hasattr(manager.backend, "close"):
            manager.backend.close()


def test_memory_manager_get_by_id_sqlite(tmp_path):
    async def _run():
        mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "mem.db")
        mid = await mm.store(
            "A durable memory with a unique lookup token alpha123",
            MemoryType.SEMANTIC,
            metadata={"trust_score": 0.7},
            trusted=True,
        )
        found = await mm.get_by_id(mid)
        missing = await mm.get_by_id("missing")
        assert found is not None
        assert found.id == mid
        assert found.content.startswith("A durable memory")
        assert found.metadata["trust_score"] == 0.7
        assert missing is None

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_nudge_does_not_report_rejected_store():
    class RejectingMemory:
        async def recall(self, *args, **kwargs):
            return []

        def recent(self, *args, **kwargs):
            return []

        def search_sync(self, *args, **kwargs):
            return []

        async def store(self, *args, **kwargs):
            return ""

    nudge = MemoryNudge(RejectingMemory(), llm_fn=None)
    nudge.drift_detector.check = AsyncMock(return_value=None)
    created = await nudge.run([
        {"role": "assistant", "content": "decided use sqlite memory backend for reliable recall"},
    ], task="memory audit")
    assert created == []


def test_phase_finalize_marks_recalled_memory_based_on_actual_content(tmp_path):
    async def _run():
        from caveman.agent.phases import phase_finalize
        from caveman.events import EventBus

        mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "mem.db")
        used = await mm.store(
            "Use pytest fail-fast when debugging regressions",
            MemoryType.PROCEDURAL,
            metadata={"trust_score": 0.5},
            trusted=True,
        )
        unused = await mm.store(
            "Deploy frontend with nginx blue green release",
            MemoryType.PROCEDURAL,
            metadata={"trust_score": 0.5},
            trusted=True,
        )

        class Skills:
            def record_outcome(self, *args, **kwargs):
                pass

            async def auto_create(self, *args, **kwargs):
                pass

        class Trajectory:
            def to_sharegpt(self):
                return []

            async def save(self):
                pass

        await phase_finalize(
            task="debug regression",
            final="Validated fix with pytest fail-fast: 1 passed, exit code 0. Fixed by running pytest fail-fast and patching the failing assertion.",
            matched_skills=[],
            memory_manager=mm,
            skill_manager=Skills(),
            trajectory_recorder=Trajectory(),
            bus=EventBus(),
            recalled_ids=[used, unused],
        )
        used_entry = await mm.get_by_id(used)
        unused_entry = await mm.get_by_id(unused)
        assert used_entry.metadata["trust_score"] > 0.5
        assert unused_entry.metadata["trust_score"] < 0.5
        assert used_entry.metadata["judge_mode"] == "heuristic"
        assert unused_entry.metadata["judge_mode"] == "heuristic"

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_memory_judge_parses_llm_json():
    from caveman.memory.judge import MemoryJudge

    async def fake_llm(prompt: str) -> str:
        assert "RECALLED_MEMORY" in prompt
        return '{"helpful": true, "confidence": 0.91, "reason": "memory directly guided the fix"}'

    memory = MemoryEntry(
        id="m1",
        content="Use pytest fail-fast when debugging regressions",
        memory_type=MemoryType.PROCEDURAL,
        created_at=datetime.now(),
        metadata={},
    )
    result = await MemoryJudge(fake_llm).judge_helpfulness(
        task="debug regression",
        final="pytest fail-fast exposed the failing assertion",
        memory=memory,
        success=True,
    )
    assert result.helpful is True
    assert result.confidence == 0.91
    assert result.mode == "llm"


def test_phase_finalize_uses_llm_judge_metadata(tmp_path):
    async def _run():
        from caveman.agent.phases import phase_finalize
        from caveman.events import EventBus

        mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "mem.db")
        mid = await mm.store(
            "Use migration dry-run before applying memory schema changes",
            MemoryType.PROCEDURAL,
            metadata={"trust_score": 0.5},
            trusted=True,
        )

        async def fake_judge(prompt: str) -> str:
            assert "migration dry-run" in prompt
            return '{"helpful": true, "confidence": 0.88, "reason": "dry-run advice was used"}'

        class Skills:
            def record_outcome(self, *args, **kwargs):
                pass

            async def auto_create(self, *args, **kwargs):
                pass

        class Trajectory:
            def to_sharegpt(self):
                return []

            async def save(self):
                pass

        await phase_finalize(
            task="apply memory schema change safely",
            final="Used migration dry-run first, then applied the schema change successfully.",
            matched_skills=[],
            memory_manager=mm,
            skill_manager=Skills(),
            trajectory_recorder=Trajectory(),
            bus=EventBus(),
            llm_fn=fake_judge,
            recalled_ids=[mid],
        )
        entry = await mm.get_by_id(mid)
        assert entry.metadata["trust_score"] > 0.5
        assert entry.metadata["judge_mode"] == "llm"
        assert entry.metadata["judge_confidence"] == 0.88
        assert "dry-run" in entry.metadata["judge_reason"]

    asyncio.run(_run())
