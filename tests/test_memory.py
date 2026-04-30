"""Tests for memory system."""
import pytest
import tempfile

from caveman.memory.flywheel_metrics import FlywheelHealth
from caveman.memory.types import MemoryType
from caveman.memory.manager import MemoryManager


def _close_manager(mgr: MemoryManager) -> None:
    if mgr.backend:
        mgr.backend.close()


def test_memory_types():
    assert MemoryType.EPISODIC.value == "episodic"
    assert MemoryType.WORKING.value == "working"


@pytest.mark.asyncio
async def test_memory_store_recall():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            mid = await mgr.store("test content about python", MemoryType.EPISODIC)
            assert mid
            results = await mgr.recall("python")
            assert len(results) >= 1
            assert "python" in results[0].content
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_memory_nudge():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            await mgr.nudge()  # should not raise
            assert True  # Nudge completed without error
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_flywheel_health_uses_real_feedback_and_recall_counters():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            stale = await mgr.store("stale memory about docker compose", MemoryType.SEMANTIC)
            helpful = await mgr.store("helpful python deployment memory", MemoryType.SEMANTIC)

            results = await mgr.recall("python deployment")
            assert {r.id for r in results} >= {helpful}
            await mgr.backend.mark_helpful(helpful, helpful=True)

            health = await FlywheelHealth.diagnose(mgr.backend)

            assert health.total_memories == 2
            assert health.memories_never_recalled == 1
            assert health.recalled_memories == 1
            assert health.recall_rate == 0.5
            assert health.memories_with_feedback == 1
            assert health.feedback_rate == 0.5
            assert health.top_recalled[0]["id"] == helpful
            assert stale not in {item["id"] for item in health.top_recalled}
            assert "recall rate=50%" in health.summary()
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_import_memories_are_excluded_from_recall_candidates():
    """Reversible quarantine must remove noisy imported memories from active recall."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            quarantined = await mgr.store(
                "docker compose restart troubleshooting",
                MemoryType.PROCEDURAL,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )
            active = await mgr.store(
                "docker compose restart troubleshooting verified in current project",
                MemoryType.PROCEDURAL,
                metadata={"source": "nudge"},
                trusted=True,
            )

            results = await mgr.recall("docker compose restart troubleshooting", top_k=5)

            assert active in {entry.id for entry in results}
            assert quarantined not in {entry.id for entry in results}
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_memories_do_not_leak_through_recall_fallback():
    """If all lexical matches are quarantined, high-trust fallback must not re-add them."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            quarantined = await mgr.store(
                "only quarantined docker compose restart memory",
                MemoryType.PROCEDURAL,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )

            results = await mgr.recall("docker compose restart", top_k=5)

            assert quarantined not in {entry.id for entry in results}
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_memories_do_not_leak_through_sync_search():
    """search_sync is also an active recall path and must honor quarantine."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            quarantined_id = await store.store(
                "sync docker compose restart memory",
                MemoryType.PROCEDURAL,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )

            results = store.search_sync("sync docker compose restart", limit=5)

            assert quarantined_id not in {entry.id for entry in results}
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantined_memories_do_not_leak_through_recent_or_all_entries():
    """List-style active memory APIs must not expose quarantined imports."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            quarantined_id = await store.store(
                "recent quarantined import memory",
                MemoryType.SEMANTIC,
                metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                trusted=True,
            )
            active_id = await store.store(
                "recent active nudge memory",
                MemoryType.SEMANTIC,
                metadata={"source": "nudge"},
                trusted=True,
            )

            recent_ids = {entry.id for entry in store.recent(limit=10)}
            all_ids = {entry.id for entry in store.all_entries()}

            assert active_id in recent_ids
            assert active_id in all_ids
            assert quarantined_id not in recent_ids
            assert quarantined_id not in all_ids
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_quarantine_sql_filter_prevents_limited_fts_page_from_hiding_active_match():
    """SQL-side filtering must avoid quarantined first-page rows crowding out active matches."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            for i in range(8):
                await store.store(
                    f"needle dominant quarantined import memory {i}",
                    MemoryType.SEMANTIC,
                    metadata={"source": "import:openclaw", "governance_state": "quarantined"},
                    trusted=True,
                )
            active_id = await store.store(
                "needle active memory should survive quarantine crowding",
                MemoryType.SEMANTIC,
                metadata={"source": "nudge"},
                trusted=True,
            )

            results = await store.recall("needle", top_k=1)

            assert [entry.id for entry in results] == [active_id]
        finally:
            _close_manager(mgr)


@pytest.mark.asyncio
async def test_malformed_metadata_does_not_break_active_memory_queries():
    """Legacy/corrupt metadata rows must not crash SQL-side quarantine filtering."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager.with_sqlite(base_dir=td)
        try:
            store = mgr.backend
            memory_id = await store.store(
                "legacy malformed metadata needle",
                MemoryType.SEMANTIC,
                metadata={"source": "legacy"},
                trusted=True,
            )
            store._get_conn().execute(
                "UPDATE memories SET metadata_json = ? WHERE id = ?",
                ("not json", memory_id),
            )
            store._get_conn().commit()

            recall_ids = {entry.id for entry in await store.recall("needle", top_k=5)}
            recent_ids = {entry.id for entry in store.recent(limit=5)}
            all_ids = {entry.id for entry in store.all_entries()}

            assert memory_id in recall_ids
            assert memory_id in recent_ids
            assert memory_id in all_ids
        finally:
            _close_manager(mgr)
