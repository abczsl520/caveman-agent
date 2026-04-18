"""Tests for Round 108 — loop slimming + nudge dedup + flywheel quality."""
import asyncio
import re
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from caveman.agent.bg_tasks import BackgroundTaskMixin
from caveman.memory.nudge import MemoryNudge, _DEDUP_THRESHOLD
from caveman.memory.types import MemoryType, MemoryEntry


# ── BackgroundTaskMixin tests ──

class FakeHost(BackgroundTaskMixin):
    """Minimal host for testing the mixin."""
    def __init__(self):
        self._bg_tasks = set()
        self.memory_manager = MagicMock()
        self.skill_manager = MagicMock()
        self.trajectory_recorder = MagicMock()
        self.trajectory_recorder.to_sharegpt.return_value = []
        self._llm_fn = None
        self._nudge = MagicMock()
        self._nudge.run = AsyncMock(return_value=[])
        self._nudge_task_ref = "test"
        self._lint = MagicMock()
        self._lint.scan = AsyncMock()
        self.engine_flags = MagicMock()
        self.engine_flags.is_enabled.return_value = True
        self.bus = MagicMock()
        self.bus.emit = AsyncMock()


@pytest.mark.asyncio
async def test_safe_bg_tracks_tasks():
    host = FakeHost()
    completed = []

    async def bg_work():
        completed.append(True)

    host._safe_bg(bg_work())
    assert len(host._bg_tasks) == 1
    await asyncio.sleep(0.05)
    assert completed == [True]


@pytest.mark.asyncio
async def test_safe_bg_logs_exceptions():
    host = FakeHost()

    async def failing():
        raise ValueError("boom")

    host._safe_bg(failing())
    await asyncio.sleep(0.05)
    # Task should be cleaned up
    assert len(host._bg_tasks) == 0


@pytest.mark.asyncio
async def test_drain_background():
    host = FakeHost()

    async def slow():
        await asyncio.sleep(0.01)

    host._safe_bg(slow())
    host._safe_bg(slow())
    assert len(host._bg_tasks) == 2
    await host.drain_background(timeout=1.0)
    assert len(host._bg_tasks) == 0


@pytest.mark.asyncio
async def test_boost_trust():
    host = FakeHost()
    backend = MagicMock()
    backend.mark_helpful = AsyncMock()
    host.memory_manager._backend = backend

    await host._boost_trust(["id1", "id2"])
    assert backend.mark_helpful.call_count == 2
    backend.mark_helpful.assert_any_call("id1", helpful=True)
    backend.mark_helpful.assert_any_call("id2", helpful=True)


@pytest.mark.asyncio
async def test_boost_trust_no_backend():
    host = FakeHost()
    host.memory_manager._backend = None
    # Should not raise
    await host._boost_trust(["id1"])


@pytest.mark.asyncio
async def test_run_lint():
    host = FakeHost()
    await host._run_lint()
    host._lint.scan.assert_awaited_once()


@pytest.mark.asyncio
async def test_end_nudge():
    host = FakeHost()
    await host._end_nudge("test task")
    host._nudge.run.assert_awaited_once()


# ── Nudge dedup tests ──

@pytest.fixture
def nudge_with_memory(tmp_path):
    """Create a MemoryNudge with a real MemoryManager."""
    from caveman.memory.manager import MemoryManager
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mm = MemoryManager(base_dir=tmp_path)
    return MemoryNudge(memory_manager=mm)


@pytest.mark.asyncio
async def test_dedup_removes_exact_duplicates(nudge_with_memory):
    nudge = nudge_with_memory
    # Store an existing memory
    await nudge.memory.store("Python 3.12 is the default version", MemoryType.SEMANTIC)

    candidates = [
        {"type": "semantic", "content": "Python 3.12 is the default version"},
        {"type": "semantic", "content": "Something completely different about Rust"},
    ]
    result = await nudge._dedup_candidates(candidates)
    assert len(result) == 1
    assert "Rust" in result[0]["content"]


@pytest.mark.asyncio
async def test_dedup_removes_near_duplicates(nudge_with_memory):
    nudge = nudge_with_memory
    await nudge.memory.store(
        "The server runs on port 8080 with nginx reverse proxy",
        MemoryType.SEMANTIC,
    )

    candidates = [
        {"type": "semantic", "content": "The server runs on port 8080 with nginx as reverse proxy"},
    ]
    result = await nudge._dedup_candidates(candidates)
    # Should be filtered out (high Jaccard similarity)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_dedup_keeps_unique_candidates(nudge_with_memory):
    nudge = nudge_with_memory
    await nudge.memory.store("Python project uses FastAPI", MemoryType.SEMANTIC)

    candidates = [
        {"type": "semantic", "content": "Database migration uses Alembic with PostgreSQL"},
        {"type": "procedural", "content": "Deploy with docker compose up -d"},
    ]
    result = await nudge._dedup_candidates(candidates)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_dedup_within_batch(nudge_with_memory):
    nudge = nudge_with_memory
    candidates = [
        {"type": "semantic", "content": "The API endpoint is /api/v1/users"},
        {"type": "semantic", "content": "The API endpoint is /api/v1/users for user management"},
        {"type": "procedural", "content": "Use curl to test the endpoint"},
    ]
    result = await nudge._dedup_candidates(candidates)
    # First two are near-duplicates, only first + third should survive
    assert len(result) == 2


@pytest.mark.asyncio
async def test_dedup_empty_candidates(nudge_with_memory):
    result = await nudge_with_memory._dedup_candidates([])
    assert result == []


# ── Loop slimming verification ──

def test_loop_under_400_lines():
    """NFR-502: loop.py must be under 400 lines."""
    from pathlib import Path
    loop_path = Path(__file__).parent.parent / "caveman" / "agent" / "loop.py"
    lines = loop_path.read_text().count("\n")
    assert lines <= 400, f"loop.py is {lines} lines, must be ≤400"


def test_bg_tasks_module_exists():
    """Verify bg_tasks.py was properly extracted."""
    from caveman.agent.bg_tasks import BackgroundTaskMixin
    assert hasattr(BackgroundTaskMixin, "_safe_bg")
    assert hasattr(BackgroundTaskMixin, "drain_background")
    assert hasattr(BackgroundTaskMixin, "_boost_trust")
    assert hasattr(BackgroundTaskMixin, "_end_nudge")
    assert hasattr(BackgroundTaskMixin, "_run_lint")
    assert hasattr(BackgroundTaskMixin, "_check_save_skill")


def test_agent_loop_inherits_mixin():
    """AgentLoop should inherit BackgroundTaskMixin."""
    from caveman.agent.loop import AgentLoop
    assert issubclass(AgentLoop, BackgroundTaskMixin)


# ── Heuristic extraction quality tests (PRD §6 Iron Law #1) ──

def test_heuristic_extracts_file_paths():
    """Heuristic should extract project file paths as semantic memory."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from caveman.memory.manager import MemoryManager
        mm = MemoryManager(base_dir="/tmp/test_heuristic")
    nudge = MemoryNudge(memory_manager=mm)
    turns = [
        {"role": "user", "content": "Fix the bug in the login page"},
        {"role": "assistant", "content": "I found the issue in /src/auth/login.py and /src/templates/login.html. The problem was..."},
    ]
    candidates = nudge._extract_heuristic(turns, "Fix login bug")
    contents = [c["content"] for c in candidates]
    # Should mention actual file paths, not just "Tools used: bash"
    path_candidates = [c for c in contents if "/src/" in c]
    assert len(path_candidates) >= 1, f"Should extract file paths, got: {contents}"


def test_heuristic_extracts_decisions():
    """Heuristic should extract decisions as procedural memory."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from caveman.memory.manager import MemoryManager
        mm = MemoryManager(base_dir="/tmp/test_heuristic2")
    nudge = MemoryNudge(memory_manager=mm)
    turns = [
        {"role": "user", "content": "Should we use Redis or Memcached?"},
        {"role": "assistant", "content": "After analysis, I decided to use Redis because it supports persistence and pub/sub."},
    ]
    candidates = nudge._extract_heuristic(turns, "Choose cache backend")
    contents = [c["content"] for c in candidates]
    decision_candidates = [c for c in contents if "decided" in c.lower() or "redis" in c.lower()]
    assert len(decision_candidates) >= 1, f"Should extract decisions, got: {contents}"


def test_heuristic_no_garbage():
    """Heuristic should NOT produce 'Completed task: X' or 'Tools used: bash'."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from caveman.memory.manager import MemoryManager
        mm = MemoryManager(base_dir="/tmp/test_heuristic3")
    nudge = MemoryNudge(memory_manager=mm)
    turns = [
        {"role": "user", "content": "List files"},
        {"role": "assistant", "content": "Here are the files in the directory."},
    ]
    candidates = nudge._extract_heuristic(turns, "List files")
    contents = [c["content"] for c in candidates]
    for c in contents:
        assert "Completed task:" not in c, f"Should not produce garbage: {c}"
        assert "Tools used" not in c, f"Should not produce low-value: {c}"


def test_heuristic_errors_have_context():
    """Error memories should include what was tried, not just the error."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from caveman.memory.manager import MemoryManager
        mm = MemoryManager(base_dir="/tmp/test_heuristic4")
    nudge = MemoryNudge(memory_manager=mm)
    turns = [
        {"role": "assistant", "content": "Running pip install numpy..."},
        {"role": "user", "content": "error: Could not find a version that satisfies the requirement"},
    ]
    candidates = nudge._extract_heuristic(turns, "Install deps")
    error_candidates = [c for c in candidates if c["type"] == "semantic" and "error" in c["content"].lower()]
    for ec in error_candidates:
        # Should have context (what was tried) not just the error
        assert "trying" in ec["content"].lower() or "pip" in ec["content"].lower(), \
            f"Error should have context: {ec['content']}"


# ── HybridScorer trust weight test ──

def test_trust_weight_is_significant():
    """Trust weight should be >= 0.20 to make the confidence loop meaningful."""
    from caveman.memory.retrieval import HybridScorer
    scorer = HybridScorer()
    assert scorer.trust_weight >= 0.20, \
        f"Trust weight {scorer.trust_weight} too low — flywheel confidence loop won't work"


# ── search_sync uses FTS5 ──

@pytest.mark.asyncio
async def test_search_sync_uses_fts5(tmp_path):
    """search_sync should use FTS5 for better Ripple propagation quality."""
    from caveman.memory.sqlite_store import SQLiteMemoryStore
    store = SQLiteMemoryStore(db_path=tmp_path / "test.db")
    await store.store("Python 3.12 is the default runtime version", MemoryType.SEMANTIC)
    await store.store("The server runs on port 8080", MemoryType.SEMANTIC)

    # FTS5 should find this even with partial match
    results = store.search_sync("Python runtime", limit=5)
    assert len(results) >= 1
    assert "Python" in results[0].content
    store.close()
