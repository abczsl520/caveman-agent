"""Tests from 20-perspective flywheel audit."""
import pytest
from caveman.memory.quality_gate import reset_cache


@pytest.fixture(autouse=True)
def _clean_qg_cache():
    reset_cache()
    yield
    reset_cache()


# ── 视角 7: 边界条件 ──

@pytest.mark.asyncio
async def test_empty_query_returns_empty(tmp_path):
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType
    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "t.db")
    await mm.store("The project uses SQLite for persistent storage", MemoryType.SEMANTIC)
    r = await mm.recall("", top_k=5)
    assert r == []


@pytest.mark.asyncio
async def test_sql_injection_safe(tmp_path):
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType
    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "t.db")
    await mm.store("Architecture decision: use layered parasitism approach", MemoryType.SEMANTIC)
    # Should not crash or delete data
    await mm.recall("'; DROP TABLE memories; --", top_k=3)
    assert mm.total_count == 1


@pytest.mark.asyncio
async def test_emoji_content(tmp_path):
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType
    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "t.db")
    await mm.store("🔥🚀 Deploy the application with 🐳 docker compose", MemoryType.SEMANTIC)
    r = await mm.recall("docker deploy", top_k=3)
    assert len(r) >= 1


# ── 视角 9: 依赖方向 ──

def test_memory_does_not_import_agent():
    """memory/ should not depend on agent/ (inner layer independence)."""
    from pathlib import Path
    memory_dir = Path(__file__).parent.parent / "caveman" / "memory"
    for py in memory_dir.glob("*.py"):
        source = py.read_text()
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "from caveman.agent" not in stripped, \
                f"{py.name} imports caveman.agent — violates dependency direction"
            assert "import caveman.agent" not in stripped, \
                f"{py.name} imports caveman.agent — violates dependency direction"


def test_engines_does_not_import_agent():
    """engines/ should not depend on agent/."""
    from pathlib import Path
    engines_dir = Path(__file__).parent.parent / "caveman" / "engines"
    for py in engines_dir.glob("*.py"):
        source = py.read_text()
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "from caveman.agent" not in stripped, \
                f"{py.name} imports caveman.agent — violates dependency direction"


# ── 视角 14: 内存增长 ──

@pytest.mark.asyncio
async def test_reflections_capped():
    """ReflectEngine should cap stored reflections to prevent memory leak."""
    from caveman.engines.reflect import ReflectEngine
    engine = ReflectEngine(max_reflections=5)
    for i in range(10):
        await engine.reflect(f"task {i}", [{"from": "gpt", "value": "done"}], "ok")
    assert len(engine.reflections) <= 5


# ── 视角 16: 错误隔离 ──

@pytest.mark.asyncio
async def test_ripple_failure_does_not_block_store(tmp_path):
    """Ripple failure should not prevent memory from being stored."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "t.db")

    class FailingRipple:
        async def propagate(self, entry):
            raise RuntimeError("Ripple exploded")

    mm.set_ripple(FailingRipple())
    mid = await mm.store("The root cause was a missing null check in auth handler", MemoryType.SEMANTIC)
    assert mid  # Store should succeed despite ripple failure
    assert mm.total_count == 1


# ── 视角 17: 配置 ──

@pytest.mark.asyncio
async def test_scorer_config_passthrough(tmp_path):
    """HybridScorer config should be passable through factory chain."""
    from caveman.memory.sqlite_store import SQLiteMemoryStore
    from caveman.memory.types import MemoryType

    store = SQLiteMemoryStore(
        db_path=tmp_path / "t.db",
        scorer_config={"trust_weight": 0.4},
    )
    assert store._scorer_config == {"trust_weight": 0.4}
    await store.store("Caveman uses a 7-engine flywheel architecture", MemoryType.SEMANTIC)
    results = await store.recall("flywheel", top_k=1)
    assert len(results) >= 0  # Just verify it doesn't crash
    store.close()
