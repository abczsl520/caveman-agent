"""Integration test: Memory → Wiki → Recall pipeline.

Verifies that memories stored via MemoryManager can be compiled into
wiki entries and recalled via RecallEngine.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryType
from caveman.wiki import WikiStore
from caveman.wiki.compiler import WikiCompiler
from caveman.engines.recall import RecallEngine


@pytest.fixture
def memory_manager(tmp_path):
    return MemoryManager(base_dir=tmp_path / "memory")


@pytest.fixture
def wiki_compiler(tmp_path):
    store = WikiStore(wiki_dir=tmp_path / "wiki")
    return WikiCompiler(store=store)


@pytest.fixture
def recall_engine(tmp_path, memory_manager):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return RecallEngine(
        sessions_dir=sessions_dir,
        memory_manager=memory_manager,
    )


MEMORIES = [
    ("Python uses 0-based indexing for lists and arrays", MemoryType.SEMANTIC),
    ("The Caveman project uses Python 3.12", MemoryType.SEMANTIC),
    ("Always run pytest before committing code changes", MemoryType.PROCEDURAL),
    ("Server 203.0.113.10 is the Aliyun Windows business server", MemoryType.SEMANTIC),
    ("Use asyncio.Lock to protect concurrent memory writes", MemoryType.PROCEDURAL),
    ("WikiCompiler compiles memories into tiered knowledge", MemoryType.SEMANTIC),
    ("RecallEngine restores context from previous sessions", MemoryType.SEMANTIC),
    ("ToolRegistry auto-discovers @tool decorated functions", MemoryType.SEMANTIC),
    ("Flywheel runs audit rounds to find and fix issues", MemoryType.EPISODIC),
    ("Shield engine saves session state after each task", MemoryType.EPISODIC),
]


@pytest.mark.asyncio
async def test_store_and_recall_pipeline(memory_manager, recall_engine):
    """Store 10 memories, then recall relevant ones via RecallEngine."""
    # Store all memories
    ids = []
    for content, mtype in MEMORIES:
        mid = await memory_manager.store(content, mtype)
        ids.append(mid)
    assert len(ids) == 10
    assert memory_manager.total_count == 10

    # Recall via RecallEngine (structured)
    result = await recall_engine.restore_structured(task="How does Python indexing work?")
    assert result.memories_loaded > 0
    assert "Python" in result.memory_text or "indexing" in result.memory_text


@pytest.mark.asyncio
async def test_store_persist_and_reload(memory_manager, tmp_path):
    """Memories survive save/load cycle."""
    for content, mtype in MEMORIES[:5]:
        await memory_manager.store(content, mtype)
    assert memory_manager.total_count == 5

    # Create a new manager pointing at the same dir
    mgr2 = MemoryManager(base_dir=tmp_path / "memory")
    await mgr2.load()
    assert mgr2.total_count == 5


@pytest.mark.asyncio
async def test_wiki_ingest_and_compile(memory_manager, wiki_compiler):
    """Store memories, ingest into wiki, compile, and verify context."""
    for content, mtype in MEMORIES:
        await memory_manager.store(content, mtype)

    # Ingest each memory into the wiki
    for entry in memory_manager.all_entries():
        wiki_compiler.ingest(
            content=entry.content,
            tags=[entry.memory_type.value],
            source=f"memory:{entry.id}",
        )

    # Compile
    result = wiki_compiler.compile()
    assert result.entries_processed >= 10

    # Get compiled context
    context = wiki_compiler.get_compiled_context(max_tokens=4000)
    assert "Python" in context or "Caveman" in context


@pytest.mark.asyncio
async def test_keyword_recall_relevance(memory_manager):
    """Keyword search returns relevant results."""
    for content, mtype in MEMORIES:
        await memory_manager.store(content, mtype)

    results = await memory_manager.recall("Python project", top_k=3)
    assert len(results) > 0
    contents = [r.content for r in results]
    assert any("Python" in c for c in contents)


@pytest.mark.asyncio
async def test_recent_memories_ordering(memory_manager):
    """Recent memories are returned newest-first."""
    for content, mtype in MEMORIES:
        await memory_manager.store(content, mtype)

    recent = memory_manager.recent(limit=5)
    assert len(recent) == 5
    # Should be reverse chronological
    for i in range(len(recent) - 1):
        assert recent[i].created_at >= recent[i + 1].created_at
