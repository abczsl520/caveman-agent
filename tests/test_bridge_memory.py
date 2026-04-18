"""Tests for MCP bridge + memory v2."""
import pytest
import tempfile
from caveman.memory.manager import MemoryManager, _cosine_similarity
from caveman.memory.types import MemoryType


def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(a, b) - 1.0) < 0.001

    c = [0.0, 1.0, 0.0]
    assert abs(_cosine_similarity(a, c)) < 0.001

    d = [0.707, 0.707, 0.0]
    assert 0.5 < _cosine_similarity(a, d) < 0.8


def test_cosine_edge_cases():
    assert _cosine_similarity([], []) == 0.0
    assert _cosine_similarity([0.0], [0.0]) == 0.0
    assert _cosine_similarity([1.0], [2.0]) == pytest.approx(1.0, abs=0.001)


@pytest.mark.asyncio
async def test_memory_v2_keyword_fallback():
    """Without embedding fn, should fall back to keyword search."""
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager(base_dir=td, embedding_fn=None)
        await mgr.store("Python is a programming language", MemoryType.SEMANTIC)
        await mgr.store("JavaScript runs in browsers", MemoryType.SEMANTIC)

        results = await mgr.recall("Python programming")
        assert len(results) >= 1
        assert "Python" in results[0].content


@pytest.mark.asyncio
async def test_memory_v2_with_mock_embeddings():
    """Test vector search with mock embedding function."""
    # Simple mock: embedding is just word presence vector
    words = ["python", "javascript", "ai", "agent", "memory"]

    async def mock_embed(text: str) -> list[float]:
        text_lower = text.lower()
        return [1.0 if w in text_lower else 0.0 for w in words]

    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager(base_dir=td, embedding_fn=mock_embed)
        await mgr.store("Python AI agent framework", MemoryType.SEMANTIC)
        await mgr.store("JavaScript browser animation", MemoryType.SEMANTIC)
        await mgr.store("AI agent memory system", MemoryType.SEMANTIC)

        results = await mgr.recall("AI agent")
        assert len(results) >= 1
        # Should prefer entries with AI and agent
        assert "AI" in results[0].content or "agent" in results[0].content


@pytest.mark.asyncio
async def test_memory_v2_forget():
    with tempfile.TemporaryDirectory() as td:
        mgr = MemoryManager(base_dir=td)
        mid = await mgr.store("temp memory", MemoryType.WORKING)
        assert mgr.total_count == 1
        assert await mgr.forget(mid)
        assert mgr.total_count == 0


@pytest.mark.asyncio
async def test_memory_v2_persistence():
    """Test that embeddings survive save/load cycle."""
    async def mock_embed(text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]

    with tempfile.TemporaryDirectory() as td:
        mgr1 = MemoryManager(base_dir=td, embedding_fn=mock_embed)
        mid = await mgr1.store("test content", MemoryType.EPISODIC)
        assert mid in mgr1._embeddings

        # Load in new instance
        mgr2 = MemoryManager(base_dir=td, embedding_fn=mock_embed)
        await mgr2.load()
        assert mgr2.total_count == 1
        assert mid in mgr2._embeddings
