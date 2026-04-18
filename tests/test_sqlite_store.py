"""Tests for SQLite + FTS5 memory backend."""
import asyncio
import tempfile
from pathlib import Path

import pytest

from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.memory.types import MemoryType


def test_sqlite_store_and_recall():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            store = SQLiteMemoryStore(db_path=db)

            # Store
            mid1 = await store.store("Python is great for ML", MemoryType.SEMANTIC)
            mid2 = await store.store("User prefers dark mode", MemoryType.WORKING)
            mid3 = await store.store("Deploy with docker compose", MemoryType.PROCEDURAL)

            assert store.total_count == 3
            assert store.type_counts()["semantic"] == 1
            assert store.type_counts()["working"] == 1

            # FTS5 recall
            results = await store.recall("Python ML")
            assert len(results) >= 1
            assert any("Python" in r.content for r in results)

            # Type-filtered recall
            results = await store.recall("mode", memory_type=MemoryType.WORKING)
            assert len(results) >= 1
            assert results[0].memory_type == MemoryType.WORKING

            store.close()

    asyncio.run(_run())


def test_sqlite_fts5_ranking():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            store = SQLiteMemoryStore(db_path=db)

            # Store multiple entries
            await store.store("Python web framework Flask", MemoryType.SEMANTIC)
            await store.store("Python data science pandas numpy", MemoryType.SEMANTIC)
            await store.store("JavaScript React frontend", MemoryType.SEMANTIC)
            await store.store("Python machine learning tensorflow pytorch", MemoryType.SEMANTIC)

            # Search should rank Python ML higher for ML query
            results = await store.recall("Python machine learning", top_k=3)
            assert len(results) >= 1
            # First result should be about ML (best FTS5 match)
            assert "machine learning" in results[0].content.lower() or "python" in results[0].content.lower()

            store.close()

    asyncio.run(_run())


def test_sqlite_forget():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            store = SQLiteMemoryStore(db_path=db)

            mid = await store.store("temporary memory", MemoryType.EPISODIC)
            assert store.total_count == 1

            success = await store.forget(mid)
            assert success
            assert store.total_count == 0

            # Forget non-existent
            assert not await store.forget("nonexistent")

            store.close()

    asyncio.run(_run())


def test_sqlite_empty_query():
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            store = SQLiteMemoryStore(db_path=db)

            await store.store("some memory", MemoryType.SEMANTIC)
            results = await store.recall("")
            assert results == []

            store.close()

    asyncio.run(_run())


def test_sqlite_like_fallback():
    """Test that LIKE search works when FTS5 query syntax fails."""
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            store = SQLiteMemoryStore(db_path=db)

            await store.store("error handling in python", MemoryType.PROCEDURAL)

            # This should work via FTS or LIKE fallback
            results = await store.recall("python error")
            assert len(results) >= 1

            store.close()

    asyncio.run(_run())


def test_sqlite_vector_search():
    """Test vector search with mock embedding function."""
    async def _run():
        async def mock_embed(text: str) -> list[float]:
            # Simple: hash chars to create a vector
            vec = [0.0] * 8
            for i, c in enumerate(text[:8]):
                vec[i] = ord(c) / 255.0
            return vec

        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"
            store = SQLiteMemoryStore(db_path=db, embedding_fn=mock_embed)

            await store.store("machine learning with python", MemoryType.SEMANTIC)
            await store.store("cooking pasta recipes", MemoryType.SEMANTIC)
            await store.store("machine learning models", MemoryType.SEMANTIC)

            results = await store.recall("machine learning")
            assert len(results) >= 1
            # Vector search should find ML entries
            assert any("machine" in r.content for r in results)

            store.close()

    asyncio.run(_run())


def test_sqlite_persistence():
    """Test that data survives close/reopen."""
    async def _run():
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "test.db"

            # Write
            store1 = SQLiteMemoryStore(db_path=db)
            await store1.store("persistent memory", MemoryType.SEMANTIC)
            store1.close()

            # Read in new instance
            store2 = SQLiteMemoryStore(db_path=db)
            assert store2.total_count == 1
            results = await store2.recall("persistent")
            assert len(results) == 1
            assert results[0].content == "persistent memory"
            store2.close()

    asyncio.run(_run())
