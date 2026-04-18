"""Tests for hybrid retrieval system (Hermes port)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from caveman.memory.retrieval import (
    tokenize,
    jaccard_similarity,
    extract_entities,
    temporal_decay,
    adjust_trust,
    HybridScorer,
)
from caveman.memory.types import MemoryEntry, MemoryType


# --- Tokenize ---

class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello world this is a test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Stop words removed
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_chinese_stop_words(self):
        tokens = tokenize("这是一个测试 的 了")
        assert "这" not in tokens
        assert "的" not in tokens
        assert "了" not in tokens

    def test_empty(self):
        assert tokenize("") == set()
        assert tokenize("the a is") == set()

    def test_single_char_removed(self):
        tokens = tokenize("I am a b c developer")
        assert "i" not in tokens  # single char
        assert "developer" in tokens


# --- Jaccard ---

class TestJaccard:
    def test_identical(self):
        a = {"hello", "world"}
        assert jaccard_similarity(a, a) == 1.0

    def test_disjoint(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        sim = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.4 < sim < 0.6  # 2/4 = 0.5

    def test_empty(self):
        assert jaccard_similarity(set(), {"a"}) == 0.0
        assert jaccard_similarity(set(), set()) == 0.0


# --- Entity extraction ---

class TestExtractEntities:
    def test_capitalized_names(self):
        entities = extract_entities("John Smith works at Google Cloud")
        assert "John Smith" in entities

    def test_ips(self):
        entities = extract_entities("Server at 192.168.1.1 and 10.0.0.1")
        assert "192.168.1.1" in entities
        assert "10.0.0.1" in entities

    def test_urls(self):
        entities = extract_entities("Visit https://example.com/path")
        assert any("example.com" in e for e in entities)

    def test_paths(self):
        entities = extract_entities("Edit /Users/test/projects/caveman/main.py")
        assert any("/Users/test" in e for e in entities)

    def test_quoted(self):
        entities = extract_entities('The project "Caveman Agent OS" is cool')
        assert "Caveman Agent OS" in entities

    def test_empty(self):
        assert extract_entities("hello world") == []

    def test_dedup(self):
        entities = extract_entities("192.168.1.1 and 192.168.1.1")
        assert entities.count("192.168.1.1") == 1


# --- Temporal decay ---

class TestTemporalDecay:
    def test_recent(self):
        now = datetime.now(timezone.utc)
        assert temporal_decay(now, half_life_days=30) > 0.99

    def test_old(self):
        old = datetime.now(timezone.utc) - timedelta(days=60)
        decay = temporal_decay(old, half_life_days=30)
        assert 0.2 < decay < 0.3  # ~0.25 (2 half-lives)

    def test_disabled(self):
        old = datetime.now(timezone.utc) - timedelta(days=365)
        assert temporal_decay(old, half_life_days=0) == 1.0

    def test_naive_datetime(self):
        old = datetime.now() - timedelta(days=30)
        decay = temporal_decay(old, half_life_days=30)
        assert 0.4 < decay < 0.6  # ~0.5


# --- Trust scoring ---

class TestTrustScoring:
    def test_helpful(self):
        assert adjust_trust(0.5, helpful=True) == 0.58

    def test_unhelpful(self):
        assert adjust_trust(0.5, helpful=False) == 0.40

    def test_clamp_max(self):
        assert adjust_trust(0.98, helpful=True) == 1.0

    def test_clamp_min(self):
        assert adjust_trust(0.05, helpful=False) == 0.0


# --- HybridScorer ---

class TestHybridScorer:
    def _make_entry(self, content: str, trust: float = 0.5, age_days: int = 0) -> MemoryEntry:
        created = datetime.now(timezone.utc) - timedelta(days=age_days)
        return MemoryEntry(
            id=f"test_{hash(content) % 1000}",
            content=content,
            memory_type=MemoryType.SEMANTIC,
            created_at=created,
            metadata={"trust_score": trust},
        )

    def test_basic_scoring(self):
        scorer = HybridScorer()
        entry = self._make_entry("Python programming language", trust=0.8)
        query_tokens = tokenize("Python programming")
        score = scorer.score(query_tokens, entry, fts_rank=-1.0)
        assert score > 0

    def test_trust_affects_score(self):
        scorer = HybridScorer()
        high_trust = self._make_entry("test content", trust=0.9)
        low_trust = self._make_entry("test content", trust=0.1)
        tokens = tokenize("test content")
        s1 = scorer.score(tokens, high_trust)
        s2 = scorer.score(tokens, low_trust)
        assert s1 > s2

    def test_decay_affects_score(self):
        scorer = HybridScorer(temporal_half_life_days=30)
        recent = self._make_entry("test", age_days=0)
        old = self._make_entry("test", age_days=90)
        tokens = tokenize("test")
        s1 = scorer.score(tokens, recent)
        s2 = scorer.score(tokens, old)
        assert s1 > s2

    def test_rerank(self):
        scorer = HybridScorer()
        entries = [
            self._make_entry("Python is great", trust=0.9),
            self._make_entry("Java is okay", trust=0.3),
            self._make_entry("Python programming rocks", trust=0.7),
        ]
        results = scorer.rerank("Python programming", entries, limit=2)
        assert len(results) == 2
        # Python entries should rank higher
        assert "Python" in results[0][1].content

    def test_rerank_empty(self):
        scorer = HybridScorer()
        assert scorer.rerank("test", []) == []


# --- SQLite store integration ---

class TestSQLiteStoreHybrid:
    def test_store_extracts_entities(self, tmp_path):
        from caveman.memory.sqlite_store import SQLiteMemoryStore

        store = SQLiteMemoryStore(db_path=tmp_path / "test.db")
        mid = asyncio.run(store.store(
            "Server 192.168.1.1 runs Ubuntu",
            MemoryType.SEMANTIC,
        ))
        assert mid

        conn = store._get_conn()
        row = conn.execute(
            "SELECT entities_json FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        entities = __import__("json").loads(row[0])
        assert "192.168.1.1" in entities
        store.close()

    def test_trust_feedback(self, tmp_path):
        from caveman.memory.sqlite_store import SQLiteMemoryStore

        store = SQLiteMemoryStore(db_path=tmp_path / "test.db")
        mid = asyncio.run(store.store("test memory", MemoryType.SEMANTIC))

        # Mark helpful
        asyncio.run(store.mark_helpful(mid, helpful=True))
        conn = store._get_conn()
        trust = conn.execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]
        assert trust == 0.58

        # Mark unhelpful
        asyncio.run(store.mark_helpful(mid, helpful=False))
        trust = conn.execute(
            "SELECT trust_score FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]
        assert abs(trust - 0.48) < 0.001
        store.close()

    def test_search_by_entity(self, tmp_path):
        from caveman.memory.sqlite_store import SQLiteMemoryStore

        store = SQLiteMemoryStore(db_path=tmp_path / "test.db")
        asyncio.run(store.store("Server 192.168.1.1 runs Ubuntu", MemoryType.SEMANTIC))
        asyncio.run(store.store("Server 10.0.0.1 runs CentOS", MemoryType.SEMANTIC))

        results = store.search_by_entity("192.168.1.1")
        assert len(results) == 1
        assert "Ubuntu" in results[0].content
        store.close()

    def test_hybrid_recall(self, tmp_path):
        from caveman.memory.sqlite_store import SQLiteMemoryStore

        store = SQLiteMemoryStore(db_path=tmp_path / "test.db")
        asyncio.run(store.store("Python programming language", MemoryType.SEMANTIC))
        asyncio.run(store.store("Java enterprise framework", MemoryType.SEMANTIC))
        asyncio.run(store.store("Python web development with Django", MemoryType.SEMANTIC))

        results = asyncio.run(store.recall("Python programming"))
        assert len(results) > 0
        # Python entries should be ranked higher
        assert "Python" in results[0].content
        store.close()

    def test_migration_idempotent(self, tmp_path):
        """Migration should be safe to run multiple times."""
        from caveman.memory.sqlite_store import SQLiteMemoryStore
        from caveman.memory.store_helpers import migrate_schema

        store = SQLiteMemoryStore(db_path=tmp_path / "test.db")
        conn = store._get_conn()  # First migration
        migrate_schema(conn)      # Second migration — should not error
        store.close()

    def test_retrieval_count_incremented(self, tmp_path):
        from caveman.memory.sqlite_store import SQLiteMemoryStore

        store = SQLiteMemoryStore(db_path=tmp_path / "test.db")
        mid = asyncio.run(store.store("unique test content xyz", MemoryType.SEMANTIC))

        # Recall should increment retrieval_count
        asyncio.run(store.recall("unique test content xyz"))

        conn = store._get_conn()
        count = conn.execute(
            "SELECT retrieval_count FROM memories WHERE id = ?", (mid,)
        ).fetchone()[0]
        assert count >= 1
        store.close()
