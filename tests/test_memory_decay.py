"""Tests for memory decay engine."""
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from caveman.memory.decay import MemoryDecay, DecayResult


def _create_test_db(tmp_path: Path) -> Path:
    """Create a test memory DB with known data."""
    db_path = tmp_path / "caveman.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            entities_json TEXT DEFAULT '[]'
        )
    """)
    return db_path


def _insert_memory(
    db_path: Path,
    mid: str,
    content: str = "test",
    trust: float = 0.5,
    created_days_ago: int = 60,
    last_accessed_days_ago: int | None = None,
    retrieval_count: int = 0,
) -> None:
    now = datetime.now(timezone.utc)
    created = now - timedelta(days=created_days_ago)
    metadata = {}
    if last_accessed_days_ago is not None:
        la = now - timedelta(days=last_accessed_days_ago)
        metadata["last_accessed"] = la.isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, "
        "trust_score, retrieval_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (mid, content, "fact", created.isoformat(),
         json.dumps(metadata), trust, retrieval_count),
    )
    conn.commit()
    conn.close()


def _get_trust(db_path: Path, mid: str) -> float | None:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT trust_score FROM memories WHERE id = ?", (mid,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


class TestMemoryDecay:
    def test_no_db(self, tmp_path):
        decay = MemoryDecay(db_path=tmp_path / "nonexistent.db")
        result = decay.run()
        assert result.memories_scanned == 0

    def test_fresh_memories_immune(self, tmp_path):
        """Memories created <30 days ago should not decay."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "fresh", trust=0.5, created_days_ago=10)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        result = decay.run()
        assert result.memories_decayed == 0
        assert _get_trust(db_path, "fresh") == 0.5

    def test_recently_accessed_immune(self, tmp_path):
        """Memories accessed within 14 days should not decay."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "recent", trust=0.5,
                       created_days_ago=120, last_accessed_days_ago=5)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        result = decay.run()
        assert result.memories_decayed == 0
        assert _get_trust(db_path, "recent") == 0.5

    def test_old_unused_memory_decays(self, tmp_path):
        """Memories unused for >30 days should lose trust."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "old", trust=0.5, created_days_ago=60)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        result = decay.run()
        assert result.memories_decayed == 1
        new_trust = _get_trust(db_path, "old")
        assert new_trust is not None
        assert new_trust < 0.5

    def test_high_trust_decays_slower(self, tmp_path):
        """High-trust memories should decay 3x slower."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "high", trust=0.8, created_days_ago=60)
        _insert_memory(db_path, "normal", trust=0.5, created_days_ago=60)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        result = decay.run()
        high_trust = _get_trust(db_path, "high")
        normal_trust = _get_trust(db_path, "normal")
        # High trust should have lost less
        high_loss = 0.8 - high_trust
        normal_loss = 0.5 - normal_trust
        assert high_loss < normal_loss

    def test_prune_old_zero_retrieval(self, tmp_path):
        """Very old, never-retrieved, near-zero trust → pruned."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "garbage", trust=0.01,
                       created_days_ago=120, retrieval_count=0)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        result = decay.run()
        assert result.memories_pruned == 1
        assert _get_trust(db_path, "garbage") is None  # deleted
        # Check archive exists
        archive_files = list((tmp_path / "archive").glob("pruned_*.jsonl"))
        assert len(archive_files) == 1

    def test_no_prune_if_retrieved(self, tmp_path):
        """Low trust but retrieved → should NOT be pruned."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "used", trust=0.01,
                       created_days_ago=120, retrieval_count=3)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        result = decay.run()
        assert result.memories_pruned == 0
        assert _get_trust(db_path, "used") is not None

    def test_dry_run(self, tmp_path):
        """Dry run should not modify anything."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "old", trust=0.5, created_days_ago=60)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        result = decay.run(dry_run=True)
        assert result.memories_decayed == 1
        assert _get_trust(db_path, "old") == 0.5  # unchanged

    def test_decay_result_summary(self):
        r = DecayResult(memories_scanned=100, memories_decayed=20,
                        memories_pruned=5, trust_total_reduced=1.234)
        s = r.summary()
        assert "scanned=100" in s
        assert "decayed=20" in s
        assert "pruned=5" in s

    def test_high_retrieval_count_slows_decay(self, tmp_path):
        """Memories with >10 retrievals should decay 50% slower."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(db_path, "popular", trust=0.5,
                       created_days_ago=60, retrieval_count=15)
        _insert_memory(db_path, "unpopular", trust=0.5,
                       created_days_ago=60, retrieval_count=0)
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")
        decay.run()
        pop_trust = _get_trust(db_path, "popular")
        unpop_trust = _get_trust(db_path, "unpopular")
        assert pop_trust > unpop_trust
