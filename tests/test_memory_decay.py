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
    helpful_count: int = 0,
    metadata: dict | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    created = now - timedelta(days=created_days_ago)
    metadata = dict(metadata or {})
    if last_accessed_days_ago is not None:
        la = now - timedelta(days=last_accessed_days_ago)
        metadata["last_accessed"] = la.isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, "
        "trust_score, retrieval_count, helpful_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (mid, content, "fact", created.isoformat(),
         json.dumps(metadata), trust, retrieval_count, helpful_count),
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
        decay.run()
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

    def test_low_signal_import_memories_are_quarantined_without_deletion(self, tmp_path):
        """Stale imported memories with no recall/helpful signal should be reversible quarantine candidates."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(
            db_path,
            "cold-import",
            trust=0.06,
            created_days_ago=120,
            retrieval_count=0,
            helpful_count=0,
            metadata={"source": "import:openclaw"},
        )
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")

        result = decay.run()

        assert result.memories_quarantined == 1
        assert result.memories_pruned == 0
        assert _get_trust(db_path, "cold-import") is not None
        conn = sqlite3.connect(str(db_path))
        meta_json, trust = conn.execute(
            "SELECT metadata_json, trust_score FROM memories WHERE id = ?",
            ("cold-import",),
        ).fetchone()
        conn.close()
        meta = json.loads(meta_json)
        assert meta["governance_state"] == "quarantined"
        assert meta["quarantine_reason"] == "stale_low_signal_import"
        assert meta["previous_trust_score"] == pytest.approx(0.06)
        assert trust == pytest.approx(0.01)

    def test_helpful_import_memories_are_not_quarantined(self, tmp_path):
        """Explicit helpful feedback should protect imported memories from quarantine."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(
            db_path,
            "helpful-import",
            trust=0.06,
            created_days_ago=120,
            retrieval_count=0,
            helpful_count=1,
            metadata={"source": "import:openclaw"},
        )
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")

        result = decay.run()

        assert result.memories_quarantined == 0
        assert _get_trust(db_path, "helpful-import") is not None
        conn = sqlite3.connect(str(db_path))
        meta_json = conn.execute(
            "SELECT metadata_json FROM memories WHERE id = ?",
            ("helpful-import",),
        ).fetchone()[0]
        conn.close()
        assert "governance_state" not in json.loads(meta_json)


    def test_import_quarantine_run_scans_enough_low_trust_rows_to_govern_bulk_imports(self, tmp_path):
        """A single decay run should govern bulk-import noise instead of tiny 500-row trickles."""
        db_path = _create_test_db(tmp_path)
        for i in range(650):
            _insert_memory(
                db_path,
                f"cold-import-{i}",
                trust=0.06,
                created_days_ago=120,
                retrieval_count=0,
                helpful_count=0,
                metadata={"source": "import:openclaw"},
            )
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")

        result = decay.run()

        assert result.memories_scanned == 650
        assert result.memories_quarantined == 650

    def test_source_aware_policy_quarantines_high_noise_imports_earlier(self, tmp_path):
        """Dashboard-proven noisy import sources should not wait 90 days before quarantine."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(
            db_path,
            "openclaw-noise",
            trust=0.07,
            created_days_ago=45,
            retrieval_count=0,
            helpful_count=0,
            metadata={"source": "import:openclaw"},
        )
        _insert_memory(
            db_path,
            "generic-import",
            trust=0.07,
            created_days_ago=45,
            retrieval_count=0,
            helpful_count=0,
            metadata={"source": "import:other"},
        )
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")

        result = decay.run()

        assert result.memories_quarantined == 1
        conn = sqlite3.connect(str(db_path))
        rows = dict(conn.execute("SELECT id, metadata_json FROM memories"))
        conn.close()
        openclaw_meta = json.loads(rows["openclaw-noise"])
        generic_meta = json.loads(rows["generic-import"])
        assert openclaw_meta["governance_state"] == "quarantined"
        assert openclaw_meta["quarantine_reason"] == "source_policy_low_signal_import"
        assert openclaw_meta["quarantine_policy"]["source"] == "import:openclaw"
        assert "governance_state" not in generic_meta

    def test_dry_run_reports_source_policy_quarantine_without_mutating_rows(self, tmp_path):
        """Source-aware dry-run should expose estimated impact while preserving DB state."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(
            db_path,
            "hermes-noise",
            trust=0.05,
            created_days_ago=45,
            retrieval_count=0,
            helpful_count=0,
            metadata={"source": "import:hermes"},
        )
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")

        result = decay.run(dry_run=True)

        assert result.memories_quarantined == 1
        assert result.quarantined_by_source == {"import:hermes": 1}
        assert result.eligible_by_source == {"import:hermes": 1}
        conn = sqlite3.connect(str(db_path))
        meta_json, trust = conn.execute(
            "SELECT metadata_json, trust_score FROM memories WHERE id = ?",
            ("hermes-noise",),
        ).fetchone()
        conn.close()
        assert "governance_state" not in json.loads(meta_json)
        assert trust == pytest.approx(0.05)

    def test_retrieved_or_helpful_noisy_source_imports_are_protected(self, tmp_path):
        """Source policy must target low-signal imports, not useful imported knowledge."""
        db_path = _create_test_db(tmp_path)
        _insert_memory(
            db_path,
            "retrieved-openclaw",
            trust=0.05,
            created_days_ago=45,
            retrieval_count=1,
            helpful_count=0,
            metadata={"source": "import:openclaw"},
        )
        _insert_memory(
            db_path,
            "helpful-hermes",
            trust=0.05,
            created_days_ago=45,
            retrieval_count=0,
            helpful_count=1,
            metadata={"source": "import:hermes"},
        )
        decay = MemoryDecay(db_path=db_path, archive_dir=tmp_path / "archive")

        result = decay.run()

        assert result.memories_quarantined == 0
        conn = sqlite3.connect(str(db_path))
        metas = [json.loads(row[0]) for row in conn.execute("SELECT metadata_json FROM memories")]
        conn.close()
        assert all("governance_state" not in meta for meta in metas)
