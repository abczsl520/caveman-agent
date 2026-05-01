"""Boundary tests for flywheel quarantine-preview helpers."""

import json
import sqlite3

from caveman.training.flywheel_dashboard import FlywheelDashboard


def test_restorable_quarantine_report_escapes_source_and_reason_labels(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    db_path = memory_dir / "caveman.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TEXT NOT NULL, "
        "metadata_json TEXT DEFAULT '{}', trust_score REAL DEFAULT 0.5, "
        "retrieval_count INTEGER DEFAULT 0, helpful_count INTEGER DEFAULT 0)"
    )
    unsafe_source = "import:evil\nSPOOF_SOURCE\x1b[31m"
    unsafe_reason = "manual\nSPOOF_REASON\x1b[32m"
    conn.execute(
        "INSERT INTO memories (id, content, type, created_at, metadata_json, trust_score, retrieval_count, helpful_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "q-unsafe",
            "unsafe quarantined memory",
            "semantic",
            "2026-03-16T00:00:00+00:00",
            json.dumps(
                {
                    "source": unsafe_source,
                    "governance_state": "quarantined",
                    "quarantine_reason": unsafe_reason,
                }
            ),
            0.01,
            0,
            0,
        ),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("caveman.training.flywheel_dashboard.MEMORY_DIR", memory_dir)

    stats = FlywheelDashboard().collect_memory_stats()
    formatted = FlywheelDashboard()
    formatted.metrics["memory"] = stats
    formatted.metrics["trajectories"] = {}
    formatted.metrics["rl_router"] = {}
    formatted.metrics["wiki"] = {}

    assert stats["restorable_quarantine_by_source"] == {unsafe_source: 1}
    assert stats["restorable_quarantine_by_reason"] == {unsafe_reason: 1}

    report = formatted.format_report()

    assert "Restorable quarantine: 'import:evil\\nSPOOF_SOURCE\\x1b[31m'=1" in report
    assert "Restorable quarantine reasons: 'manual\\nSPOOF_REASON\\x1b[32m'=1" in report
    assert "\nSPOOF_SOURCE" not in report
    assert "\nSPOOF_REASON" not in report
    assert "Restorable quarantine: import:evil\n" not in report
    assert "Restorable quarantine reasons: manual\n" not in report
