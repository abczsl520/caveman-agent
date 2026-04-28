"""Tests for FlywheelDashboard — observability for the self-improvement loop."""
import json
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from caveman.training.flywheel_dashboard import FlywheelDashboard


@pytest.fixture
def dashboard():
    return FlywheelDashboard()


def test_collect_memory_stats_no_db(dashboard, tmp_path):
    """Should handle missing database gracefully."""
    with patch("caveman.training.flywheel_dashboard.MEMORY_DIR", tmp_path):
        stats = dashboard.collect_memory_stats()
    assert stats["status"] == "no database"


def test_collect_memory_stats_uses_canonical_caveman_db(dashboard, tmp_path):
    """Dashboard should read the same caveman.db that MemoryDecay and stores use."""
    db_path = tmp_path / "caveman.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE memories (
        id TEXT, content TEXT, trust_score REAL, retrieval_count INTEGER,
        helpful_count INTEGER DEFAULT 0
    )""")
    conn.execute("INSERT INTO memories VALUES ('m1', 'useful', 0.8, 5, 1)")
    conn.execute("INSERT INTO memories VALUES ('m2', 'stale', 0.1, 0, 0)")
    conn.commit()
    conn.close()

    with patch("caveman.training.flywheel_dashboard.MEMORY_DIR", tmp_path):
        stats = dashboard.collect_memory_stats()

    assert stats["status"] == "ok"
    assert stats["total"] == 2
    assert stats["never_recalled"] == 1
    assert stats["recalled"] == 1
    assert stats["helpful"] == 1


def test_collect_memory_stats_with_db(dashboard, tmp_path):
    """Should read real SQLite stats."""
    db_path = tmp_path / "caveman.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE memories (
        id TEXT, content TEXT, trust_score REAL, retrieval_count INTEGER
    )""")
    conn.execute("INSERT INTO memories VALUES ('m1', 'test', 0.8, 5)")
    conn.execute("INSERT INTO memories VALUES ('m2', 'test2', 0.1, 0)")
    conn.execute("INSERT INTO memories VALUES ('m3', 'test3', 0.5, 2)")
    conn.commit()
    conn.close()

    with patch("caveman.training.flywheel_dashboard.MEMORY_DIR", tmp_path):
        stats = dashboard.collect_memory_stats()
    assert stats["total"] == 3
    assert stats["never_recalled"] == 1
    assert 0.4 < stats["avg_trust"] < 0.5
    assert stats["status"] == "ok"


def test_collect_trajectory_stats(dashboard, tmp_path):
    """Should count trajectory quality distribution."""
    for i in range(5):
        q = 0.9 if i < 3 else 0.2
        traj = {"conversations": [{"from": "human", "value": "x"}],
                "metadata": {"quality_score": q, "tool_calls": 1 if i < 3 else 0}}
        (tmp_path / f"t_{i}.json").write_text(json.dumps(traj))

    with patch("caveman.training.flywheel_dashboard.TRAJECTORIES_DIR", tmp_path):
        stats = dashboard.collect_trajectory_stats()
    assert stats["total"] == 5
    assert stats["high_quality"] == 3
    assert stats["low_quality"] == 2
    assert stats["with_tools"] == 3


def test_collect_rl_router_stats(dashboard, tmp_path):
    """Should parse RL Router state."""
    state = {"arms": {
        "search": {"alpha": 5, "beta": 2},
        "code": {"alpha": 1, "beta": 1},
    }}
    (tmp_path / ".rl_router_state.json").write_text(json.dumps(state))

    with patch("caveman.training.flywheel_dashboard.SKILLS_DIR", tmp_path):
        stats = dashboard.collect_rl_router_stats()
    assert stats["arms"]["search"]["win_rate"] > 0.5
    assert stats["total_updates"] == 5  # (5+2-2) + (1+1-2)


def test_diagnose_issues(dashboard, tmp_path):
    """Should detect flywheel issues."""
    dashboard.metrics = {
        "memory": {"total": 100, "avg_trust": 0.2, "never_recalled": 80, "prune_candidates": 150},
        "trajectories": {"total": 200, "with_tools": 0},
        "rl_router": {"total_updates": 0},
        "wiki": {},
    }
    issues = dashboard.diagnose()
    assert len(issues) >= 3  # low trust, never recalled, no tools


def test_format_report(dashboard):
    """Should produce a formatted report string."""
    dashboard.metrics = {
        "memory": {"total": 10, "avg_trust": 0.6, "trust_buckets": {"0.4-0.6": 5, "0.6-0.8": 5},
                    "never_recalled": 2, "prune_candidates": 0},
        "trajectories": {"total": 50, "avg_quality": 0.65, "with_tools": 30,
                         "high_quality": 20, "low_quality": 10, "dpo_pairs_possible": 10},
        "rl_router": {"total_updates": 15, "arms": {"search": {"alpha": 5, "beta": 2, "win_rate": 0.714}}},
        "wiki": {"total_entries": 5, "tiers": {"working": 3, "episodic": 2}},
        "timestamp": "2026-04-22T10:00:00+00:00",
    }
    report = dashboard.format_report()
    assert "Flywheel Health Report" in report
    assert "Memory" in report
    assert "RL Router" in report
