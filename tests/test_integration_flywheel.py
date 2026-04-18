"""Integration test: Flywheel Stats pipeline.

Verifies FlywheelStats recording, persistence, and summary aggregation.
"""
from __future__ import annotations

import json
import pytest

from caveman.cli.flywheel import FlywheelStats


@pytest.fixture
def flywheel(tmp_path):
    return FlywheelStats(stats_file=tmp_path / "flywheel_stats.json")


def test_record_and_summary(flywheel):
    """Record 5 rounds and verify summary aggregation."""
    rounds = [
        (1, "memory", 3, 5, 2, 8, 12.5),
        (2, "tools", 1, 3, 4, 4, 8.2),
        (3, "agent", 2, 1, 0, 3, 15.0),
        (4, "memory", 0, 2, 1, 2, 6.3),
        (5, "wiki", 1, 0, 3, 1, 9.8),
    ]
    for r, target, p0, p1, p2, fixed, dur in rounds:
        flywheel.record(r, target, p0, p1, p2, fixed, dur, commit=f"abc{r}")

    summary = flywheel.summary()
    assert summary["total_rounds"] == 5
    assert summary["total_p0_found"] == 3 + 1 + 2 + 0 + 1  # 7
    assert summary["total_p1_found"] == 5 + 3 + 1 + 2 + 0  # 11
    assert summary["total_fixed"] == 8 + 4 + 3 + 2 + 1  # 18
    assert abs(summary["avg_duration_s"] - (12.5 + 8.2 + 15.0 + 6.3 + 9.8) / 5) < 0.01
    assert set(summary["subsystems_audited"]) == {"memory", "tools", "agent", "wiki"}


def test_json_file_valid(flywheel):
    """Verify the JSON file is valid and contains all records."""
    for i in range(5):
        flywheel.record(i + 1, f"sub_{i}", i, i * 2, 0, i, float(i))

    raw = json.loads(flywheel.stats_file.read_text())
    assert isinstance(raw, list)
    assert len(raw) == 5
    for entry in raw:
        assert "round" in entry
        assert "target" in entry
        assert "p0" in entry
        assert "timestamp" in entry


def test_empty_summary(flywheel):
    """Summary of empty stats returns zeroed values."""
    summary = flywheel.summary()
    assert summary["total_rounds"] == 0
    assert summary["total_p0_found"] == 0
    assert summary["total_fixed"] == 0
    assert summary["avg_duration_s"] == 0
    assert summary["subsystems_audited"] == []


def test_incremental_recording(flywheel):
    """Records accumulate across multiple calls."""
    flywheel.record(1, "a", 1, 0, 0, 1, 5.0)
    assert flywheel.summary()["total_rounds"] == 1

    flywheel.record(2, "b", 2, 0, 0, 2, 3.0)
    assert flywheel.summary()["total_rounds"] == 2
    assert flywheel.summary()["total_fixed"] == 3


def test_stats_persist_across_instances(tmp_path):
    """Stats survive creating a new FlywheelStats instance."""
    path = tmp_path / "stats.json"
    fs1 = FlywheelStats(stats_file=path)
    fs1.record(1, "test", 5, 3, 1, 4, 10.0)

    fs2 = FlywheelStats(stats_file=path)
    summary = fs2.summary()
    assert summary["total_rounds"] == 1
    assert summary["total_p0_found"] == 5
