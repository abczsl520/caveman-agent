"""Tests for flywheel v2: parallel mode, auto-discovery, stats tracker."""
from __future__ import annotations
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from caveman.cli.flywheel import (
    discover_subsystems,
    FlywheelStats,
    run_flywheel_parallel,
)


# ── discover_subsystems ──

def test_discover_subsystems(tmp_path):
    """discover_subsystems finds Python packages (dirs with __init__.py)."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "__init__.py").touch()
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "__init__.py").touch()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "__init__.py").touch()  # should be excluded
    (tmp_path / "no_init").mkdir()  # no __init__.py → excluded
    (tmp_path / "plain_file.py").touch()  # file, not dir → excluded

    result = discover_subsystems(project_root=tmp_path)
    assert result == ["alpha", "beta"]


def test_discover_subsystems_empty(tmp_path):
    """Empty directory returns empty list."""
    assert discover_subsystems(project_root=tmp_path) == []


# ── FlywheelStats ──

def test_flywheel_stats_record_and_summary(tmp_path):
    stats_file = tmp_path / "stats.json"
    fs = FlywheelStats(stats_file=stats_file)

    fs.record(round_num=1, target="tools", p0_count=3, p1_count=2, p2_count=1, fixed=4, duration_s=12.5)
    fs.record(round_num=2, target="memory", p0_count=1, p1_count=0, p2_count=0, fixed=1, duration_s=8.0)

    s = fs.summary()
    assert s["total_rounds"] == 2
    assert s["total_p0_found"] == 4
    assert s["total_p1_found"] == 2
    assert s["total_fixed"] == 5
    assert s["avg_duration_s"] == pytest.approx(10.25)
    assert sorted(s["subsystems_audited"]) == ["memory", "tools"]


def test_flywheel_stats_empty(tmp_path):
    stats_file = tmp_path / "stats.json"
    fs = FlywheelStats(stats_file=stats_file)
    s = fs.summary()
    assert s["total_rounds"] == 0
    assert s["total_fixed"] == 0
    assert s["avg_duration_s"] == 0
    assert s["subsystems_audited"] == []


def test_flywheel_stats_corrupt_json(tmp_path):
    """Corrupt stats file doesn't crash — returns empty."""
    stats_file = tmp_path / "stats.json"
    stats_file.write_text("NOT JSON")
    fs = FlywheelStats(stats_file=stats_file)
    assert fs.summary()["total_rounds"] == 0


# ── run_flywheel_parallel ──

@pytest.mark.asyncio
async def test_run_flywheel_parallel():
    """Parallel flywheel runs multiple targets concurrently."""
    mock_result = {
        "rounds_completed": 1,
        "successful": 1,
        "results": [{"round": 1, "subsystem": "tools", "result": "ok", "success": True}],
    }

    with patch("caveman.cli.flywheel.run_flywheel", new_callable=AsyncMock, return_value=mock_result):
        results = await run_flywheel_parallel(["tools", "memory"], max_iterations=5)

    assert len(results) == 2
    assert results[0]["successful"] == 1
    assert results[1]["successful"] == 1


@pytest.mark.asyncio
async def test_run_flywheel_parallel_with_error():
    """Parallel flywheel captures exceptions per-target."""
    async def mock_flywheel(rounds=1, target=None, max_iterations=15, project_dir=None):
        if target == "bad":
            raise RuntimeError("boom")
        return {"rounds_completed": 1, "successful": 1, "results": []}

    with patch("caveman.cli.flywheel.run_flywheel", side_effect=mock_flywheel):
        results = await run_flywheel_parallel(["good", "bad"], max_iterations=5)

    assert len(results) == 2
    assert results[0]["successful"] == 1
    assert "error" in results[1]
    assert "boom" in results[1]["error"]
