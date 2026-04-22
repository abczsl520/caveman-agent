"""Tests for trajectory backfill script."""
import json
import pytest
from pathlib import Path

from caveman.training.backfill_trajectories import (
    backfill_tool_calls,
    _recount_tool_calls,
    _recalc_quality,
)


def test_recount_function_call():
    convs = [
        {"from": "human", "value": "do something"},
        {"from": "function_call", "value": '{"name": "search"}'},
        {"from": "function_response", "value": "results"},
        {"from": "function_call", "value": '{"name": "write"}'},
        {"from": "function_response", "value": "done"},
        {"from": "gpt", "value": "All done!"},
    ]
    assert _recount_tool_calls(convs) == 2


def test_recount_no_tools():
    convs = [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi"},
    ]
    assert _recount_tool_calls(convs) == 0


def test_recalc_quality_with_tools():
    convs = [{"from": "human", "value": "x"}] * 5 + [{"from": "gpt", "value": "a" * 30}]
    score = _recalc_quality(convs, tool_calls=3)
    assert score > 0.7  # tools + multi-turn + completion


def test_recalc_quality_no_tools():
    convs = [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}]
    score = _recalc_quality(convs, tool_calls=0)
    assert score < 0.5  # short, no tools


def test_backfill_updates_metadata(tmp_path):
    """Should update tool_calls and quality_score."""
    d = tmp_path / "traj"
    d.mkdir()

    traj = {
        "conversations": [
            {"from": "human", "value": "build feature"},
            {"from": "function_call", "value": '{"name": "search"}'},
            {"from": "function_response", "value": "found"},
            {"from": "gpt", "value": "Here is the implementation with full details."},
        ],
        "metadata": {"tool_calls": 0, "quality_score": 0.5},
    }
    (d / "test.json").write_text(json.dumps(traj))

    stats = backfill_tool_calls(trajectories_dir=d)
    assert stats.updated == 1
    assert stats.skipped == 0

    updated = json.loads((d / "test.json").read_text())
    assert updated["metadata"]["tool_calls"] == 1
    assert updated["metadata"]["tool_calls_backfilled"] is True
    assert updated["metadata"]["tool_calls_old"] == 0
    assert updated["metadata"]["quality_score"] > 0.5


def test_backfill_skips_correct(tmp_path):
    """Should skip trajectories with correct tool_calls."""
    d = tmp_path / "traj"
    d.mkdir()

    traj = {
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi"},
        ],
        "metadata": {"tool_calls": 0, "quality_score": 0.4},
    }
    (d / "correct.json").write_text(json.dumps(traj))

    stats = backfill_tool_calls(trajectories_dir=d)
    assert stats.skipped == 1
    assert stats.updated == 0


def test_backfill_dry_run(tmp_path):
    """Dry run should not modify files."""
    d = tmp_path / "traj"
    d.mkdir()

    traj = {
        "conversations": [
            {"from": "human", "value": "task"},
            {"from": "function_call", "value": "{}"},
            {"from": "function_response", "value": "ok"},
            {"from": "gpt", "value": "done"},
        ],
        "metadata": {"tool_calls": 0},
    }
    original = json.dumps(traj)
    (d / "test.json").write_text(original)

    stats = backfill_tool_calls(trajectories_dir=d, dry_run=True)
    assert stats.updated == 1
    # File should be unchanged
    assert (d / "test.json").read_text() == original


def test_backfill_old_tool_role(tmp_path):
    """Should count old 'tool' role turns too."""
    d = tmp_path / "traj"
    d.mkdir()

    traj = {
        "conversations": [
            {"from": "human", "value": "task"},
            {"from": "tool", "value": "result"},
            {"from": "gpt", "value": "done"},
        ],
        "metadata": {"tool_calls": 0},
    }
    (d / "old_format.json").write_text(json.dumps(traj))

    stats = backfill_tool_calls(trajectories_dir=d)
    assert stats.updated == 1
    updated = json.loads((d / "old_format.json").read_text())
    assert updated["metadata"]["tool_calls"] == 1
