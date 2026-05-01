"""Boundary tests for trajectory stats operator output."""

import json

from caveman.training.stats import show_training_stats


def test_training_stats_escapes_trajectory_directory_in_operator_output(tmp_path):
    unsafe_dir = tmp_path / "traj\nSPOOF_TRAJ\x1b[31m"
    unsafe_dir.mkdir()
    (unsafe_dir / "sample.jsonl").write_text(
        json.dumps(
            {
                "quality_score": 0.9,
                "task": "safe task",
                "turns": [{"role": "user"}, {"role": "assistant"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = show_training_stats(str(unsafe_dir), min_quality=0.7)

    assert "📊 Trajectory Stats (" in report
    assert "traj\\nSPOOF_TRAJ\\x1b[31m" in report
    assert "\nSPOOF_TRAJ" not in report
    assert "\x1b[31m" not in report


def test_training_stats_escapes_missing_trajectory_directory(tmp_path):
    unsafe_dir = tmp_path / "missing\nSPOOF_MISSING\x1b[32m"

    report = show_training_stats(str(unsafe_dir), min_quality=0.7)

    assert "📂 No trajectories found at " in report
    assert "missing\\nSPOOF_MISSING\\x1b[32m" in report
    assert "\nSPOOF_MISSING" not in report
    assert "\x1b[32m" not in report
