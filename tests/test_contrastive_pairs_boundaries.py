"""Regression coverage for contrastive pair type boundaries."""
from __future__ import annotations

import json
from pathlib import Path

from caveman.training.contrastive_pairs import generate_contrastive_pairs


def _write_traj(path: Path, quality: object, response: object = "assistant response") -> None:
    path.write_text(
        json.dumps(
            {
                "conversations": [
                    {"from": "human", "value": "please solve this task"},
                    {"from": "gpt", "value": response},
                ],
                "metadata": {"quality_score": quality},
            }
        ),
        encoding="utf-8",
    )


def test_generate_contrastive_pairs_handles_non_numeric_quality(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    _write_traj(traj_dir / "chosen.json", 0.9)
    _write_traj(traj_dir / "ignored.json", "high")
    _write_traj(traj_dir / "rejected.json", 0.2)
    output = tmp_path / "pairs.jsonl"

    stats = generate_contrastive_pairs(traj_dir, output)

    assert stats.total_trajectories == 3
    assert stats.high_quality == 1
    assert stats.low_quality == 1
    assert stats.pairs_generated == 1
    pair = json.loads(output.read_text(encoding="utf-8"))
    assert pair["chosen_quality"] == 0.9
    assert pair["rejected_quality"] == 0.2


def test_generate_contrastive_pairs_skips_non_string_responses(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    _write_traj(traj_dir / "chosen.json", 0.9, response={"bad": "shape"})
    _write_traj(traj_dir / "rejected.json", 0.2)
    output = tmp_path / "pairs.jsonl"

    stats = generate_contrastive_pairs(traj_dir, output)

    assert stats.pairs_generated == 0
    assert output.read_text(encoding="utf-8") == ""
