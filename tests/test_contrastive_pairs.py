"""Tests for contrastive trajectory pair generation."""
import json
import pytest
from pathlib import Path

from caveman.training.contrastive_pairs import (
    generate_contrastive_pairs,
    ContrastivePairStats,
    _extract_prompt,
    _extract_response,
    _get_quality,
)


def _make_trajectory(task: str, response: str, quality: float, tool_calls: int = 0) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": task},
            {"from": "gpt", "value": response},
        ],
        "metadata": {
            "quality_score": quality,
            "tool_calls": tool_calls,
        },
    }


@pytest.fixture
def traj_dir(tmp_path):
    """Create a directory with mixed-quality trajectories."""
    d = tmp_path / "trajectories"
    d.mkdir()

    # High quality
    for i in range(5):
        t = _make_trajectory(
            f"Implement feature {i} with proper error handling",
            f"Here's the implementation with tests and error handling for feature {i}...",
            quality=0.85,
            tool_calls=3,
        )
        (d / f"high_{i}.json").write_text(json.dumps(t))

    # Low quality
    for i in range(3):
        t = _make_trajectory(
            f"Please complete this task number {i} for me",
            "ok",
            quality=0.2,
        )
        (d / f"low_{i}.json").write_text(json.dumps(t))

    # Medium quality (should be excluded from pairs)
    for i in range(2):
        t = _make_trajectory(
            f"Medium task {i}",
            f"Medium response {i}",
            quality=0.55,
        )
        (d / f"med_{i}.json").write_text(json.dumps(t))

    return d


def test_generate_pairs(traj_dir, tmp_path):
    """Should generate pairs from high/low quality trajectories."""
    out = tmp_path / "pairs.jsonl"
    stats = generate_contrastive_pairs(
        trajectories_dir=traj_dir,
        output_path=out,
    )
    assert stats.total_trajectories == 10
    assert stats.high_quality == 5
    assert stats.low_quality == 3
    assert stats.pairs_generated == 5

    # Verify output format
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 5
    pair = json.loads(lines[0])
    assert "prompt" in pair
    assert "chosen" in pair
    assert "rejected" in pair
    assert pair["chosen_quality"] >= 0.7
    assert pair["rejected_quality"] <= 0.4


def test_max_pairs_limit(traj_dir, tmp_path):
    """Should respect max_pairs limit."""
    out = tmp_path / "pairs.jsonl"
    stats = generate_contrastive_pairs(
        trajectories_dir=traj_dir,
        output_path=out,
        max_pairs=2,
    )
    assert stats.pairs_generated == 2


def test_empty_directory(tmp_path):
    """Should handle empty trajectory directory."""
    d = tmp_path / "empty"
    d.mkdir()
    out = tmp_path / "pairs.jsonl"
    stats = generate_contrastive_pairs(
        trajectories_dir=d,
        output_path=out,
    )
    assert stats.total_trajectories == 0
    assert stats.pairs_generated == 0


def test_no_low_quality(tmp_path):
    """Should generate 0 pairs if no low-quality trajectories."""
    d = tmp_path / "all_high"
    d.mkdir()
    for i in range(3):
        t = _make_trajectory(f"Good task {i}", f"Good response {i}", quality=0.9)
        (d / f"t_{i}.json").write_text(json.dumps(t))

    out = tmp_path / "pairs.jsonl"
    stats = generate_contrastive_pairs(trajectories_dir=d, output_path=out)
    assert stats.pairs_generated == 0


def test_extract_prompt():
    t = _make_trajectory("hello world", "response", 0.5)
    assert _extract_prompt(t) == "hello world"


def test_extract_response():
    t = _make_trajectory("task", "final answer", 0.5)
    assert _extract_response(t) == "final answer"


def test_get_quality():
    t = _make_trajectory("task", "resp", 0.75)
    assert _get_quality(t) == 0.75


def test_corrupt_files_skipped(tmp_path):
    """Should skip corrupt JSON files."""
    d = tmp_path / "corrupt"
    d.mkdir()
    (d / "bad.json").write_text("not json")
    (d / "good.json").write_text(json.dumps(
        _make_trajectory("valid task here", "valid response", 0.8)
    ))
    out = tmp_path / "pairs.jsonl"
    stats = generate_contrastive_pairs(trajectories_dir=d, output_path=out)
    assert stats.total_trajectories == 1
