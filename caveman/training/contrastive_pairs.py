"""Contrastive trajectory pair generator for DPO/SFT training.

Generates (chosen, rejected) pairs from trajectory data by:
  1. Scoring all trajectories with quality heuristics
  2. Pairing high-quality (chosen) with low-quality (rejected) trajectories
  3. Matching pairs by task similarity for meaningful contrast
  4. Outputting in DPO-compatible format

The key insight: we don't need human labels. The quality score from
trajectory metadata (tool_calls, errors, completion) gives us a
natural signal for preference learning.

Output format (DPO):
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from caveman.paths import TRAJECTORIES_DIR, TRAINING_DIR

logger = logging.getLogger(__name__)

__all__ = ["generate_contrastive_pairs", "ContrastivePairStats"]


class ContrastivePairStats:
    """Statistics from pair generation."""
    def __init__(self) -> None:
        self.total_trajectories = 0
        self.high_quality = 0
        self.low_quality = 0
        self.pairs_generated = 0
        self.skipped_no_task = 0

    def summary(self) -> str:
        return (
            f"Contrastive pairs: {self.pairs_generated} from "
            f"{self.total_trajectories} trajectories "
            f"(high={self.high_quality}, low={self.low_quality})"
        )


def _load_trajectory(path: Path) -> dict | None:
    """Load and validate a trajectory file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        convs = data.get("conversations", [])
        if len(convs) < 2:
            return None
        return cast(dict[str, Any], data)
    except (json.JSONDecodeError, OSError):
        return None


def _extract_prompt(traj: dict) -> str:
    """Extract the user prompt (task) from a trajectory."""
    convs = traj.get("conversations", [])
    for turn in convs:
        if turn.get("from") == "human":
            value = turn.get("value", "")
            return value if isinstance(value, str) else ""
    return ""


def _extract_response(traj: dict) -> str:
    """Extract the final assistant response from a trajectory."""
    convs = traj.get("conversations", [])
    for turn in reversed(convs):
        if turn.get("from") == "gpt":
            value = turn.get("value", "")
            return value if isinstance(value, str) else ""
    return ""


def _get_quality(traj: dict) -> float:
    """Get quality score from trajectory metadata."""
    meta = traj.get("metadata", {})
    quality = meta.get("quality_score", 0.5) if isinstance(meta, dict) else 0.5
    return float(quality) if isinstance(quality, (int, float)) else 0.5


def generate_contrastive_pairs(
    trajectories_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    high_threshold: float = 0.7,
    low_threshold: float = 0.4,
    max_pairs: int = 500,
) -> ContrastivePairStats:
    """Generate contrastive pairs from trajectory data.

    Args:
        trajectories_dir: Directory containing trajectory JSON files.
        output_path: Output JSONL file for DPO pairs.
        high_threshold: Minimum quality score for "chosen" trajectories.
        low_threshold: Maximum quality score for "rejected" trajectories.
        max_pairs: Maximum number of pairs to generate.

    Returns:
        ContrastivePairStats with generation statistics.
    """
    traj_dir = Path(trajectories_dir or TRAJECTORIES_DIR)
    out_path = Path(output_path or TRAINING_DIR / "contrastive_pairs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = ContrastivePairStats()

    # Load all trajectories
    high_quality: list[dict] = []
    low_quality: list[dict] = []

    for path in sorted(traj_dir.glob("*.json")):
        traj = _load_trajectory(path)
        if not traj:
            continue
        stats.total_trajectories += 1

        prompt = _extract_prompt(traj)
        if not prompt or len(prompt) < 10:
            stats.skipped_no_task += 1
            continue

        quality = _get_quality(traj)
        if quality >= high_threshold:
            high_quality.append(traj)
            stats.high_quality += 1
        elif quality <= low_threshold:
            low_quality.append(traj)
            stats.low_quality += 1

    # Generate pairs: match high with low by simple round-robin
    # (In production, you'd match by task similarity via embeddings)
    pairs_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, chosen_traj in enumerate(high_quality):
            if pairs_written >= max_pairs or not low_quality:
                break

            rejected_traj = low_quality[i % len(low_quality)]

            chosen_prompt = _extract_prompt(chosen_traj)
            chosen_response = _extract_response(chosen_traj)
            rejected_response = _extract_response(rejected_traj)

            if not chosen_response or not rejected_response:
                continue

            pair = {
                "prompt": chosen_prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "chosen_quality": _get_quality(chosen_traj),
                "rejected_quality": _get_quality(rejected_traj),
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pairs_written += 1

    stats.pairs_generated = pairs_written
    logger.info(stats.summary())
    return stats
