"""Trajectory recorder v2 — record, score, export agent interactions.

Records in ShareGPT format for fine-tuning compatibility.
Supports quality scoring, filtering, and batch export.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrajectoryRecorder:
    """Records agent turns for training data generation."""

    def __init__(self, base_dir: Path | str | None = None):
        from caveman.paths import TRAJECTORIES_DIR
        self.base_dir = Path(base_dir).expanduser() if base_dir else TRAJECTORIES_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._turns: list[dict] = []
        self._session_id = str(uuid.uuid4())[:12]
        self._task: str = ""
        self._start_time: datetime = datetime.now()
        self._tool_calls: int = 0
        self._errors: int = 0
        self._quality_score: float | None = None

    async def record_turn(
        self, role: str, content: str | list, metadata: dict | None = None
    ) -> None:
        """Record a turn. role: human|gpt|function_call|function_response|system"""
        value = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        self._turns.append({
            "from": role,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })

        # Track stats
        if role == "function_call":
            self._tool_calls += 1
        if metadata and metadata.get("error"):
            self._errors += 1

    def set_task(self, task: str) -> None:
        self._task = task

    def to_sharegpt(self) -> list[dict]:
        """Convert to standard ShareGPT format (from/value only)."""
        return [{"from": t["from"], "value": t["value"]} for t in self._turns]

    def to_sharegpt_rich(self) -> dict:
        """Full format with metadata for training pipeline."""
        duration = (datetime.now() - self._start_time).total_seconds()
        return {
            "conversations": self.to_sharegpt(),
            "session_id": self._session_id,
            "task": self._task,
            "metadata": {
                "timestamp": self._start_time.isoformat(),
                "duration_seconds": round(duration, 1),
                "turn_count": len(self._turns),
                "tool_calls": self._tool_calls,
                "errors": self._errors,
                "quality_score": self._quality_score,
            },
        }

    def score_quality(self) -> float:
        """Heuristic quality score 0.0-1.0 for training data filtering."""
        if not self._turns:
            return 0.0

        score = 0.5  # baseline

        # Bonus: has tool usage (demonstrates capability)
        if self._tool_calls > 0:
            score += 0.15

        # Bonus: multi-turn conversation (shows persistence)
        if len(self._turns) >= 4:
            score += 0.1

        # Bonus: task completed (last turn is gpt with content)
        if self._turns and self._turns[-1].get("from") == "gpt":
            last_val = self._turns[-1].get("value", "")
            if len(last_val) > 20:
                score += 0.15

        # Penalty: too many errors
        if self._errors > 0:
            error_ratio = self._errors / max(len(self._turns), 1)
            score -= min(error_ratio * 0.5, 0.3)

        # Penalty: too short (might be trivial)
        if len(self._turns) <= 2:
            score -= 0.1

        self._quality_score = max(0.0, min(1.0, score))
        return self._quality_score

    async def save(self, path: Path | None = None) -> Path | None:
        """Save trajectory to disk with quality score. Returns None if empty."""
        if not self._turns:
            logger.debug("Empty trajectory, nothing to save")
            return None

        self.score_quality()
        save_path = path or self.base_dir / f"{self._session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = self.to_sharegpt_rich()
        save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return save_path

    @staticmethod
    def load(path: Path) -> dict:
        """Load a saved trajectory."""
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def batch_export(
        trajectory_dir: Path | str,
        min_quality: float = 0.5,
        output_path: Path | str | None = None,
    ) -> Path:
        """Export high-quality trajectories as JSONL for training.

        Args:
            trajectory_dir: Directory containing trajectory JSON files
            min_quality: Minimum quality score filter (0.0-1.0)
            output_path: Output JSONL file path

        Returns:
            Path to the exported JSONL file
        """
        tdir = Path(trajectory_dir).expanduser()
        out = Path(output_path) if output_path else tdir / "training_export.jsonl"

        exported = 0
        with open(out, "w", encoding="utf-8") as f:
            for p in sorted(tdir.glob("*.json")):
                if p.name.startswith("training_"):
                    continue
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    quality = data.get("metadata", {}).get("quality_score", 0.0)
                    if quality >= min_quality:
                        # Write as single-line JSONL
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        exported += 1
                except Exception as e:
                    logger.warning("Skipping corrupt trajectory %s: %s", p.name, e)

        return out
