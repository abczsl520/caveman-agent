"""Training stats — trajectory readiness analysis."""
from __future__ import annotations
import json
from pathlib import Path


def show_training_stats(trajectory_dir: str | None, min_quality: float) -> str:
    """Analyze trajectory stats for training readiness. Returns formatted text."""
    from caveman.paths import TRAJECTORIES_DIR
    traj_dir = Path(trajectory_dir).expanduser() if trajectory_dir else TRAJECTORIES_DIR

    if not traj_dir.exists():
        return f"📂 No trajectories found at {traj_dir}"

    total = 0
    quality_bins = {"high": 0, "mid": 0, "low": 0, "unscored": 0}
    tasks: set[str] = set()
    total_turns = 0

    for f in sorted(traj_dir.rglob("*.jsonl")):
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    q = entry.get("quality_score")
                    if q is None:
                        quality_bins["unscored"] += 1
                    elif q >= min_quality:
                        quality_bins["high"] += 1
                    elif q >= 0.4:
                        quality_bins["mid"] += 1
                    else:
                        quality_bins["low"] += 1
                    tasks.add(entry.get("task", entry.get("goal", "unknown")))
                    total_turns += len(entry.get("turns", entry.get("trajectory", [])))
        except (OSError, UnicodeDecodeError) as e:
            import logging
            logging.warning("Failed to read trajectory file %s: %s", f, e)
            continue

    lines = [
        f"📊 Trajectory Stats ({traj_dir})",
        f"   Total entries: {total}",
        f"   Unique tasks: {len(tasks)}",
        f"   Total turns: {total_turns}",
        f"   Quality: ✅ high(≥{min_quality}): {quality_bins['high']} | "
        f"🟡 mid: {quality_bins['mid']} | ❌ low: {quality_bins['low']} | "
        f"❓ unscored: {quality_bins['unscored']}",
    ]
    need = 50 - quality_bins["high"]
    ready = quality_bins["high"] >= 50
    lines.append(f"   SFT ready: {'✅ Yes' if ready else f'❌ Need {need} more high-quality entries'}")
    dpo_ready = quality_bins["high"] >= 10 and quality_bins["low"] >= 10
    lines.append(f"   DPO ready: {'✅ Yes' if dpo_ready else '❌ Need both high + low quality entries'}")
    return "\n".join(lines)
