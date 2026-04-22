"""Flywheel Health Dashboard — observability for the self-improvement loop.

Aggregates metrics from all flywheel subsystems into a single health report:
  - Memory: trust distribution, decay stats, retrieval rates
  - Skills: RL Router arm stats, reflect activity
  - Wiki: compilation stats, tier distribution
  - Training: trajectory quality, contrastive pair availability
  - Event chain: handler registration, firing rates

Usage:
    python -m caveman.training.flywheel_dashboard
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caveman.paths import (
    MEMORY_DIR, TRAJECTORIES_DIR, TRAINING_DIR,
    SKILLS_DIR, WIKI_DIR,
)

logger = logging.getLogger(__name__)

__all__ = ["FlywheelDashboard", "generate_report"]


class FlywheelDashboard:
    """Collect and display flywheel health metrics."""

    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}

    def collect_memory_stats(self) -> dict:
        """Collect memory subsystem stats."""
        stats = {"total": 0, "trust_buckets": {}, "never_recalled": 0, "avg_trust": 0.0}
        db_path = MEMORY_DIR / "memories.db"
        if not db_path.exists():
            stats["status"] = "no database"
            return stats

        try:
            from caveman.db import connect as db_connect
            conn = db_connect(db_path)
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM memories")
            stats["total"] = cur.fetchone()[0]

            cur.execute("SELECT AVG(trust_score) FROM memories")
            row = cur.fetchone()
            stats["avg_trust"] = round(row[0] or 0, 3)

            cur.execute("SELECT COUNT(*) FROM memories WHERE retrieval_count = 0")
            stats["never_recalled"] = cur.fetchone()[0]

            # Trust distribution
            for label, lo, hi in [
                ("0.0-0.2", 0, 0.2), ("0.2-0.4", 0.2, 0.4),
                ("0.4-0.6", 0.4, 0.6), ("0.6-0.8", 0.6, 0.8),
                ("0.8-1.0", 0.8, 1.01),
            ]:
                cur.execute(
                    "SELECT COUNT(*) FROM memories WHERE trust_score >= ? AND trust_score < ?",
                    (lo, hi),
                )
                stats["trust_buckets"][label] = cur.fetchone()[0]

            # Decay candidates
            cur.execute(
                "SELECT COUNT(*) FROM memories WHERE trust_score < 0.05 AND retrieval_count = 0"
            )
            stats["prune_candidates"] = cur.fetchone()[0]

            conn.close()
            stats["status"] = "ok"
        except Exception as e:
            stats["status"] = f"error: {e}"

        self.metrics["memory"] = stats
        return stats

    def collect_trajectory_stats(self) -> dict:
        """Collect trajectory subsystem stats."""
        stats = {"total": 0, "with_tools": 0, "avg_quality": 0.0, "high_quality": 0, "low_quality": 0}
        traj_dir = Path(TRAJECTORIES_DIR)
        if not traj_dir.exists():
            stats["status"] = "no directory"
            self.metrics["trajectories"] = stats
            return stats

        quality_sum = 0.0
        for path in traj_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                meta = data.get("metadata", {})
                stats["total"] += 1
                tc = meta.get("tool_calls", 0)
                if tc > 0:
                    stats["with_tools"] += 1
                q = meta.get("quality_score", 0.5)
                quality_sum += q
                if q >= 0.7:
                    stats["high_quality"] += 1
                elif q <= 0.4:
                    stats["low_quality"] += 1
            except (json.JSONDecodeError, OSError):
                continue

        if stats["total"] > 0:
            stats["avg_quality"] = round(quality_sum / stats["total"], 3)
        stats["dpo_pairs_possible"] = min(stats["high_quality"], stats["low_quality"])
        stats["status"] = "ok"
        self.metrics["trajectories"] = stats
        return stats

    def collect_rl_router_stats(self) -> dict:
        """Collect RL Router arm statistics."""
        stats = {"arms": {}, "total_updates": 0}
        state_path = SKILLS_DIR / ".rl_router_state.json"
        if not state_path.exists():
            stats["status"] = "no state file"
            self.metrics["rl_router"] = stats
            return stats

        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            arms = data.get("arms", {})
            for name, arm in arms.items():
                alpha = arm.get("alpha", 1)
                beta = arm.get("beta", 1)
                total = alpha + beta - 2  # subtract priors
                win_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0
                stats["arms"][name] = {
                    "alpha": alpha, "beta": beta,
                    "updates": max(0, total),
                    "win_rate": round(win_rate, 3),
                }
                stats["total_updates"] += max(0, total)
            stats["status"] = "ok"
        except Exception as e:
            stats["status"] = f"error: {e}"

        self.metrics["rl_router"] = stats
        return stats

    def collect_wiki_stats(self) -> dict:
        """Collect wiki subsystem stats."""
        stats = {"tiers": {}, "total_entries": 0}
        wiki_dir = Path(WIKI_DIR)
        if not wiki_dir.exists():
            stats["status"] = "no directory"
            self.metrics["wiki"] = stats
            return stats

        try:
            for tier in ["working", "episodic", "semantic", "procedural"]:
                tier_file = wiki_dir / f"{tier}.json"
                if tier_file.exists():
                    data = json.loads(tier_file.read_text(encoding="utf-8"))
                    count = len(data) if isinstance(data, list) else 0
                    stats["tiers"][tier] = count
                    stats["total_entries"] += count
                else:
                    stats["tiers"][tier] = 0
            stats["status"] = "ok"
        except Exception as e:
            stats["status"] = f"error: {e}"

        self.metrics["wiki"] = stats
        return stats

    def collect_all(self) -> dict:
        """Collect all metrics."""
        self.collect_memory_stats()
        self.collect_trajectory_stats()
        self.collect_rl_router_stats()
        self.collect_wiki_stats()
        self.metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
        return self.metrics

    def diagnose(self) -> list[str]:
        """Run diagnostics and return list of issues."""
        if not self.metrics:
            self.collect_all()

        issues = []
        mem = self.metrics.get("memory", {})
        if mem.get("avg_trust", 1) < 0.3:
            issues.append(f"⚠️ Low avg memory trust ({mem['avg_trust']:.2f}) — memories may be unreliable")
        if mem.get("total", 0) > 0:
            never_pct = mem.get("never_recalled", 0) / mem["total"]
            if never_pct > 0.7:
                issues.append(f"⚠️ {never_pct:.0%} memories never recalled — retrieval may be broken")
        prune = mem.get("prune_candidates", 0)
        if prune > 100:
            issues.append(f"🗑️ {prune} memories ready for pruning — run decay")

        traj = self.metrics.get("trajectories", {})
        if traj.get("total", 0) > 100 and traj.get("with_tools", 0) == 0:
            issues.append("⚠️ No trajectories have tool_calls — backfill needed")

        rl = self.metrics.get("rl_router", {})
        if rl.get("total_updates", 0) == 0 and traj.get("total", 0) > 50:
            issues.append("⚠️ RL Router has no updates — outcome signal not flowing")

        return issues

    def format_report(self) -> str:
        """Format a human-readable health report."""
        if not self.metrics:
            self.collect_all()

        lines = ["═══ Flywheel Health Report ═══", ""]

        # Memory
        mem = self.metrics.get("memory", {})
        lines.append(f"📦 Memory: {mem.get('total', 0)} entries, avg trust={mem.get('avg_trust', 0):.2f}")
        buckets = mem.get("trust_buckets", {})
        if buckets:
            dist = " | ".join(f"{k}: {v}" for k, v in buckets.items())
            lines.append(f"   Trust distribution: {dist}")
        lines.append(f"   Never recalled: {mem.get('never_recalled', 0)}, Prune candidates: {mem.get('prune_candidates', 0)}")
        lines.append("")

        # Trajectories
        traj = self.metrics.get("trajectories", {})
        lines.append(f"📊 Trajectories: {traj.get('total', 0)} total, avg quality={traj.get('avg_quality', 0):.2f}")
        lines.append(f"   With tools: {traj.get('with_tools', 0)}, High quality: {traj.get('high_quality', 0)}, Low: {traj.get('low_quality', 0)}")
        lines.append(f"   DPO pairs possible: {traj.get('dpo_pairs_possible', 0)}")
        lines.append("")

        # RL Router
        rl = self.metrics.get("rl_router", {})
        lines.append(f"🎰 RL Router: {rl.get('total_updates', 0)} total updates")
        for name, arm in rl.get("arms", {}).items():
            lines.append(f"   {name}: win_rate={arm['win_rate']:.1%} (α={arm['alpha']}, β={arm['beta']})")
        lines.append("")

        # Wiki
        wiki = self.metrics.get("wiki", {})
        lines.append(f"📚 Wiki: {wiki.get('total_entries', 0)} entries")
        for tier, count in wiki.get("tiers", {}).items():
            lines.append(f"   {tier}: {count}")
        lines.append("")

        # Diagnostics
        issues = self.diagnose()
        if issues:
            lines.append("🔍 Issues:")
            for issue in issues:
                lines.append(f"   {issue}")
        else:
            lines.append("✅ All systems healthy")

        return "\n".join(lines)


def generate_report() -> str:
    """Generate and return a flywheel health report."""
    dashboard = FlywheelDashboard()
    return dashboard.format_report()


if __name__ == "__main__":
    print(generate_report())
