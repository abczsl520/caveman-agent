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
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from caveman.memory.decay import MemoryDecay
from caveman.paths import (
    MEMORY_DIR, MEMORY_DB_PATH, TRAJECTORIES_DIR,
    SKILLS_DIR, WIKI_DIR,
)
from caveman.training._flywheel_dashboard_values import (
    _count_value as count_value,
    _number_value as number_value,
    _optional_number as optional_number,
)
from caveman.training._flywheel_memory_diagnostics import (
    _collect_memory_source_breakdown as collect_memory_source_breakdown,
    _collect_memory_source_governance as collect_memory_source_governance,
    _collect_memory_type_breakdown as collect_memory_type_breakdown,
    _memory_columns as memory_columns,
)


def _json_from_file(path: Path) -> Any | None:
    """Load JSON from disk; return None for malformed/unreadable data."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _json_object_from_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from disk; return None for malformed/non-object data."""
    data = _json_from_file(path)
    return cast(dict[str, Any], data) if isinstance(data, dict) else None


def _json_objects_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL objects from disk, skipping malformed/non-object lines."""
    entries: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    entries.append(cast(dict[str, Any], item))
    except (OSError, UnicodeDecodeError):
        return []
    return entries


logger = logging.getLogger(__name__)

__all__ = ["FlywheelDashboard", "generate_report"]


class FlywheelDashboard:
    """Collect and display flywheel health metrics."""

    def __init__(self) -> None:
        self.metrics: dict[str, Any] = {}

    def collect_memory_stats(self) -> dict:
        """Collect memory subsystem stats from the canonical SQLite store."""
        stats: dict[str, Any] = {
            "total": 0,
            "trust_buckets": {},
            "source_breakdown": [],
            "type_breakdown": [],
            "never_recalled": 0,
            "recalled": 0,
            "helpful": 0,
            "avg_trust": 0.0,
        }
        db_path = MEMORY_DIR / MEMORY_DB_PATH.name
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

            cur.execute("SELECT COUNT(*) FROM memories WHERE COALESCE(retrieval_count, 0) = 0")
            stats["never_recalled"] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memories WHERE COALESCE(retrieval_count, 0) > 0")
            stats["recalled"] = cur.fetchone()[0]

            try:
                cur.execute("SELECT COUNT(*) FROM memories WHERE COALESCE(helpful_count, 0) > 0")
                stats["helpful"] = cur.fetchone()[0]
            except Exception:
                stats["helpful"] = 0

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
                cast(dict[str, int], stats["trust_buckets"])[label] = cur.fetchone()[0]

            columns = memory_columns(cur)
            if {"type", "trust_score", "retrieval_count", "helpful_count"}.issubset(columns):
                stats["type_breakdown"] = collect_memory_type_breakdown(cur)
            if {"metadata_json", "created_at", "trust_score", "retrieval_count", "helpful_count"}.issubset(columns):
                stats["source_breakdown"] = collect_memory_source_breakdown(cur)
                stats["source_governance"] = collect_memory_source_governance(cur)
                try:
                    decay_preview = MemoryDecay(db_path=db_path).run(dry_run=True)
                except (json.JSONDecodeError, OSError, sqlite3.Error, TypeError, ValueError) as e:
                    logger.debug("Memory decay preview skipped: %s", e)
                    decay_preview = None
                if decay_preview is not None:
                    stats["decay_dry_run"] = {
                        "scanned": decay_preview.memories_scanned,
                        "would_decay": decay_preview.memories_decayed,
                        "would_prune": decay_preview.memories_pruned,
                        "would_quarantine": decay_preview.memories_quarantined,
                        "trust_total_reduced": round(decay_preview.trust_total_reduced, 3),
                        "would_quarantine_by_source": decay_preview.quarantined_by_source,
                        "eligible_by_source": decay_preview.eligible_by_source,
                    }
                cur.execute(
                    "SELECT COUNT(*) FROM memories "
                    "WHERE json_valid(metadata_json) "
                    "AND json_extract(metadata_json, '$.governance_state') = 'quarantined'"
                )
                stats["already_quarantined"] = cur.fetchone()[0]

            # Decay candidates
            cur.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE trust_score < 0.05 AND COALESCE(retrieval_count, 0) = 0"
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
        stats: dict[str, Any] = {
            "total": 0,
            "with_tools": 0,
            "avg_quality": 0.0,
            "high_quality": 0,
            "low_quality": 0,
        }
        traj_dir = Path(TRAJECTORIES_DIR)
        if not traj_dir.exists():
            stats["status"] = "no directory"
            self.metrics["trajectories"] = stats
            return stats

        quality_sum = 0.0
        for path in traj_dir.rglob("*.json"):
            data = _json_object_from_file(path)
            if data is None:
                continue
            quality_sum += self._accumulate_trajectory(stats, data)
        for path in traj_dir.rglob("*.jsonl"):
            for data in _json_objects_from_jsonl(path):
                quality_sum += self._accumulate_trajectory(stats, data)

        if stats["total"] > 0:
            stats["avg_quality"] = round(quality_sum / stats["total"], 3)
        stats["dpo_pairs_possible"] = min(stats["high_quality"], stats["low_quality"])
        stats["status"] = "ok"
        self.metrics["trajectories"] = stats
        return stats

    @staticmethod
    def _accumulate_trajectory(stats: dict[str, Any], data: dict[str, Any]) -> float:
        """Accumulate one trajectory-like object and return its quality contribution."""
        meta_value = data.get("metadata")
        meta = meta_value if isinstance(meta_value, dict) else data
        stats["total"] += 1
        tc = count_value(meta.get("tool_calls", data.get("tool_calls", 0)))
        if tc > 0:
            stats["with_tools"] += 1
        q = number_value(meta.get("quality_score", data.get("quality_score", 0.5)), 0.5)
        if q >= 0.7:
            stats["high_quality"] += 1
        elif q <= 0.4:
            stats["low_quality"] += 1
        return q

    def collect_rl_router_stats(self) -> dict:
        """Collect RL Router arm statistics."""
        stats: dict[str, Any] = {"arms": {}, "total_updates": 0}
        state_path = SKILLS_DIR / ".rl_router_state.json"
        if not state_path.exists():
            stats["status"] = "no state file"
            self.metrics["rl_router"] = stats
            return stats

        try:
            data = _json_object_from_file(state_path)
            if data is None:
                stats["status"] = "error: invalid state file"
                self.metrics["rl_router"] = stats
                return stats
            arms_value = data.get("arms")
            if arms_value is None:
                # Current SkillRLRouter persists the arm map directly as
                # {skill_name: {alpha, beta, total}}. Older/alternate snapshots
                # may wrap it under {"arms": ...}; accept both so the dashboard
                # reflects the actual feedback signal instead of reporting zero.
                arms_value = data
            arms = arms_value if isinstance(arms_value, dict) else {}
            for name, arm_value in arms.items():
                if not isinstance(arm_value, dict):
                    continue
                alpha = optional_number(arm_value.get("alpha", 1))
                beta = optional_number(arm_value.get("beta", 1))
                if alpha is None or beta is None or alpha < 0 or beta < 0:
                    continue
                total_value = optional_number(arm_value.get("total"))
                total = alpha + beta - 2 if total_value is None else total_value
                denom = alpha + beta
                win_rate = alpha / denom if denom > 0 else 0.0
                updates = max(0, int(total))
                cast(dict[str, dict[str, Any]], stats["arms"])[str(name)] = {
                    "alpha": alpha, "beta": beta,
                    "updates": updates,
                    "win_rate": round(win_rate, 3),
                }
                stats["total_updates"] += updates
            stats["status"] = "ok"
        except Exception as e:
            stats["status"] = f"error: {e}"

        self.metrics["rl_router"] = stats
        return stats

    def collect_wiki_stats(self) -> dict:
        """Collect wiki subsystem stats."""
        stats: dict[str, Any] = {"tiers": {}, "total_entries": 0}
        wiki_dir = Path(WIKI_DIR)
        if not wiki_dir.exists():
            stats["status"] = "no directory"
            self.metrics["wiki"] = stats
            return stats

        try:
            for tier in ["working", "episodic", "semantic", "procedural"]:
                tier_file = wiki_dir / f"{tier}.json"
                if tier_file.exists():
                    data = _json_from_file(tier_file)
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        entries = data.get("entries", data.get("items", []))
                        count = len(entries) if isinstance(entries, list) else 0
                    else:
                        count = 0
                    cast(dict[str, int], stats["tiers"])[tier] = count
                    stats["total_entries"] += count
                else:
                    cast(dict[str, int], stats["tiers"])[tier] = 0
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
        lines.append(
            f"   Recalled: {mem.get('recalled', 0)}, Never recalled: {mem.get('never_recalled', 0)}, "
            f"Helpful: {mem.get('helpful', 0)}, Prune candidates: {mem.get('prune_candidates', 0)}"
        )
        decay_dry_run = mem.get("decay_dry_run", {})
        if decay_dry_run:
            lines.append(
                "   Decay dry-run: "
                f"scan={decay_dry_run.get('scanned', 0)}, "
                f"would_decay={decay_dry_run.get('would_decay', 0)}, "
                f"would_prune={decay_dry_run.get('would_prune', 0)}, "
                f"would_quarantine={decay_dry_run.get('would_quarantine', 0)}"
            )
        source_breakdown = mem.get("source_breakdown", [])
        if source_breakdown:
            lines.append("   By source (top):")
            for row in source_breakdown[:6]:
                if "active" in row:
                    lines.append(
                        "      "
                        f"{row['label']}: n={row['total']}, active={row['active']}, "
                        f"quarantined={row['quarantined']}, eligible={row['eligible_for_source_policy']}, "
                        f"noise={row['never_recalled_pct'] * (1 - row['helpful_pct']):.0%}, "
                        f"recall-reduction={row['quarantined'] / row['total']:.0%}"
                    )
                    continue
                lines.append(
                    "      "
                    f"{row['label']}: n={row['total']}, avg={row['avg_trust']:.2f}, "
                    f"never={row['never_recalled_pct']:.0%}, helpful={row['helpful_pct']:.0%}"
                )
        source_governance = mem.get("source_governance", [])
        if source_governance:
            lines.append("   Source governance actions:")
            for row in source_governance[:5]:
                lines.append(
                    "      "
                    f"{row['label']}: eligible={row['eligible_for_source_policy']}, "
                    f"quarantined={row['quarantined']}, noise={row['noise_score']:.0%}"
                )
        type_breakdown = mem.get("type_breakdown", [])
        if type_breakdown:
            lines.append("   By type:")
            for row in type_breakdown:
                lines.append(
                    "      "
                    f"{row['label']}: n={row['total']}, avg={row['avg_trust']:.2f}, "
                    f"never={row['never_recalled_pct']:.0%}, helpful={row['helpful_pct']:.0%}"
                )
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
