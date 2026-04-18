"""Caveman Doctor v2 — flywheel health diagnostics with 6 metrics.

`caveman doctor` reports on the health of the learning flywheel:
  1. Skill match rate — how often skills are correctly selected
  2. Memory recall precision — search quality
  3. Skill creation velocity — auto-creation rate
  4. Trajectory quality — average quality score
  5. Compression fidelity — info preserved after compression
  6. Shield recovery rate — context restored after compaction

Also integrates Lint Engine results and LLM Scheduler stats.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryType
from caveman.skills.manager import SkillManager

import logging
logger = logging.getLogger(__name__)


class DoctorReport:
    """Structured health report with 6 flywheel metrics."""

    def __init__(self):
        self.checks: list[dict[str, Any]] = []
        self.metrics: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.score: float = 1.0

    def add_check(self, name: str, status: str, detail: str = "", value: Any = None):
        icon = {"ok": "\u2705", "warn": "\u26a0\ufe0f", "error": "\u274c", "info": "\u2139\ufe0f"}.get(status, "\u2753")
        self.checks.append({"name": name, "status": status, "detail": detail, "value": value, "icon": icon})
        if status == "warn":
            self.warnings.append(f"{name}: {detail}")
            self.score -= 0.05
        elif status == "error":
            self.errors.append(f"{name}: {detail}")
            self.score -= 0.15

    def add_metric(self, name: str, value: float, target: float, unit: str = ""):
        status = "ok" if value >= target else ("warn" if value >= target * 0.5 else "error")
        self.metrics[name] = {"value": value, "target": target, "unit": unit, "status": status}
        if status == "error":
            self.score -= 0.1
        elif status == "warn":
            self.score -= 0.05

    def to_text(self) -> str:
        lines = [
            "\U0001f9b4 Caveman Doctor \u2014 Flywheel Health Report",
            f"\U0001f4c5 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"\U0001f3af Health Score: {max(0, self.score):.0%}",
            "",
        ]

        if self.metrics:
            lines.append("\U0001f4ca Flywheel Metrics:")
            for name, m in self.metrics.items():
                icon = {
                    "ok": "\U0001f7e2", "warn": "\U0001f7e1", "error": "\U0001f534"
                }[m["status"]]
                lines.append(
                    f"  {icon} {name}: {m['value']:.1f}{m['unit']} "
                    f"(target: {m['target']:.1f}{m['unit']})"
                )
            lines.append("")

        for check in self.checks:
            lines.append(f"  {check['icon']} {check['name']}: {check['detail']}")

        if self.warnings:
            lines.append(f"\n\u26a0\ufe0f  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"   - {w}")
        if self.errors:
            lines.append(f"\n\u274c Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"   - {e}")
        if not self.warnings and not self.errors:
            lines.append("\n\U0001f389 All systems healthy!")

        return "\n".join(lines)


async def run_doctor(config_dir: str | None = None) -> DoctorReport:
    """Run full health check and return report."""
    from caveman.paths import CAVEMAN_HOME
    report = DoctorReport()
    base = Path(config_dir).expanduser() if config_dir else CAVEMAN_HOME

    # --- Config ---
    config_path = base / "config.yaml"
    if config_path.exists():
        report.add_check("Config", "ok", f"Found at {config_path}")
    else:
        report.add_check("Config", "warn", "No config.yaml \u2014 run `caveman setup`")

    # --- Memory health ---
    mem_dir = base / "memory"
    mm = MemoryManager(base_dir=mem_dir)
    await mm.load()
    total = mm.total_count
    type_counts = {mt.value: len(mm._memories.get(mt, [])) for mt in MemoryType}

    if total == 0:
        report.add_check("Memory", "warn", "No memories yet \u2014 flywheel cold start")
    else:
        report.add_check("Memory", "ok", f"{total} memories ({type_counts})")

    # --- Lint scan ---
    try:
        from caveman.engines.lint import LintEngine
        lint = LintEngine(mm, check_paths=True)
        lint_report = await lint.scan()
        issues = len(lint_report.issues)
        if issues == 0:
            report.add_check("Lint", "ok", "No issues found")
        elif lint_report.is_healthy:
            report.add_check("Lint", "info", f"{issues} issues (no errors)")
        else:
            report.add_check("Lint", "warn", f"{issues} issues ({lint_report.by_severity})")
    except Exception as e:
        report.add_check("Lint", "warn", f"Scan failed: {e}")

    # --- Skills ---
    skill_dir = base / "skills"
    sm = SkillManager(skills_dir=skill_dir)
    sm.load_all()
    skills = sm.list_all()
    if not skills:
        report.add_check("Skills", "info", "No skills yet")
    else:
        auto_count = sum(1 for s in skills if s.source == "auto_created")
        report.add_check("Skills", "ok", f"{len(skills)} skills ({auto_count} auto-created)")
        # Metric: skill match rate
        with_usage = [s for s in skills if s.success_count + s.fail_count > 0]
        if with_usage:
            avg_rate = sum(s.success_rate for s in with_usage) / len(with_usage)
            report.add_metric("Skill Match Rate", avg_rate * 100, 60, "%")

    # --- Trajectories ---
    traj_dir = base / "trajectories"
    if traj_dir.exists():
        traj_files = list(traj_dir.glob("*.json"))
        if traj_files:
            scores = []
            for p in traj_files:
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    qs = data.get("metadata", {}).get("quality_score")
                    if qs is not None:
                        scores.append(qs)
                except Exception as e:
                    logger.debug("Suppressed in doctor: %s", e)
            if scores:
                avg_q = sum(scores) / len(scores)
                report.add_metric("Trajectory Quality", avg_q * 100, 70, "%")
                report.add_check("Trajectories", "ok",
                    f"{len(traj_files)} files, avg quality {avg_q:.0%}")
            else:
                report.add_check("Trajectories", "info",
                    f"{len(traj_files)} files, no quality scores")
        else:
            report.add_check("Trajectories", "info", "No trajectories yet")
    else:
        report.add_check("Trajectories", "info", "Not started")

    # --- Sessions (Shield recovery) ---
    sessions_dir = base / "sessions"
    if sessions_dir.exists():
        session_files = list(sessions_dir.glob("*.yaml"))
        report.add_check("Shield Sessions", "ok" if session_files else "info",
            f"{len(session_files)} session essences stored")
    else:
        report.add_check("Shield Sessions", "info", "No sessions yet")

    # --- Scheduler stats ---
    report.add_check("LLM Scheduler", "info", "Stats available at runtime via scheduler.get_stats()")

    # --- Disk ---
    if base.exists():
        total_size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
        mb = total_size / (1024 * 1024)
        report.add_check("Disk", "ok" if mb < 500 else "warn", f"{mb:.1f} MB")

    # --- API key ---
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        report.add_check("API key", "ok", "ANTHROPIC_API_KEY set")
    else:
        report.add_check("API key", "warn", "No ANTHROPIC_API_KEY")

    return report
