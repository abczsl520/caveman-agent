"""Caveman Flywheel — self-improvement loop.

Usage:
    caveman flywheel [--rounds N] [--target SUBSYSTEM]
    caveman flywheel --all
    caveman flywheel --parallel tools memory agent
    caveman flywheel --stats

Runs Caveman against its own codebase:
1. Audit a subsystem
2. Identify P0/P1 issues
3. Fix them
4. Run tests
5. Commit
6. Repeat
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SUBSYSTEMS = [
    "security", "tools", "memory", "agent", "compression",
    "providers", "gateway", "config", "wiki", "coordinator",
    "trajectory", "skills", "engines", "bridge", "mcp",
]

AUDIT_PROMPT = """You are Caveman, auditing YOUR OWN {subsystem} subsystem at {project_dir}/.

## Level 1: Code Quality (existing)
Read all Python files in caveman/{subsystem}/ and grep for external usage.
Audit for: dead code, missing error handling, integration gaps, data integrity.

## Level 2: Architecture (NEW — highest compound value)
- Cross-module dependencies: is this subsystem properly wired into the agent loop?
- API consistency: do function signatures match how callers actually use them?
- Missing features: what does OpenClaw/Hermes have in this area that we don't?
- Scaling bottlenecks: what breaks at 10K memories / 100 tools / 1000 sessions?

## Level 3: Observability
- Are errors logged with enough context to debug remotely?
- Are key operations timed (for metrics)?
- Can a user tell if this subsystem is working or silently failing?

Rate each finding P0/P1/P2. Be concise.

After the audit, if you find any P0 issues, fix them using file_edit.
Then run: bash -c "cd {project_dir} && {python} -m pytest tests/ -x -q --tb=short"
If tests pass, commit with: bash -c "cd {project_dir} && git add -A && git commit -m 'fix({subsystem}): <summary> (Round {{round_num}}, self-fix)'"
Report what you fixed."""

FIX_PROMPT = """You are Caveman, fixing YOUR OWN code at {project_dir}/.

The following issues were found in the {subsystem} subsystem:
{issues}

Fix the P0 issues first, then P1 if time permits.
Use file_edit for surgical changes. Keep changes minimal.
After fixing, run tests: bash -c "cd {project_dir} && {python} -m pytest tests/ -x -q --tb=short"
"""


async def run_flywheel(
    rounds: int = 5,
    target: str | None = None,
    project_dir: str | None = None,
    max_iterations: int = 15,
) -> dict:
    """Run the meta-flywheel: Caveman audits and fixes itself."""
    import sys
    from caveman.agent.factory import create_loop

    project = Path(project_dir or ".").resolve()
    results = []
    stats_tracker = FlywheelStats()

    # Detect correct python path (venv or system)
    python_path = sys.executable or "python3"

    subsystems = [target] if target else SUBSYSTEMS[:rounds]

    for i, subsystem in enumerate(subsystems):
        logger.info("Flywheel round %d: %s", i + 1, subsystem)
        round_start = time.time()

        loop = create_loop(max_iterations=max_iterations)
        prompt = AUDIT_PROMPT.format(
            subsystem=subsystem,
            project_dir=project,
            python=python_path,
        )

        try:
            result = await loop.run(prompt)
            resp = result.get("response", str(result)) if isinstance(result, dict) else str(result)
            duration = time.time() - round_start

            # Parse P0/P1/P2 counts from response
            p0 = len(re.findall(r'\bP0\b', resp))
            p1 = len(re.findall(r'\bP1\b', resp))
            p2 = len(re.findall(r'\bP2\b', resp))
            fixed = len(re.findall(r'(?:fixed|修复|✅)', resp, re.IGNORECASE))

            # Get latest commit hash
            try:
                commit = subprocess.check_output(
                    ["git", "log", "-1", "--format=%h"], cwd=str(project), text=True
                ).strip()
            except Exception:
                commit = None

            stats_tracker.record(
                round_num=i + 1, target=subsystem,
                p0_count=p0, p1_count=p1, p2_count=p2,
                fixed=fixed, duration_s=duration, commit=commit,
            )

            results.append({
                "round": i + 1,
                "subsystem": subsystem,
                "result": resp[:500],
                "success": True,
                "p0": p0, "p1": p1, "p2": p2, "fixed": fixed,
                "duration_s": round(duration, 1),
            })
        except Exception as e:
            results.append({
                "round": i + 1,
                "subsystem": subsystem,
                "error": str(e),
                "success": False,
            })

    return {
        "rounds_completed": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "results": results,
    }


def run_flywheel_sync(rounds: int = 5, target: str | None = None, max_iterations: int = 15) -> None:
    """Synchronous wrapper for CLI."""
    result = asyncio.run(run_flywheel(rounds=rounds, target=target, max_iterations=max_iterations))
    print(f"\n{'='*50}")
    print(f"Flywheel: {result['successful']}/{result['rounds_completed']} rounds successful")
    for r in result["results"]:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} Round {r['round']}: {r['subsystem']}")
        if "error" in r:
            print(f"     Error: {r['error'][:100]}")


# ── Parallel Audit Mode ──

async def run_flywheel_parallel(
    targets: list[str],
    max_iterations: int = 20,
) -> list[dict]:
    """Run multiple subsystem audits in parallel using asyncio.gather."""
    tasks = [
        run_flywheel(rounds=1, target=t, max_iterations=max_iterations)
        for t in targets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [
        r if not isinstance(r, Exception) else {"target": t, "error": str(r)}
        for t, r in zip(targets, results)
    ]


# ── Auto-Discovery ──

def discover_subsystems(project_root: Path | None = None) -> list[str]:
    """Discover all Python package directories under caveman/."""
    root = project_root or Path(__file__).resolve().parent.parent
    return sorted([
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and d.name != "__pycache__"
    ])


# ── Stats Tracker ──

class FlywheelStats:
    """Track flywheel run history for analysis."""

    def __init__(self, stats_file: Path | None = None):
        self.stats_file = stats_file or Path.home() / ".caveman" / "flywheel_stats.json"
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        round_num: int,
        target: str,
        p0_count: int,
        p1_count: int,
        p2_count: int,
        fixed: int,
        duration_s: float,
        commit: str | None = None,
    ) -> None:
        """Record a flywheel round result."""
        stats = self._load()
        stats.append({
            "round": round_num,
            "target": target,
            "p0": p0_count,
            "p1": p1_count,
            "p2": p2_count,
            "fixed": fixed,
            "duration_s": duration_s,
            "commit": commit,
            "timestamp": datetime.now().isoformat(),
        })
        self.stats_file.write_text(json.dumps(stats, indent=2))

    def summary(self) -> dict:
        """Get aggregate stats."""
        stats = self._load()
        if not stats:
            return {
                "total_rounds": 0,
                "total_p0_found": 0,
                "total_p1_found": 0,
                "total_fixed": 0,
                "avg_duration_s": 0,
                "subsystems_audited": [],
            }
        return {
            "total_rounds": len(stats),
            "total_p0_found": sum(s["p0"] for s in stats),
            "total_p1_found": sum(s["p1"] for s in stats),
            "total_fixed": sum(s["fixed"] for s in stats),
            "avg_duration_s": sum(s["duration_s"] for s in stats) / len(stats),
            "subsystems_audited": sorted(set(s["target"] for s in stats)),
        }

    def _load(self) -> list:
        if self.stats_file.exists():
            try:
                return json.loads(self.stats_file.read_text())
            except (json.JSONDecodeError, OSError):
                return []
        return []


# ── CLI handler (called from main.py) ──

def flywheel_cli(
    target: str | None = None,
    all_: bool = False,
    parallel: list[str] | None = None,
    rounds: int = 5,
    max_iter: int = 15,
    stats: bool = False,
) -> None:
    """Dispatch flywheel CLI subcommands."""
    if stats:
        s = FlywheelStats().summary()
        print(f"Flywheel Stats:")
        print(f"  Rounds: {s['total_rounds']}")
        print(f"  P0 found: {s['total_p0_found']}")
        print(f"  P1 found: {s['total_p1_found']}")
        print(f"  Fixed: {s['total_fixed']}")
        print(f"  Avg duration: {s['avg_duration_s']:.1f}s")
        print(f"  Subsystems: {', '.join(s['subsystems_audited']) or 'none'}")
        return

    if parallel:
        results = asyncio.run(run_flywheel_parallel(parallel, max_iterations=max_iter))
        for r in results:
            if "error" in r:
                print(f"  ❌ {r.get('target', '?')}: {r['error'][:100]}")
            else:
                print(f"  ✅ {r.get('successful', 0)}/{r.get('rounds_completed', 0)} rounds OK")
        return

    if all_:
        subs = discover_subsystems()
        print(f"Discovered {len(subs)} subsystems: {', '.join(subs)}")
        results = asyncio.run(run_flywheel_parallel(subs, max_iterations=max_iter))
        ok = sum(1 for r in results if not isinstance(r, dict) or "error" not in r)
        print(f"\n{ok}/{len(subs)} subsystems audited successfully")
        return

    run_flywheel_sync(rounds=rounds, target=target, max_iterations=max_iter)
