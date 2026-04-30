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
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from caveman.operator_output import operator_literal

__all__ = [
    "SUBSYSTEMS",
    "AUDIT_PROMPT",
    "FIX_PROMPT",
    "run_flywheel",
    "run_flywheel_sync",
    "run_flywheel_parallel",
    "discover_subsystems",
    "FlywheelStats",
    "flywheel_cli",
]


logger = logging.getLogger(__name__)

SUBSYSTEMS = [
    "security", "tools", "memory", "agent", "compression",
    "providers", "gateway", "config", "wiki", "coordinator",
    "trajectory", "skills", "engines", "bridge", "mcp", "flywheel",
]

FLYWHEEL_AUDIT_PATHS = [
    "caveman/cli/flywheel.py",
    "caveman/tools/builtin/flywheel_tool.py",
    "caveman/memory/flywheel_metrics.py",
    "caveman/training/flywheel_dashboard.py",
]


def _repo_root() -> Path:
    """Return the Caveman repository root independent of caller cwd."""
    return Path(__file__).resolve().parents[2]


def _audit_paths_for_subsystem(subsystem: str) -> str:
    """Return concrete source paths for a flywheel audit target."""
    if subsystem == "flywheel":
        return "\n".join(f"- {path}" for path in FLYWHEEL_AUDIT_PATHS)
    return f"- caveman/{subsystem}/"

AUDIT_PROMPT = """You are Caveman, auditing YOUR OWN {subsystem} subsystem at {project_dir}/.

## Level 1: Code Quality (existing)
Read all Python files in the concrete {subsystem} source paths below and grep for external usage:
{audit_paths}
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

FLYWHEEL_AUDIT_PROMPT = """You are Caveman auditing the meta-flywheel subsystem itself at {project_dir}/.

Concrete files to inspect only:
{audit_paths}

Goal: produce a bounded diagnostic that can finish inside a short wall-clock budget.

Rules:
- Do not modify files in this audit round.
- Do not run the full test suite.
- Do not commit changes.
- Use at most targeted reads/searches of the concrete files above.
- Prioritize defects that prevent future flywheel runs from producing useful evidence.
- Keep the full response under 80 lines.
- End the response with exactly: END_FLYWHEEL_AUDIT

Return exactly three sections:
1. S/A findings — only real flywheel-breaking or flywheel-leaking issues.
2. Minimal fix plan — each item must name files and estimated LOC.
3. Verification plan — targeted tests/commands only.

Static pre-audit evidence:
{preaudit}
"""


def _static_flywheel_preaudit(project_dir: str | Path) -> str:
    """Return deterministic flywheel risk evidence before invoking the agent.

    Meta-flywheel self-audits should not spend their whole budget rediscovering
    the same orchestration files. This cheap pass narrows the LLM's job to
    validating/triaging concrete evidence.
    """
    project = Path(project_dir)
    lines: list[str] = []
    checks = {
        "hard_timeout_helper": ("caveman/cli/flywheel.py", "_run_round_with_hard_timeout"),
        "cli_round_timeout": ("caveman/cli/utility_commands.py", "--round-timeout"),
        "self_audit_sentinel": ("caveman/cli/flywheel.py", _FLYWHEEL_AUDIT_DONE),
        "continuation_toggle": ("caveman/agent/loop.py", "allow_continuation_repair"),
        "diagnostic_profile": ("caveman/agent/factory.py", "diagnostic_profile"),
    }
    for name, (rel, needle) in checks.items():
        try:
            present = needle in (project / rel).read_text(encoding="utf-8")
        except OSError:
            present = False
        lines.append(f"- {name}: {'present' if present else 'MISSING'} ({rel})")
    return "\n".join(lines)


def _deterministic_flywheel_self_audit(project_dir: str | Path) -> str:
    """Produce a bounded self-audit without invoking the full AgentLoop.

    This is the meta-flywheel safety valve: if provider/tool schema startup is slow,
    the flywheel must still be able to report its own orchestration health.
    """
    preaudit = _static_flywheel_preaudit(project_dir)
    missing = [line for line in preaudit.splitlines() if "MISSING" in line]
    if missing:
        finding = "P0: missing required meta-flywheel safety controls:\n" + "\n".join(missing)
        plan = "Add the missing controls in the named files; keep changes surgical."
    else:
        finding = "No P0: required meta-flywheel safety controls are present."
        plan = "No immediate code change; run targeted regression tests and a bounded CLI smoke test."
    return (
        "1. S/A findings — " + finding + "\n"
        "2. Minimal fix plan — " + plan + "\n"
        "3. Verification plan — pytest tests/test_flywheel_cli_timeout.py -q; "
        "caveman flywheel --target flywheel --rounds 1 --max-iter 80 --round-timeout 10.\n"
        f"Static pre-audit evidence:\n{preaudit}\n"
        f"{_FLYWHEEL_AUDIT_DONE}"
    )


def _audit_prompt_for_subsystem(subsystem: str, project_dir: str | Path, python_path: str) -> str:
    """Build a bounded audit prompt for a subsystem."""
    audit_paths = _audit_paths_for_subsystem(subsystem)
    if subsystem == "flywheel":
        return FLYWHEEL_AUDIT_PROMPT.format(
            subsystem=subsystem,
            audit_paths=audit_paths,
            project_dir=project_dir,
            python=python_path,
            preaudit=_static_flywheel_preaudit(project_dir),
        )
    return AUDIT_PROMPT.format(
        subsystem=subsystem,
        audit_paths=audit_paths,
        project_dir=project_dir,
        python=python_path,
    )

_TEST_PASS_RE = re.compile(
    r"(?:\b[1-9]\d*\s+passed\b|\ball\s+tests?\s+pass(?:ed)?\b|\btests?\s+pass(?:ed)?\b)",
    re.IGNORECASE,
)
_FAILURE_RE = re.compile(
    r"(?:\bFAILED\b|\bfailed\b|\btraceback\b|\berror:\s|\bexited with code\s+[1-9]|\btests?\s+fail(?:ed)?\b|\bdue\s+timeout\b|\btimeout\s*(?:error|exceeded)\b|\btimed out\b|\bno tests ran\b|\binterrupted\b|\bcancelled\b|\bkilled\b)",
    re.IGNORECASE,
)
_NO_P0_RE = re.compile(r"(?:no\s+P0|P0\s*[:=-]\s*(?:0|none|no\s+issues?))", re.IGNORECASE)
_FLYWHEEL_AUDIT_DONE = "END_FLYWHEEL_AUDIT"


def _latest_commit(project: Path) -> str | None:
    """Return current HEAD short hash, or None outside git repos."""
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%h"], cwd=str(project), text=True
        ).strip()
    except Exception:
        return None


def _evaluate_round_response(resp: str, before_commit: str | None, after_commit: str | None) -> dict:
    """Conservative flywheel round evaluator.

    A round is not successful merely because the agent returned text. It needs
    objective completion evidence: tests passed, a commit changed, or a clean
    audit that explicitly found no P0. This prevents "done/fixed" language from
    inflating flywheel progress.
    """
    mentions_no_p0 = bool(_NO_P0_RE.search(resp))
    p0_finding_re = re.compile(r'(?<!No\s)\bP0\s*[:—-]\s*(?!(?:0|none|no\b|no\s+issues?)\b)', re.IGNORECASE)
    explicit_p0_finding = bool(p0_finding_re.search(resp))
    raw_p0 = len(p0_finding_re.findall(resp))
    p0 = raw_p0
    p1 = len(re.findall(r'\bP1\b', resp))
    p2 = len(re.findall(r'\bP2\b', resp))
    tests_passed = bool(_TEST_PASS_RE.search(resp))
    failure = bool(_FAILURE_RE.search(resp))
    commit_changed = bool(before_commit and after_commit and before_commit != after_commit)

    flywheel_audit_complete = _FLYWHEEL_AUDIT_DONE in resp
    clean_no_p0 = mentions_no_p0 and p0 == 0 and not explicit_p0_finding
    flywheel_audit_clean = flywheel_audit_complete and p0 == 0 and not explicit_p0_finding and not failure and "MISSING" not in resp
    success = (not failure) and (commit_changed or tests_passed or clean_no_p0 or flywheel_audit_clean)

    fixed = 1 if commit_changed else 0
    return {
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "fixed": fixed,
        "success": success,
        "tests_passed": tests_passed,
        "commit_changed": commit_changed,
        "failure_detected": failure,
        "explicit_no_p0": mentions_no_p0,
        "explicit_flywheel_audit_complete": flywheel_audit_complete,
    }


async def _run_round_with_hard_timeout(awaitable, timeout_s: float | None):
    """Await a flywheel round without waiting forever for cancellation cleanup.

    `asyncio.wait_for` cancels the underlying task and then waits until that
    cancellation completes. If AgentLoop/tool cleanup ignores cancellation,
    wait_for can exceed the advertised timeout indefinitely. This helper gives
    the CLI a hard observation boundary: return TimeoutError at timeout, cancel
    best-effort in the background, and let run_flywheel record objective
    failure evidence.
    """
    if timeout_s is None:
        return await awaitable
    task = asyncio.create_task(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=timeout_s)
    if task in done:
        return await task
    task.cancel()
    task.add_done_callback(_consume_task_result)
    raise asyncio.TimeoutError


def _consume_task_result(task: asyncio.Task) -> None:
    """Best-effort retrieval of late task failures without blocking shutdown."""
    if not task.done() or task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


async def run_flywheel(
    rounds: int = 5,
    target: str | None = None,
    project_dir: str | None = None,
    max_iterations: int = 50,
    round_timeout_s: float | None = 900,
) -> dict:
    """Run the meta-flywheel: Caveman audits and fixes itself.

    round_timeout_s bounds each AgentLoop round so a wedged self-audit returns
    objective failure evidence instead of leaving the CLI black-box hung.
    """
    import sys
    from caveman.agent.factory import create_loop

    project = Path(project_dir).resolve() if project_dir else _repo_root()
    results = []
    stats_tracker = FlywheelStats()

    # Detect correct python path (venv or system)
    python_path = sys.executable or "python3"

    subsystems = [target] if target else SUBSYSTEMS[:rounds]

    for i, subsystem in enumerate(subsystems):
        logger.info("Flywheel round %d: %s", i + 1, subsystem)
        round_start = time.time()

        try:
            before_commit = _latest_commit(project)
            if subsystem == "flywheel":
                resp = _deterministic_flywheel_self_audit(project)
            else:
                loop = create_loop(
                    max_iterations=max_iterations,
                    allow_continuation_repair=True,
                    diagnostic_profile=False,
                )
                prompt = _audit_prompt_for_subsystem(
                    subsystem=subsystem,
                    project_dir=project,
                    python_path=python_path,
                )
                result = await _run_round_with_hard_timeout(loop.run(prompt), round_timeout_s)
                resp = result.get("response", str(result)) if isinstance(result, dict) else str(result)
            duration = time.time() - round_start

            after_commit = _latest_commit(project)
            evaluation = _evaluate_round_response(resp, before_commit, after_commit)
            p0 = evaluation["p0"]
            p1 = evaluation["p1"]
            p2 = evaluation["p2"]
            fixed = evaluation["fixed"]
            commit = after_commit

            stats_tracker.record(
                round_num=i + 1, target=subsystem,
                p0_count=p0, p1_count=p1, p2_count=p2,
                fixed=fixed, duration_s=duration, commit=commit,
            )

            results.append({
                "round": i + 1,
                "subsystem": subsystem,
                "result": resp[:500],
                "success": evaluation["success"],
                "p0": p0, "p1": p1, "p2": p2, "fixed": fixed,
                "tests_passed": evaluation["tests_passed"],
                "commit_changed": evaluation["commit_changed"],
                "failure_detected": evaluation["failure_detected"],
                "duration_s": round(duration, 1),
            })
        except asyncio.TimeoutError:
            duration = time.time() - round_start
            timeout_desc = "unbounded" if round_timeout_s is None else f"{round_timeout_s:g}s"
            logger.warning(
                "Flywheel round %d timed out after %s: %s",
                i + 1,
                timeout_desc,
                subsystem,
            )
            results.append({
                "round": i + 1,
                "subsystem": subsystem,
                "error": f"Round timed out after {timeout_desc}",
                "success": False,
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


def run_flywheel_sync(
    rounds: int = 5,
    target: str | None = None,
    max_iterations: int = 50,
    project_dir: str | None = None,
    round_timeout_s: float | None = 900,
) -> None:
    """Synchronous wrapper for CLI."""
    result = asyncio.run(
        run_flywheel(
            rounds=rounds,
            target=target,
            project_dir=project_dir or str(_repo_root()),
            max_iterations=max_iterations,
            round_timeout_s=round_timeout_s,
        )
    )
    print(f"\n{'='*50}")
    print(f"Flywheel: {result['successful']}/{result['rounds_completed']} rounds successful")
    for r in result["results"]:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} Round {r['round']}: {r['subsystem']}")
        if "error" in r:
            print(f"     Error: {r['error'][:100]}")
    if result["successful"] < result["rounds_completed"]:
        raise SystemExit(1)


# ── Parallel Audit Mode ──

async def run_flywheel_parallel(
    targets: list[str],
    max_iterations: int = 20,
    round_timeout_s: float | None = 900,
) -> list[dict[str, Any]]:
    """Run multiple subsystem audits in parallel using asyncio.gather."""
    tasks = [
        run_flywheel(
            rounds=1,
            target=t,
            max_iterations=max_iterations,
            round_timeout_s=round_timeout_s,
        )
        for t in targets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    typed_results: list[dict[str, Any]] = []
    for target, result in zip(targets, results):
        if isinstance(result, BaseException):
            typed_results.append({
                "target": target,
                "error": str(result),
                "success": False,
                "successful": 0,
                "rounds_completed": 0,
                "results": [],
            })
        else:
            typed_results.append(cast(dict[str, Any], result))
    return typed_results


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
        from caveman.paths import CAVEMAN_HOME
        self.stats_file = stats_file or CAVEMAN_HOME / "flywheel_stats.json"
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

    def _load(self) -> list[Any]:
        if self.stats_file.exists():
            try:
                return cast(list[Any], json.loads(self.stats_file.read_text()))
            except (json.JSONDecodeError, OSError):
                return []
        return []


# ── CLI handler (called from main.py) ──

def flywheel_cli(
    target: str | None = None,
    all_: bool = False,
    parallel: list[str] | None = None,
    rounds: int = 5,
    max_iter: int = 50,
    round_timeout_s: float | None = 900,
    stats: bool = False,
) -> None:
    """Dispatch flywheel CLI subcommands."""
    if stats:
        s = FlywheelStats().summary()
        print("Flywheel Stats:")
        print(f"  Rounds: {s['total_rounds']}")
        print(f"  P0 found: {s['total_p0_found']}")
        print(f"  P1 found: {s['total_p1_found']}")
        print(f"  Fixed: {s['total_fixed']}")
        print(f"  Avg duration: {s['avg_duration_s']:.1f}s")
        subsystems = ", ".join(operator_literal(source) for source in s["subsystems_audited"])
        print(f"  Subsystems: {subsystems or 'none'}")
        return

    if parallel:
        results = asyncio.run(run_flywheel_parallel(parallel, max_iterations=max_iter, round_timeout_s=round_timeout_s))
        for r in results:
            if "error" in r:
                print(f"  ❌ {r.get('target', '?')}: {r['error'][:100]}")
            else:
                print(f"  ✅ {r.get('successful', 0)}/{r.get('rounds_completed', 0)} rounds OK")
        return

    if all_:
        subs = discover_subsystems()
        print(f"Discovered {len(subs)} subsystems: {', '.join(subs)}")
        results = asyncio.run(run_flywheel_parallel(subs, max_iterations=max_iter, round_timeout_s=round_timeout_s))
        ok = sum(1 for r in results if not isinstance(r, dict) or "error" not in r)
        print(f"\n{ok}/{len(subs)} subsystems audited successfully")
        return

    run_flywheel_sync(
        rounds=rounds,
        target=target,
        max_iterations=max_iter,
        project_dir=str(_repo_root()),
        round_timeout_s=round_timeout_s,
    )
