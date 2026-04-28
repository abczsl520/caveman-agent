#!/usr/bin/env python3
"""Baseline-aware mypy gate for CI.

The repository currently has a large historical mypy baseline. This gate keeps
full baseline visibility while blocking new type errors in files changed by the
current commit/PR. It is intentionally a ratchet, not a bypass: full-project
mypy output is still emitted, and touched-file errors fail the job.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


ERROR_RE = re.compile(r"^(?P<file>[^:\n]+\.py):(?P<line>\d+): error: ")


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _python_files(files: list[str]) -> list[str]:
    return [f for f in files if f.startswith("caveman/") and f.endswith(".py")]


def changed_files() -> list[str]:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    before = os.environ.get("GITHUB_EVENT_BEFORE") or os.environ.get("GITHUB_BEFORE")

    if base_ref:
        proc = run(["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"])
        if proc.returncode == 0:
            return _python_files([line.strip() for line in proc.stdout.splitlines() if line.strip()])

    if before and before != "0000000000000000000000000000000000000000":
        proc = run(["git", "diff", "--name-only", f"{before}..HEAD"])
        if proc.returncode == 0:
            return _python_files([line.strip() for line in proc.stdout.splitlines() if line.strip()])

    # Local pre-commit path: validate uncommitted Python files when present.
    proc = run(["git", "diff", "--name-only", "HEAD"])
    files = [line.strip() for line in proc.stdout.splitlines() if line.strip()] if proc.returncode == 0 else []
    untracked = run(["git", "ls-files", "--others", "--exclude-standard"])
    if untracked.returncode == 0:
        files.extend(line.strip() for line in untracked.stdout.splitlines() if line.strip())
    current = _python_files(files)
    if current:
        return current

    proc = run(["git", "diff", "--name-only", "HEAD~1..HEAD"])
    if proc.returncode == 0:
        return _python_files([line.strip() for line in proc.stdout.splitlines() if line.strip()])

    # Shallow CI checkouts may not have parent commits. Fall back to the current
    # commit's file list so push checks remain a real ratchet instead of an
    # accidental pass.
    proc = run(["git", "show", "--name-only", "--format=", "HEAD"])
    if proc.returncode == 0:
        return _python_files([line.strip() for line in proc.stdout.splitlines() if line.strip()])
    return []


def main() -> int:
    cmd = [sys.executable, "-m", "mypy", "caveman", "--ignore-missing-imports"]
    proc = run(cmd)
    output = proc.stdout + proc.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")

    if proc.returncode != 0 and "No module named mypy" in output:
        print("mypy invocation failed: mypy is not installed", file=sys.stderr)
        return proc.returncode

    if proc.returncode == 0:
        print("mypy full-project gate passed")
        return 0

    touched = set(changed_files())
    print(f"mypy baseline-aware gate: full-project mypy failed; changed Python files={sorted(touched)}")
    if not touched:
        print("No changed caveman/*.py files detected; preserving baseline visibility without blocking.")
        return 0

    blocking: list[str] = []
    for line in output.splitlines():
        match = ERROR_RE.match(line)
        if match and Path(match.group("file")).as_posix() in touched:
            blocking.append(line)

    if blocking:
        print("\nNew/touched-file mypy errors (blocking):")
        for line in blocking:
            print(line)
        return 1

    print("No mypy errors in changed Python files; historical baseline remains visible above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
