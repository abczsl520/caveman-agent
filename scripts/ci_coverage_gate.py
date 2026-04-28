#!/usr/bin/env python3
"""Baseline-aware coverage gate for CI.

The project has historical coverage debt below the long-term 80% target. This
gate preserves full coverage visibility while preventing regressions: tests must
pass, coverage must not drop below the configured baseline, and the aspirational
target remains printed for tracking.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BASELINE = float(os.environ.get("CAVEMAN_COVERAGE_BASELINE", "68.25"))
TARGET = float(os.environ.get("CAVEMAN_COVERAGE_TARGET", "80.0"))
COVERAGE_JSON = Path("coverage.json")


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=short",
        "-q",
        "--cov=caveman",
        "--cov-report=term-missing",
        "--cov-report=json:coverage.json",
        "--cov-fail-under=0",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if "unrecognized arguments: --cov" in proc.stdout:
        print("pytest-cov is required for scripts/ci_coverage_gate.py", file=sys.stderr)
        return 2
    if proc.returncode != 0:
        return proc.returncode

    try:
        data = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))
        coverage = float(data["totals"]["percent_covered"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not parse exact TOTAL coverage from coverage.json: {exc}", file=sys.stderr)
        return 2

    print(f"Coverage baseline gate: observed={coverage:.2f}% baseline={BASELINE:.2f}% target={TARGET:.2f}%")
    if coverage + 1e-9 < BASELINE:
        print(f"Coverage regressed below baseline ({coverage:.2f}% < {BASELINE:.2f}%)", file=sys.stderr)
        return 1
    if coverage < TARGET:
        print(f"Coverage remains below long-term target ({coverage:.2f}% < {TARGET:.2f}%); baseline debt stays visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
