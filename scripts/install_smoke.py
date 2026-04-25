#!/usr/bin/env python3
"""Install smoke benchmark for Caveman v1.0 release gate (#30).

Runs in a clean virtual environment and verifies:
- package can be installed from the current checkout
- `caveman version` works
- `caveman --help` works
- `python -m caveman version` works

The script records elapsed seconds and exits non-zero if installation exceeds the
configured threshold. CI runs this on ubuntu/macos/windows to collect objective
evidence instead of relying on manual claims.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Caveman install smoke benchmark")
    parser.add_argument("--source", default=".", help="Package source path to install")
    parser.add_argument("--max-seconds", type=float, default=180.0, help="Install+smoke threshold")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    report: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "source": str(source),
        "max_seconds": args.max_seconds,
        "steps": [],
    }

    if not source.exists():
        report["status"] = "failed"
        report["reason"] = f"source not found: {source}"
        print(json.dumps(report, indent=2))
        return 2

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="caveman-install-smoke-") as td:
        venv_dir = Path(td) / "venv"
        create = run([sys.executable, "-m", "venv", str(venv_dir)])
        report["steps"].append({"cmd": "venv", "returncode": create.returncode, "output": create.stdout[-4000:]})
        if create.returncode != 0:
            report["status"] = "failed"
            report["reason"] = "venv creation failed"
            print(json.dumps(report, indent=2))
            return create.returncode or 1

        if os.name == "nt":
            python = venv_dir / "Scripts" / "python.exe"
            caveman = venv_dir / "Scripts" / "caveman.exe"
        else:
            python = venv_dir / "bin" / "python"
            caveman = venv_dir / "bin" / "caveman"

        pip_install = run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        report["steps"].append({"cmd": "pip upgrade", "returncode": pip_install.returncode, "output": pip_install.stdout[-4000:]})
        if pip_install.returncode != 0:
            report["status"] = "failed"
            report["reason"] = "pip upgrade failed"
            print(json.dumps(report, indent=2))
            return pip_install.returncode or 1

        install = run([str(python), "-m", "pip", "install", str(source)])
        report["steps"].append({"cmd": "pip install source", "returncode": install.returncode, "output": install.stdout[-4000:]})
        if install.returncode != 0:
            report["status"] = "failed"
            report["reason"] = "package install failed"
            print(json.dumps(report, indent=2))
            return install.returncode or 1

        for cmd in ([str(caveman), "version"], [str(caveman), "--help"], [str(python), "-m", "caveman", "version"]):
            proc = run(cmd)
            report["steps"].append({"cmd": " ".join(cmd), "returncode": proc.returncode, "output": proc.stdout[-4000:]})
            if proc.returncode != 0:
                report["status"] = "failed"
                report["reason"] = f"smoke command failed: {' '.join(cmd)}"
                print(json.dumps(report, indent=2))
                return proc.returncode or 1

    elapsed = time.perf_counter() - start
    report["elapsed_seconds"] = round(elapsed, 3)
    report["status"] = "passed" if elapsed <= args.max_seconds else "failed"
    if elapsed > args.max_seconds:
        report["reason"] = f"elapsed {elapsed:.1f}s exceeds {args.max_seconds:.1f}s"

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
