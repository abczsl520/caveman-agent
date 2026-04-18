"""Caveman Stats — project statistics at a glance."""
from __future__ import annotations
import subprocess
from pathlib import Path

CAVEMAN_DIR = Path(__file__).resolve().parent.parent


def get_stats() -> str:
    """Generate project statistics."""
    # File count
    py_files = list(CAVEMAN_DIR.rglob("*.py"))
    py_files = [f for f in py_files if "__pycache__" not in str(f)]
    file_count = len(py_files)

    # LOC
    loc = sum(
        len(f.read_text(encoding="utf-8").splitlines())
        for f in py_files
    )

    # Test count (from pytest --co)
    test_count = "?"
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", "tests/", "--co", "-q"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=CAVEMAN_DIR.parent, timeout=30,
        )
        for line in r.stdout.splitlines():
            if "tests collected" in line or "test collected" in line:
                test_count = line.split()[0]
                break
    except Exception:
        pass  # intentional: non-critical

    # Commit count
    commit_count = "?"
    try:
        r = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, encoding="utf-8",
            cwd=CAVEMAN_DIR.parent, timeout=10,
        )
        if r.returncode == 0:
            commit_count = r.stdout.strip()
    except Exception:
        pass  # intentional: non-critical

    # Version
    try:
        from caveman import __version__
        version = __version__
    except ImportError:
        version = "?"

    # Module count
    modules = set()
    for f in py_files:
        rel = f.relative_to(CAVEMAN_DIR)
        if len(rel.parts) > 1:
            modules.add(rel.parts[0])
    module_count = len(modules)

    return (
        f"🦴 Caveman v{version}\n"
        f"\n"
        f"  Files:    {file_count}\n"
        f"  LOC:      {loc:,}\n"
        f"  Tests:    {test_count}\n"
        f"  Commits:  {commit_count}\n"
        f"  Modules:  {module_count}\n"
    )
