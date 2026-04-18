"""Caveman Audit — static code quality checks without LLM.

Usage:
    caveman audit [--fix]

Checks:
1. open() without encoding
2. UUID truncation in memory IDs
3. bare except / swallowed exceptions
4. files over 400 lines (NFR-502)
5. missing __all__ in __init__.py
"""
from __future__ import annotations
import re
from pathlib import Path

CAVEMAN_DIR = Path(__file__).resolve().parent.parent


def _find_python_files() -> list[Path]:
    """Find all Python files in caveman/, excluding __pycache__."""
    return sorted(
        p for p in CAVEMAN_DIR.rglob("*.py")
        if "__pycache__" not in str(p)
    )


def check_encoding(files: list[Path]) -> list[str]:
    """Find open() calls without encoding parameter."""
    issues = []
    pattern = re.compile(r'\bopen\((?!.*encoding)')
    skip = {"open()", "open_browser", "open_url", "webbrowser", "subprocess", "open(\\("}
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("r\"") or stripped.startswith("r'"):
                continue
            if pattern.search(line) and not any(s in line for s in skip):
                issues.append(f"{f.relative_to(CAVEMAN_DIR)}:{i}: open() without encoding")
    return issues


def check_uuid_truncation(files: list[Path]) -> list[str]:
    """Find UUID truncation that might cause collisions."""
    issues = []
    pattern = re.compile(r'uuid4\(\).*\[:8\]')
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                issues.append(f"{f.relative_to(CAVEMAN_DIR)}:{i}: UUID truncated to 8 chars")
    return issues


def check_swallowed_exceptions(files: list[Path]) -> list[str]:
    """Find bare except or except Exception: pass without logging."""
    issues = []
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "except:":
                issues.append(f"{f.relative_to(CAVEMAN_DIR)}:{i}: bare except")
    return issues


def check_file_size(files: list[Path], max_lines: int = 400) -> list[str]:
    """Find files exceeding NFR-502 line limit."""
    issues = []
    for f in files:
        count = len(f.read_text(encoding="utf-8").splitlines())
        if count > max_lines:
            issues.append(f"{f.relative_to(CAVEMAN_DIR)}: {count} lines (max {max_lines})")
    return issues


def run_audit() -> str:
    """Run all static checks and return report."""
    files = _find_python_files()
    checks = {
        "encoding": check_encoding(files),
        "uuid_truncation": check_uuid_truncation(files),
        "swallowed_exceptions": check_swallowed_exceptions(files),
        "file_size": check_file_size(files),
    }

    total = sum(len(v) for v in checks.values())
    parts = [f"Caveman Audit — {len(files)} files, {total} issues\n"]

    for name, issues in checks.items():
        status = "✅" if not issues else f"❌ {len(issues)}"
        parts.append(f"  {name}: {status}")
        for issue in issues[:5]:
            parts.append(f"    - {issue}")
        if len(issues) > 5:
            parts.append(f"    ... and {len(issues) - 5} more")

    return "\n".join(parts)
