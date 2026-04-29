"""Code health checker — single source of truth for NFR-502.

Reads policy from pyproject.toml [tool.caveman.code-health].
All tests should call `check_code_health()` instead of hardcoding thresholds.

Two checks:
  1. File size (lines) — tiered by file role
  2. Function size — no single function should be a god-function
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent  # caveman/
_PROJECT_ROOT = _ROOT.parent  # repo root

# Defaults if pyproject.toml is missing or malformed
_DEFAULTS = {
    "default_max_lines": 450,
    "default_max_function_lines": 100,
    "engine_max_lines": 550,
    "cli_max_lines": 600,
    "overrides": {},
}


def _load_policy() -> dict[str, Any]:
    """Load code-health policy from pyproject.toml."""
    toml_path = _PROJECT_ROOT / "pyproject.toml"
    if not toml_path.exists():
        return dict(_DEFAULTS)
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return dict(_DEFAULTS)
    try:
        data = tomllib.loads(toml_path.read_text())
        policy = data.get("tool", {}).get("caveman", {}).get("code-health", {})
        result = dict(_DEFAULTS)
        for key in ("default_max_lines", "default_max_function_lines",
                     "engine_max_lines", "cli_max_lines"):
            if key in policy:
                result[key] = int(policy[key])
        if "overrides" in policy:
            result["overrides"] = {str(k): int(v) for k, v in policy["overrides"].items()}
        return result
    except Exception as e:
        logger.debug("Failed to load code-health policy: %s", e)
        return dict(_DEFAULTS)


def _get_threshold(rel_path: str, policy: dict[str, Any]) -> int:
    """Determine the line threshold for a file based on its role."""
    # Per-file override takes priority
    overrides = policy.get("overrides", {})
    for pattern, limit in overrides.items():
        if rel_path.endswith(pattern) or pattern in rel_path:
            return cast(int, limit)
    # Tier: engine files
    if "/engines/" in rel_path or rel_path.startswith("engines/"):
        return cast(int, policy["engine_max_lines"])
    # Tier: CLI files
    if "/cli/" in rel_path or rel_path.startswith("cli/"):
        return cast(int, policy["cli_max_lines"])
    return cast(int, policy["default_max_lines"])


def _check_function_sizes(source: str, path: Path, max_fn_lines: int) -> list[str]:
    """Find functions exceeding the max function size."""
    issues = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            size = end - node.lineno
            if size > max_fn_lines:
                issues.append(
                    f"{path}: {node.name}() is {size} lines (max {max_fn_lines})"
                )
    return issues


def check_code_health(
    root: Path | None = None,
    check_functions: bool = True,
) -> dict[str, list[str]]:
    """Run all code health checks. Returns dict of category → issues.

    Usage in tests:
        from caveman.cli.code_health import check_code_health
        result = check_code_health()
        assert not result["file_size"], result["file_size"]
        assert not result["function_size"], result["function_size"]
    """
    root = root or _ROOT
    policy = _load_policy()
    file_issues: list[str] = []
    fn_issues: list[str] = []

    for py in sorted(root.rglob("*.py")):
        if "__pycache__" in str(py):
            continue
        rel = str(py.relative_to(root))
        try:
            source = py.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = len(source.splitlines())
        threshold = _get_threshold(rel, policy)
        if lines > threshold:
            file_issues.append(f"{rel}: {lines} lines (max {threshold})")
        if check_functions:
            fn_issues.extend(
                _check_function_sizes(source, Path(rel), policy["default_max_function_lines"])
            )

    return {"file_size": file_issues, "function_size": fn_issues}


def format_report(result: dict[str, list[str]]) -> str:
    """Format check results as human-readable report."""
    total = sum(len(v) for v in result.values())
    parts = [f"Code Health — {total} issues\n"]
    for category, issues in result.items():
        status = "✅" if not issues else f"❌ {len(issues)}"
        parts.append(f"  {category}: {status}")
        for issue in issues[:10]:
            parts.append(f"    - {issue}")
        if len(issues) > 10:
            parts.append(f"    ... and {len(issues) - 10} more")
    return "\n".join(parts)
