"""Terminal Tool v2 — compatibility shim.

All functionality has been merged into terminal_tool.py.
This module re-exports symbols for backward compatibility.
"""
from __future__ import annotations

from caveman.tools.builtin.terminal_tool import (  # noqa: F401
    _BLOCKED_PATTERNS as BLOCKED_PATTERNS,
    _check_guards,
    _truncate_output,
    _validate_workdir,
    execute_command,
)

# Re-export constants under v2 names
MAX_OUTPUT_CHARS = 100_000
MAX_OUTPUT_LINES = 500
TRUNCATION_NOTICE = "\n... (output truncated, {total} chars / {lines} lines total)"

__all__ = [
    "BLOCKED_PATTERNS",
    "MAX_OUTPUT_CHARS",
    "MAX_OUTPUT_LINES",
    "TRUNCATION_NOTICE",
    "_check_guards",
    "_truncate_output",
    "_validate_workdir",
    "terminal_execute",
]


async def terminal_execute(
    command: str,
    workdir: str = "",
    timeout: int = 120,
    env=None,
):
    """Compatibility wrapper — delegates to execute_command."""
    result = await execute_command(
        command, timeout=timeout, cwd=workdir or None,
        extra_env=env,
    )
    # Add "ok" field for v2 API compatibility
    if "ok" not in result:
        result["ok"] = result.get("exit_code", -1) == 0
    # Map stderr to "error" for v2 API compatibility when command failed
    if not result["ok"] and "error" not in result and result.get("stderr"):
        result["error"] = result["stderr"]
    return result
