"""Environment variable passthrough registry for sandboxed execution.

Skills that declare required_environment_variables need those vars available
in sandboxed environments. This module provides a session-scoped allowlist
so skill-declared vars pass through to child processes.

Uses ContextVar for session isolation in concurrent gateway processing.
"""
from __future__ import annotations

import logging
import os
from contextvars import ContextVar
from typing import Iterable

__all__ = [
    "register_passthrough",
    "is_passthrough",
    "get_passthrough_env",
    "clear_passthrough",
]


logger = logging.getLogger(__name__)

_allowed_env_vars: ContextVar[set[str]] = ContextVar("_allowed_env_vars")

# Always allowed (safe, non-secret)
_BUILTIN_PASSTHROUGH = frozenset({
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL",
    "TERM", "COLORTERM", "EDITOR", "VISUAL",
    "TZ", "PYTHONPATH", "NODE_PATH",
})


def _get_allowed() -> set[str]:
    try:
        return _allowed_env_vars.get()
    except LookupError:
        val: set[str] = set()
        _allowed_env_vars.set(val)
        return val


def register_passthrough(names: Iterable[str]) -> None:
    """Register environment variable names for passthrough."""
    allowed = _get_allowed()
    for name in names:
        name = name.strip()
        if name:
            allowed.add(name)
            logger.debug("Env passthrough registered: %s", name)


def is_passthrough(name: str) -> bool:
    """Check if an env var should pass through to sandboxed execution."""
    if name in _BUILTIN_PASSTHROUGH:
        return True
    return name in _get_allowed()


def get_passthrough_env() -> dict[str, str]:
    """Get all allowed env vars with their current values."""
    allowed = _get_allowed() | _BUILTIN_PASSTHROUGH
    return {k: v for k, v in os.environ.items() if k in allowed}


def clear_passthrough() -> None:
    """Clear session-scoped passthrough registrations."""
    try:
        _allowed_env_vars.set(set())
    except Exception as exc:
        logger.debug("clear_passthrough: suppressed %s", exc)
