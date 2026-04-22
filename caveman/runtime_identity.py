"""Runtime identity and environment hygiene for Caveman.

Solves the "environment pollution" problem: when Caveman is launched from
another agent's shell (e.g., OpenClaw's nohup), it inherits environment
variables that can mislead both the agent and the developer.

This module:
1. Declares Caveman's own identity variables
2. Sanitizes inherited env vars from other agent frameworks
3. Provides a clean env builder for subprocess spawning
4. Exposes runtime identity for self-awareness in prompts

Called once at startup (gateway/CLI entry points).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from caveman.paths import CAVEMAN_HOME

__all__ = [
    "CAVEMAN_SERVICE_NAME",
    "CAVEMAN_VERSION",
    "sanitize_environment",
    "build_clean_env",
    "get_runtime_identity",
]


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caveman identity constants
# ---------------------------------------------------------------------------

CAVEMAN_SERVICE_NAME = "caveman"
CAVEMAN_VERSION = "0.4.0"

# Environment variable prefixes from other agent frameworks
# These get stripped from Caveman's process environment at startup
_FOREIGN_ENV_PREFIXES = (
    "OPENCLAW_",
    "HERMES_",
    "CLAUDE_CODE_",
    "CODEX_",
)

# Caveman's own env vars (never stripped)
_CAVEMAN_ENV_PREFIX = "CAVEMAN_"

# Variables to preserve even if they match foreign prefixes
# (e.g., if user explicitly set them for interop)
_PRESERVE_VARS = frozenset({
    # None currently — add here if needed
})


# ---------------------------------------------------------------------------
# Environment sanitization
# ---------------------------------------------------------------------------

def sanitize_environment() -> dict[str, str]:
    """Remove foreign agent env vars from the current process.

    Returns dict of removed variables for logging.
    Called once at Caveman startup.
    """
    removed: dict[str, str] = {}

    for key in list(os.environ.keys()):
        if key in _PRESERVE_VARS:
            continue
        if any(key.startswith(prefix) for prefix in _FOREIGN_ENV_PREFIXES):
            removed[key] = os.environ.pop(key)

    # Set Caveman's own identity
    os.environ["CAVEMAN_SERVICE_NAME"] = CAVEMAN_SERVICE_NAME
    os.environ["CAVEMAN_VERSION"] = CAVEMAN_VERSION
    os.environ["CAVEMAN_PID"] = str(os.getpid())

    if removed:
        logger.info(
            "Sanitized %d foreign env vars: %s",
            len(removed),
            ", ".join(sorted(removed.keys())),
        )

    return removed


def build_clean_env(
    extra: dict[str, str] | None = None,
    inherit_venv: bool = True,
) -> dict[str, str]:
    """Build a clean environment dict for subprocess spawning.

    - Starts from current (already sanitized) os.environ
    - Adds TERM=dumb, NO_COLOR=1 for clean output
    - Optionally adds venv PATH
    - Merges any extra vars
    """
    env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1"}

    if inherit_venv:
        # Find venv relative to caveman package
        pkg_dir = Path(__file__).resolve().parent.parent
        venv_bin = pkg_dir / ".venv" / "bin"
        if venv_bin.is_dir():
            env["PATH"] = str(venv_bin) + ":" + env.get("PATH", "")
            env["VIRTUAL_ENV"] = str(venv_bin.parent)

    if extra:
        env.update(extra)

    return env


# ---------------------------------------------------------------------------
# Runtime identity (for prompt injection / self-awareness)
# ---------------------------------------------------------------------------

def get_runtime_identity() -> dict[str, str]:
    """Get Caveman's runtime identity for prompt injection.

    Used by the prompt builder to give the agent accurate self-awareness
    about its execution environment.
    """
    return {
        "service": CAVEMAN_SERVICE_NAME,
        "version": CAVEMAN_VERSION,
        "pid": str(os.getpid()),
        "home": str(CAVEMAN_HOME),
        "surface": os.environ.get("CAVEMAN_SURFACE", "unknown"),
    }
