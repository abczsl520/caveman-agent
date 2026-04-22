"""Path safety — prevent path traversal and symlink attacks.

Validates file paths before operations to prevent:
- Directory traversal (../../etc/passwd)
- Symlink following to sensitive locations
- Access to system directories
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["is_safe_path", "resolve_safe_path", "BLOCKED_PATHS"]

# Paths that should never be accessed by the agent
BLOCKED_PATHS: frozenset[str] = frozenset({
    "/etc/shadow", "/etc/passwd", "/etc/sudoers",
    "/root/.ssh", "/root/.bash_history",
    "/var/log/auth.log", "/var/log/secure",
    "/proc/self/environ",
})

# Directories that should never be written to
BLOCKED_WRITE_DIRS: frozenset[str] = frozenset({
    "/etc", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc",
    "/var/run", "/var/lock",
})

# Home directory sensitive files
_HOME_SENSITIVE = frozenset({
    ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/config",
    ".aws/credentials", ".azure/credentials",
    ".gcloud/credentials.db", ".config/gcloud/credentials.db",
    ".npmrc", ".pypirc", ".netrc",
    ".env", ".env.local", ".env.production",
})


def is_safe_path(path: str, allow_write: bool = False) -> tuple[bool, str]:
    """Check if a path is safe to access.

    Returns (is_safe, reason).
    """
    try:
        resolved = str(Path(path).expanduser().resolve())
    except (ValueError, OSError) as e:
        return False, f"Invalid path: {e}"

    # Check blocked paths (compare both resolved and original)
    for blocked in BLOCKED_PATHS:
        blocked_resolved = str(Path(blocked).resolve())
        if (resolved == blocked_resolved or resolved.startswith(blocked_resolved + "/")
                or resolved == blocked or resolved.startswith(blocked + "/")):
            return False, f"Access to {blocked} is blocked"

    # Check write restrictions
    if allow_write:
        for blocked_dir in BLOCKED_WRITE_DIRS:
            blocked_resolved = str(Path(blocked_dir).resolve())
            if (resolved.startswith(blocked_resolved + "/") or resolved == blocked_resolved
                    or resolved.startswith(blocked_dir + "/") or resolved == blocked_dir):
                return False, f"Writing to {blocked_dir} is blocked"

    # Check home sensitive files
    home = str(Path.home())
    if resolved.startswith(home):
        rel = resolved[len(home):].lstrip("/")
        for sensitive in _HOME_SENSITIVE:
            if rel == sensitive or rel.startswith(sensitive + "/"):
                return False, f"Access to ~/{sensitive} is blocked"

    # Check for symlink attacks (resolve and compare)
    try:
        real = os.path.realpath(path)
        if real != resolved and not real.startswith(resolved):
            # Symlink points somewhere unexpected
            for blocked in BLOCKED_PATHS:
                if real.startswith(blocked):
                    return False, f"Symlink resolves to blocked path: {blocked}"
    except OSError:
        pass  # intentional: OSError suppressed

    return True, "ok"


def resolve_safe_path(path: str, base_dir: str | None = None, allow_write: bool = False) -> str:
    """Resolve a path safely, raising ValueError if unsafe."""
    safe, reason = is_safe_path(path, allow_write=allow_write)
    if not safe:
        raise ValueError(f"Unsafe path '{path}': {reason}")

    resolved = str(Path(path).expanduser().resolve())

    # If base_dir specified, ensure path is within it
    if base_dir:
        base = str(Path(base_dir).resolve())
        if not resolved.startswith(base + "/") and resolved != base:
            raise ValueError(f"Path '{path}' is outside base directory '{base_dir}'")

    return resolved
