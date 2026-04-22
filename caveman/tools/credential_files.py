"""Credential file registry for remote environments.

Remote backends (Docker, SSH) need host-side credential files mounted.
This module tracks which files to mount, using ContextVar for session isolation.
"""
from __future__ import annotations

import logging
import os
from contextvars import ContextVar

from caveman.paths import CAVEMAN_HOME

__all__ = [
    "register_credential_file",
    "get_credential_file_mounts",
    "get_skills_directory_mount",
    "get_cache_directory_mounts",
    "clear_registered",
]


logger = logging.getLogger(__name__)

_registered_files_var: ContextVar[dict[str, str]] = ContextVar("_registered_files")


def _get_registered() -> dict[str, str]:
    try:
        return _registered_files_var.get()
    except LookupError:
        d: dict[str, str] = {}
        _registered_files_var.set(d)
        return d


def register_credential_file(host_path: str, container_path: str | None = None) -> None:
    """Register a credential file for mounting in remote environments.

    Args:
        host_path: Path on the host machine.
        container_path: Path inside the container (defaults to same as host).
    """
    reg = _get_registered()
    target = container_path or host_path
    reg[host_path] = target
    logger.debug("Registered credential file: %s -> %s", host_path, target)


def get_credential_file_mounts() -> dict[str, str]:
    """Get all registered credential file mounts.

    Returns dict of {host_path: container_path}.
    """
    reg = _get_registered()

    # Also include default credential locations
    defaults = [
        str(CAVEMAN_HOME / "config.yaml"),
        os.path.expanduser("~/.ssh/config"),
        os.path.expanduser("~/.gitconfig"),
    ]
    for path in defaults:
        if os.path.exists(path) and path not in reg:
            reg[path] = path

    return dict(reg)


def get_skills_directory_mount() -> tuple[str, str] | None:
    """Get the skills directory mount point."""
    skills_dir = CAVEMAN_HOME / "skills"
    if skills_dir.exists():
        return str(skills_dir), str(skills_dir)
    return None


def get_cache_directory_mounts() -> dict[str, str]:
    """Get cache directories to mount (read-only in containers)."""
    mounts = {}
    cache_dirs = ["cache/media", "cache/tts", "cache/browser"]
    for subdir in cache_dirs:
        path = CAVEMAN_HOME / subdir
        if path.exists():
            mounts[str(path)] = str(path)
    return mounts


def clear_registered() -> None:
    """Clear all registered files (for testing)."""
    try:
        _registered_files_var.set({})
    except Exception as exc:
        logger.debug("clear_registered: suppressed %s", exc)
