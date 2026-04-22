"""Website access policy for URL-capable tools.

Loads a user-managed website blocklist from config and optional shared list files.
Browser/web tools call check_url() before accessing any URL.

Policy is cached in memory with a short TTL so config changes take effect quickly.
"""
from __future__ import annotations

import fnmatch
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)

_CACHE_TTL = 30  # seconds
_cache: dict[str, Any] = {"policy": None, "loaded_at": 0.0}

_DEFAULT_POLICY = {
    "enabled": False,
    "blocked_domains": [],
    "allowed_domains": [],  # If non-empty, only these are allowed (allowlist mode)
    "shared_files": [],
}


def check_url(url: str, config: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Check if a URL is allowed by the website policy.

    Args:
        url: The URL to check.
        config: Optional policy config override (for testing).

    Returns:
        (allowed, reason) — True if allowed, False with reason if blocked.
    """
    policy = config or _get_policy()

    if not policy.get("enabled", False):
        return True, ""

    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
    except Exception:
        return False, "invalid URL"

    if not hostname:
        return False, "no hostname"

    # Allowlist mode: if allowed_domains is set, only those are permitted
    allowed = policy.get("allowed_domains", [])
    if allowed:
        if not _matches_any(hostname, allowed):
            return False, f"not in allowlist"
        return True, ""

    # Blocklist mode
    blocked = policy.get("blocked_domains", [])
    # Load shared files
    for shared_file in policy.get("shared_files", []):
        blocked.extend(_load_shared_file(shared_file))

    if _matches_any(hostname, blocked):
        return False, f"blocked by policy"

    return True, ""


def _matches_any(hostname: str, patterns: list[str]) -> bool:
    """Check if hostname matches any pattern (supports wildcards)."""
    for pattern in patterns:
        pattern = pattern.lower().strip()
        if not pattern:
            continue
        # Exact match
        if hostname == pattern:
            return True
        # Wildcard match (e.g., *.example.com)
        if fnmatch.fnmatch(hostname, pattern):
            return True
        # Subdomain match: blocking example.com also blocks sub.example.com
        if hostname.endswith("." + pattern):
            return True
    return False


def _get_policy() -> dict[str, Any]:
    """Get cached policy, reloading from config if TTL expired."""
    now = time.time()
    if _cache["policy"] is not None and (now - _cache["loaded_at"]) < _CACHE_TTL:
        return _cache["policy"]

    policy = _load_policy_from_config()
    _cache["policy"] = policy
    _cache["loaded_at"] = now
    return policy


def _load_policy_from_config() -> dict[str, Any]:
    """Load website policy from Caveman config."""
    try:
        from caveman.config.loader import load_config
        config = load_config()
        return config.get("website_policy", dict(_DEFAULT_POLICY))
    except Exception:
        return dict(_DEFAULT_POLICY)


def _load_shared_file(path_str: str) -> list[str]:
    """Load domain list from a shared file (one domain per line)."""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = CAVEMAN_HOME / path_str

    if not path.exists():
        return []

    try:
        domains = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                domains.append(line)
        return domains
    except Exception as e:
        logger.debug("Failed to load shared blocklist %s: %s", path, e)
        return []


def clear_cache() -> None:
    """Clear the policy cache (for testing or config reload)."""
    _cache["policy"] = None
    _cache["loaded_at"] = 0.0
