"""OSV malware check for MCP extension packages.

Before launching an MCP server via npx/uvx, queries the OSV API to check
if the package has known malware advisories (MAL-* IDs).
Regular CVEs are ignored — only confirmed malware is blocked.

The API is free, public, and maintained by Google. Typical latency ~300ms.
Fail-open: network errors allow the package to proceed.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_OSV_ENDPOINT = "https://api.osv.dev/v1/query"
_TIMEOUT = 10  # seconds


def check_package_for_malware(command: str, args: list[str]) -> str | None:
    """Check if an MCP server package has known malware advisories.

    Args:
        command: The command (e.g., "npx", "uvx", "pip")
        args: Command arguments (package name is extracted from these)

    Returns:
        Warning message if malware found, None if clean or check failed.
    """
    pkg_name, ecosystem = _extract_package_info(command, args)
    if not pkg_name:
        return None

    try:
        advisories = _query_osv(pkg_name, ecosystem)
        malware = [a for a in advisories if _is_malware(a)]
        if malware:
            ids = ", ".join(a.get("id", "?") for a in malware[:3])
            return (
                f"⚠️ MALWARE DETECTED: Package '{pkg_name}' has known malware "
                f"advisories: {ids}. Blocking execution."
            )
    except Exception as e:
        logger.debug("OSV check failed for %s (fail-open): %s", pkg_name, e)

    return None


def _extract_package_info(command: str, args: list[str]) -> tuple[str | None, str]:
    """Extract package name and ecosystem from command + args."""
    cmd = command.lower().strip()

    if cmd in ("npx", "npm", "yarn", "pnpm", "bunx"):
        # npx @scope/package or npx package
        for arg in args:
            if not arg.startswith("-"):
                # Strip version specifier
                name = re.split(r"[@](?=\d)", arg)[0] if "@" in arg and not arg.startswith("@") else arg
                return name, "npm"
        return None, "npm"

    if cmd in ("uvx", "pip", "pip3", "pipx"):
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg.startswith("-"):
                continue
            # Skip pip subcommands
            if arg in ("install", "uninstall", "download", "show", "run"):
                continue
            name = re.split(r"[>=<~!]", arg)[0]
            return name, "PyPI"
        return None, "PyPI"

    return None, ""


def _query_osv(package: str, ecosystem: str) -> list[dict[str, Any]]:
    """Query the OSV API for vulnerabilities."""
    payload = json.dumps({
        "package": {"name": package, "ecosystem": ecosystem}
    }).encode()

    req = urllib.request.Request(
        _OSV_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        data = json.loads(resp.read())

    return data.get("vulns", [])


def _is_malware(advisory: dict[str, Any]) -> bool:
    """Check if an advisory is a malware advisory (MAL-* ID)."""
    adv_id = advisory.get("id", "")
    return adv_id.startswith("MAL-")
