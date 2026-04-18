"""URL safety tool — check URLs for suspicious patterns."""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

_SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".buzz", ".top", ".xyz",
    ".club", ".work", ".date", ".racing", ".win", ".bid", ".stream",
    ".download", ".loan", ".click",
}

_IP_PATTERN = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


@tool(
    name="url_check",
    description="Check if a URL is safe",
    params={
        "url": {"type": "string", "description": "URL to check"},
    },
    required=["url"],
)
async def url_check(url: str) -> dict:
    """Check a URL for suspicious patterns."""
    warnings: list[str] = []

    # Basic parse
    try:
        parsed = urlparse(url)
    except Exception:
        return {"ok": True, "safe": False, "warnings": ["Invalid URL"], "domain": ""}

    domain = parsed.hostname or ""
    scheme = parsed.scheme.lower()

    # data: URI
    if scheme == "data":
        warnings.append("data: URI can contain embedded content")
        return {"ok": True, "safe": False, "warnings": warnings, "domain": ""}

    # javascript: URI
    if scheme == "javascript":
        warnings.append("javascript: URI can execute code")
        return {"ok": True, "safe": False, "warnings": warnings, "domain": ""}

    # No HTTPS
    if scheme == "http":
        warnings.append("URL uses HTTP instead of HTTPS")

    # IP-based URL
    if _IP_PATTERN.match(domain):
        warnings.append("URL uses IP address instead of domain name")

    # Suspicious TLD
    for tld in _SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            warnings.append(f"Suspicious TLD: {tld}")
            break

    # URL length
    if len(url) > 2048:
        warnings.append(f"URL unusually long: {len(url)} chars")

    # Encoded characters in domain
    if "%" in domain:
        warnings.append("Domain contains encoded characters")

    # Multiple subdomains (potential phishing)
    if domain.count(".") > 3:
        warnings.append(f"Excessive subdomains: {domain}")

    # @ in URL (credential stuffing)
    if "@" in parsed.netloc:
        warnings.append("URL contains @ sign (possible credential phishing)")

    safe = len(warnings) == 0
    return {"ok": True, "safe": safe, "warnings": warnings, "domain": domain}
