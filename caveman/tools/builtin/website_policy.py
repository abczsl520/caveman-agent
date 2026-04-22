"""Website Policy — robots.txt, rate limiting, domain rules.

Manages per-domain access policies for web tools.
Extracted from Hermes tools/website_policy.py (282 lines).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

__all__ = ["DomainPolicy", "WebsitePolicyManager"]


logger = logging.getLogger("caveman.tools.website_policy")


@dataclass
class DomainPolicy:
    """Access policy for a domain."""
    domain: str
    allowed: bool = True
    rate_limit_rpm: int = 60  # requests per minute
    crawl_delay: float = 1.0  # seconds between requests
    max_pages: int = 50
    respect_robots: bool = True
    custom_user_agent: str = ""
    notes: str = ""


# ── Default Policies ──

_BLOCKED_DOMAINS: Set[str] = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",
}

_RATE_LIMITED_DOMAINS: Dict[str, int] = {
    "api.github.com": 30,
    "api.openai.com": 10,
    "api.anthropic.com": 10,
    "twitter.com": 20,
    "x.com": 20,
}


class WebsitePolicyManager:
    """Manages website access policies."""

    def __init__(self):
        self._policies: Dict[str, DomainPolicy] = {}
        self._request_log: Dict[str, List[float]] = {}

    def get_policy(self, url: str) -> DomainPolicy:
        """Get policy for a URL's domain."""
        domain = self._extract_domain(url)

        # Check custom policies
        if domain in self._policies:
            return self._policies[domain]

        # Check blocked
        if domain in _BLOCKED_DOMAINS or self._is_private_ip(domain):
            return DomainPolicy(domain=domain, allowed=False, notes="blocked:private")

        # Check rate-limited
        rpm = _RATE_LIMITED_DOMAINS.get(domain, 60)
        return DomainPolicy(domain=domain, rate_limit_rpm=rpm)

    def set_policy(self, domain: str, **kwargs) -> DomainPolicy:
        """Set a custom policy for a domain."""
        policy = self._policies.get(domain, DomainPolicy(domain=domain))
        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        self._policies[domain] = policy
        return policy

    def check_access(self, url: str) -> Dict[str, Any]:
        """Check if a URL can be accessed right now."""
        policy = self.get_policy(url)
        domain = self._extract_domain(url)

        if not policy.allowed:
            return {"allowed": False, "reason": policy.notes or "domain_blocked"}

        # Rate limit check
        now = time.time()
        window = 60.0
        log = self._request_log.get(domain, [])
        # Clean old entries
        log = [t for t in log if now - t < window]
        self._request_log[domain] = log

        if len(log) >= policy.rate_limit_rpm:
            wait = log[0] + window - now
            return {
                "allowed": False,
                "reason": "rate_limited",
                "retry_after": round(wait, 1),
            }

        # Crawl delay check
        if log and (now - log[-1]) < policy.crawl_delay:
            wait = policy.crawl_delay - (now - log[-1])
            return {
                "allowed": False,
                "reason": "crawl_delay",
                "retry_after": round(wait, 2),
            }

        return {"allowed": True}

    def record_request(self, url: str) -> None:
        """Record a request for rate limiting."""
        domain = self._extract_domain(url)
        if domain not in self._request_log:
            self._request_log[domain] = []
        self._request_log[domain].append(time.time())

    def _extract_domain(self, url: str) -> str:
        try:
            parsed = urlparse(url)
            return parsed.netloc.split(":")[0].lower()
        except Exception:
            return url.lower()

    def _is_private_ip(self, domain: str) -> bool:
        """Check if domain resolves to a private IP."""
        # Simple pattern check (not DNS resolution)
        private_patterns = [
            r"^10\.",
            r"^172\.(1[6-9]|2[0-9]|3[01])\.",
            r"^192\.168\.",
            r"^127\.",
            r"^0\.",
            r"^fc[0-9a-f]{2}:",  # IPv6 ULA
            r"^fe80:",  # IPv6 link-local
        ]
        return any(re.match(p, domain) for p in private_patterns)

    def list_policies(self) -> List[DomainPolicy]:
        return list(self._policies.values())

    def reset(self) -> None:
        self._policies.clear()
        self._request_log.clear()
