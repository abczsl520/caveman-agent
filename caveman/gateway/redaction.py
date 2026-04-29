"""Redaction Engine — PII and secret detection/masking.

Detects and redacts sensitive information from text before it's
sent to external services or displayed. Extracted from Hermes
agent/redact.py + OpenClaw redaction patterns.
"""
from __future__ import annotations

import re
from typing import List, Tuple, TypedDict

__all__ = [
    "PREFIX_RE",
    "redact_secrets",
    "redact_pii",
    "redact_all",
    "detect_secrets",
    "detect_pii",
    "is_safe_for_external",
]


# ── Secret Patterns ──

_SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("api_key_openai", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
    ("api_key_anthropic", re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}")),
    ("github_token", re.compile(r"ghp_[a-zA-Z0-9]{36}")),
    ("github_token_fine", re.compile(r"github_pat_[a-zA-Z0-9_]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[A-Z0-9]{16}")),
    ("aws_secret_key", re.compile(r"(?:aws_secret|secret_key)['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})")),
    ("google_api_key", re.compile(r"AIza[a-zA-Z0-9_-]{35}")),
    ("slack_token", re.compile(r"xox[bpras]-[a-zA-Z0-9-]+")),
    ("discord_token", re.compile(r"[MN][A-Za-z0-9]{23,}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,}")),
    ("stripe_key", re.compile(r"[sr]k_(live|test)_[a-zA-Z0-9]{20,}")),
    ("bearer_token", re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}")),
    ("basic_auth", re.compile(r"Basic\s+[A-Za-z0-9+/=]{20,}")),
    ("private_key", re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----")),
    ("jwt", re.compile(r"eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}")),
    ("password_assign", re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"]?[^\s'\"]{8,}", re.IGNORECASE)),
    ("connection_string", re.compile(r"(?:mongodb|postgres|mysql|redis)://[^\s]+")),
]

# ── PII Patterns ──

_PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("phone_us", re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")),
    ("phone_intl", re.compile(r"\+\d{1,3}[-.\s]?\d{4,14}")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")),
    ("ip_address", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")),
]


class DetectionFinding(TypedDict):
    """Location and preview for a redaction detector match."""

    type: str
    start: int
    end: int
    preview: str

# Prefix pattern for URL exfiltration detection
PREFIX_RE = re.compile(
    r"(?:sk-[a-zA-Z0-9]|ghp_|github_pat_|AKIA|xox[bpras]-|Bearer\s)",
    re.IGNORECASE,
)


def redact_secrets(text: str, replacement: str = "[REDACTED]") -> str:
    """Redact known secret patterns from text."""
    for name, pattern in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_pii(text: str, replacement: str = "[PII]") -> str:
    """Redact PII patterns from text."""
    for name, pattern in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_all(text: str) -> str:
    """Redact both secrets and PII."""
    text = redact_secrets(text)
    text = redact_pii(text)
    return text


def detect_secrets(text: str) -> list[DetectionFinding]:
    """Detect secrets in text without redacting. Returns list of findings."""
    findings: list[DetectionFinding] = []
    for name, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append({
                "type": name,
                "start": match.start(),
                "end": match.end(),
                "preview": match.group()[:8] + "...",
            })
    return findings


def detect_pii(text: str) -> list[DetectionFinding]:
    """Detect PII in text without redacting."""
    findings: list[DetectionFinding] = []
    for name, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            findings.append({
                "type": name,
                "start": match.start(),
                "end": match.end(),
                "preview": match.group()[:8] + "...",
            })
    return findings


def is_safe_for_external(text: str) -> bool:
    """Check if text is safe to send to external services."""
    return not detect_secrets(text) and not detect_pii(text)
