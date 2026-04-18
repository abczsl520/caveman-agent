"""Regex-based secret redaction for logs and tool output.

Ported from Hermes redact.py (MIT, Nous Research).
Masks API keys, tokens, credentials, phone numbers, private keys,
and database connection strings before they reach logs or output.
"""
from __future__ import annotations

import logging
import re

# Known API key prefixes — match prefix + contiguous token chars
_PREFIX_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",           # OpenAI / Anthropic
    r"ghp_[A-Za-z0-9]{10,}",            # GitHub PAT (classic)
    r"github_pat_[A-Za-z0-9_]{10,}",    # GitHub PAT (fine-grained)
    r"gho_[A-Za-z0-9]{10,}",            # GitHub OAuth
    r"xox[baprs]-[A-Za-z0-9-]{10,}",    # Slack tokens
    r"AIza[A-Za-z0-9_-]{30,}",          # Google API keys
    r"AKIA[A-Z0-9]{16}",                # AWS Access Key ID
    r"sk_live_[A-Za-z0-9]{10,}",        # Stripe live
    r"sk_test_[A-Za-z0-9]{10,}",        # Stripe test
    r"SG\.[A-Za-z0-9_-]{10,}",          # SendGrid
    r"hf_[A-Za-z0-9]{10,}",             # HuggingFace
    r"tvly-[A-Za-z0-9]{10,}",           # Tavily
    r"gsk_[A-Za-z0-9]{10,}",            # Groq
    r"pplx-[A-Za-z0-9]{10,}",           # Perplexity
    r"npm_[A-Za-z0-9]{10,}",            # npm
    r"pypi-[A-Za-z0-9_-]{10,}",         # PyPI
    r"cr_[A-Za-z0-9_-]{10,}",           # Custom reverse proxy keys
]

# ENV assignment: KEY=value where KEY looks secret-like
_SECRET_ENV_NAMES = r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})\s*=\s*(['\"]?)(\S+)\2",
)

# JSON field: "apiKey": "value"
_JSON_KEY_NAMES = (
    r"(?:api_?[Kk]ey|token|secret|password|access_token|"
    r"refresh_token|auth_token|bearer|secret_value)"
)
_JSON_FIELD_RE = re.compile(
    rf'("{_JSON_KEY_NAMES}")\s*:\s*"([^"]+)"', re.IGNORECASE,
)

# Authorization headers
_AUTH_HEADER_RE = re.compile(
    r"(Authorization:\s*Bearer\s+)(\S+)", re.IGNORECASE,
)

# Telegram bot tokens
_TELEGRAM_RE = re.compile(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})")

# Private key blocks
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"
)

# Database connection strings
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:)([^@]+)(@)",
    re.IGNORECASE,
)

# E.164 phone numbers
_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")

# Compiled prefix pattern
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)


def _mask(token: str) -> str:
    """Mask a token, preserving prefix for long tokens."""
    if len(token) < 18:
        return "***"
    return f"{token[:6]}...{token[-4:]}"


def redact_text(text: str) -> str:
    """Apply all redaction patterns to text.

    Safe to call on any string — non-matching text passes through unchanged.
    """
    if not text or not isinstance(text, str):
        return text or ""

    # Known prefixes
    text = _PREFIX_RE.sub(lambda m: _mask(m.group(1)), text)

    # ENV assignments
    text = _ENV_ASSIGN_RE.sub(
        lambda m: f"{m.group(1)}={m.group(2)}{_mask(m.group(3))}{m.group(2)}",
        text,
    )

    # JSON fields
    text = _JSON_FIELD_RE.sub(
        lambda m: f'{m.group(1)}: "{_mask(m.group(2))}"', text,
    )

    # Auth headers
    text = _AUTH_HEADER_RE.sub(
        lambda m: m.group(1) + _mask(m.group(2)), text,
    )

    # Telegram bot tokens
    text = _TELEGRAM_RE.sub(
        lambda m: f"{m.group(1) or ''}{m.group(2)}:***", text,
    )

    # Private keys
    text = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", text)

    # DB connection strings
    text = _DB_CONNSTR_RE.sub(
        lambda m: f"{m.group(1)}***{m.group(3)}", text,
    )

    # Phone numbers
    def _redact_phone(m):
        p = m.group(1)
        return p[:4] + "****" + p[-4:] if len(p) > 8 else p[:2] + "****" + p[-2:]
    text = _PHONE_RE.sub(_redact_phone, text)

    return text


class RedactingFormatter(logging.Formatter):
    """Log formatter that redacts secrets from all messages."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))
