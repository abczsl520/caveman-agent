"""Exception hierarchy — structured errors for the entire framework.

One hierarchy, clear categories, machine-parseable.
Every exception carries enough context for logging + debugging.

Base:
  CavemanError
  ├── ConfigError        — bad config, missing keys, wrong types
  ├── ProviderError      — LLM API failures
  │   ├── RateLimitError — 429 / quota exceeded
  │   └── AuthError      — invalid API key
  ├── ToolError          — tool execution failures
  │   ├── ToolNotFoundError
  │   ├── ToolPermissionError
  │   └── ToolTimeoutError
  ├── MemoryError        — memory store/recall failures
  ├── BridgeError        — IPC / external system failures
  └── SecurityError      — secret scan, permission denied
"""
from __future__ import annotations


class CavemanError(Exception):
    """Base exception for all Caveman errors."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}

    def to_dict(self) -> dict:
        return {
            "error_type": type(self).__name__,
            "message": str(self),
            "context": self.context,
        }


# ── Config ──

class ConfigError(CavemanError):
    """Configuration validation or loading error."""
    pass


# ── Provider / LLM ──

class ProviderError(CavemanError):
    """LLM provider API error."""
    pass


class RateLimitError(ProviderError):
    """Rate limited by provider (429)."""

    def __init__(self, message: str = "Rate limited", retry_after: float | None = None, **kwargs):
        super().__init__(message, kwargs)
        self.retry_after = retry_after


class AuthError(ProviderError):
    """Authentication failed (invalid API key)."""
    pass


# ── Tools ──

class ToolError(CavemanError):
    """Tool execution error."""
    pass


class ToolNotFoundError(ToolError):
    """Tool not registered."""
    pass


class ToolPermissionError(ToolError):
    """Tool call denied by permission manager."""
    pass


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""
    pass


# ── Memory ──

class MemoryError(CavemanError):
    """Memory store/recall error."""
    pass


# ── Bridge / IPC ──

class BridgeError(CavemanError):
    """Bridge communication error (UDS, MCP, etc.)."""
    pass


# ── Security ──

class SecurityError(CavemanError):
    """Security violation (secret detected, permission denied)."""
    pass
