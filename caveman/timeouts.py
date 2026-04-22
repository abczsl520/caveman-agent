"""Centralized timeout constants.

All hardcoded timeouts should reference these constants instead of
magic numbers. Grouped by semantic category for easy tuning.

To override at runtime, set the corresponding environment variable:
    CAVEMAN_TIMEOUT_HTTP=60  (overrides HTTP_DEFAULT)
"""
from __future__ import annotations

import os

__all__ = [
    "HTTP_DEFAULT",
    "HTTP_FAST",
    "HTTP_SLOW",
    "HTTP_LLM",
    "HTTP_IMAGE_GEN",
    "HTTP_TRANSCRIBE",
    "HTTP_TTS",
    "HTTP_EMBEDDING",
    "SUBPROCESS_FAST",
    "SUBPROCESS_DEFAULT",
    "SUBPROCESS_SLOW",
    "SUBPROCESS_FLYWHEEL",
    "WS_PING",
    "WS_CONNECT",
    "WS_MESSAGE",
    "DRAIN_DEFAULT",
    "DRAIN_LONG",
    "TASK_DEFAULT",
    "TASK_SHORT",
    "GATEWAY_POLL",
    "GATEWAY_RECONNECT",
    "DISCORD_HEARTBEAT",
    "MCP_PROCESS_STOP",
    "MCP_PROCESS_KILL",
    "MCP_TOOL_CALL",
    "MCP_READLINE",
    "SANDBOX_DEFAULT",
    "SANDBOX_QUICK",
    "BROWSER_SESSION",
    "OLLAMA_DEFAULT",
    "OLLAMA_HEALTH",
    "HUB_DEFAULT",
    "HUB_UPLOAD",
    "WEB_FETCH_DEFAULT",
    "WEB_FETCH_SLOW",
    "WEB_FETCH_HEALTH",
    "SQLITE_BUSY",
    "SQLITE_CONNECT",
]


def _env(name: str, default: int | float) -> float:
    """Read timeout from env, falling back to default."""
    val = os.environ.get(f"CAVEMAN_TIMEOUT_{name}")
    return float(val) if val else float(default)


# ── HTTP / API calls ──
HTTP_DEFAULT = _env("HTTP", 30)          # General HTTP requests
HTTP_FAST = _env("HTTP_FAST", 10)        # Quick health checks, metadata
HTTP_SLOW = _env("HTTP_SLOW", 60)        # Large downloads, media
HTTP_LLM = _env("HTTP_LLM", 300)        # LLM provider calls (can be slow)
HTTP_IMAGE_GEN = _env("HTTP_IMAGE_GEN", 120)  # Image generation APIs
HTTP_TRANSCRIBE = _env("HTTP_TRANSCRIBE", 120)  # Audio transcription
HTTP_TTS = _env("HTTP_TTS", 60)          # Text-to-speech APIs
HTTP_EMBEDDING = _env("HTTP_EMBEDDING", 30)  # Embedding API calls

# ── Subprocess / shell ──
SUBPROCESS_FAST = _env("SUBPROCESS_FAST", 5)    # Quick checks (version, echo)
SUBPROCESS_DEFAULT = _env("SUBPROCESS_DEFAULT", 10)  # Normal commands
SUBPROCESS_SLOW = _env("SUBPROCESS_SLOW", 30)   # Builds, installs
SUBPROCESS_FLYWHEEL = _env("SUBPROCESS_FLYWHEEL", 600)  # Long-running analysis

# ── WebSocket / streaming ──
WS_PING = _env("WS_PING", 10)           # WebSocket ping timeout
WS_CONNECT = _env("WS_CONNECT", 15)     # WebSocket connection timeout
WS_MESSAGE = _env("WS_MESSAGE", 120)    # Waiting for WS message

# ── Internal async coordination ──
DRAIN_DEFAULT = _env("DRAIN", 3)         # Background task drain
DRAIN_LONG = _env("DRAIN_LONG", 5)       # Extended drain on shutdown
TASK_DEFAULT = _env("TASK", 600)         # Background task execution
TASK_SHORT = _env("TASK_SHORT", 300)     # Short background tasks

# ── Gateway / adapters ──
GATEWAY_POLL = _env("GATEWAY_POLL", 60)  # Polling interval
GATEWAY_RECONNECT = _env("GATEWAY_RECONNECT", 5)  # Reconnect backoff base
DISCORD_HEARTBEAT = _env("DISCORD_HEARTBEAT", 2100)  # Discord WS timeout

# ── MCP ──
MCP_PROCESS_STOP = _env("MCP_PROCESS_STOP", 5)  # MCP process shutdown
MCP_PROCESS_KILL = _env("MCP_PROCESS_KILL", 3)  # MCP process force kill
MCP_TOOL_CALL = _env("MCP_TOOL_CALL", 30)       # MCP tool execution
MCP_READLINE = _env("MCP_READLINE", 10)          # MCP stdio readline

# ── Sandbox ──
SANDBOX_DEFAULT = _env("SANDBOX", 10)    # Sandbox code execution
SANDBOX_QUICK = _env("SANDBOX_QUICK", 3) # Quick sandbox checks

# ── Browser ──
BROWSER_SESSION = _env("BROWSER_SESSION", 1800)  # Browser session lifetime

# ── Ollama ──
OLLAMA_DEFAULT = _env("OLLAMA", 120)     # Ollama inference
OLLAMA_HEALTH = _env("OLLAMA_HEALTH", 5) # Ollama health check

# ── Hub ──
HUB_DEFAULT = _env("HUB", 15)           # Hub API calls
HUB_UPLOAD = _env("HUB_UPLOAD", 30)     # Hub file uploads

# ── Web fetch ──
WEB_FETCH_DEFAULT = _env("WEB_FETCH", 15)  # Web page fetch
WEB_FETCH_SLOW = _env("WEB_FETCH_SLOW", 60)  # Heavy web fetch (JS render)
WEB_FETCH_HEALTH = _env("WEB_FETCH_HEALTH", 5)  # URL health check

# ── SQLite ──
SQLITE_BUSY = _env("SQLITE_BUSY", 5000)  # SQLite busy timeout (ms)
SQLITE_CONNECT = _env("SQLITE_CONNECT", 10)  # SQLite connection timeout
