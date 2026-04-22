"""Centralized constants — single source of truth for paths, defaults, and magic numbers.

All hardcoded "~/.caveman/..." paths, timeout values, token limits, and other
repeated constants should reference this module. Changing a value here changes
it everywhere.
"""
from __future__ import annotations
import os
from pathlib import Path

__all__ = [
    "CAVEMAN_HOME",
    "MEMORY_DIR",
    "SKILLS_DIR",
    "WIKI_DIR",
    "PLUGINS_DIR",
    "TRAJECTORIES_DIR",
    "SESSIONS_DIR",
    "TRAINING_DIR",
    "TRAINING_SFT_DIR",
    "TRAINING_RL_DIR",
    "HUB_CACHE_DIR",
    "CONFIG_PATH",
    "MEMORY_DB_PATH",
    "UDS_SOCK",
    "OPENCLAW_SOCK",
    "DEFAULT_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_MAX_TOKENS_ANTHROPIC",
    "DEFAULT_MAX_TOKENS_OPENAI",
    "DEFAULT_CONTEXT_WINDOW",
    "OPENAI_CONTEXT_WINDOW",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_COMPRESSION_THRESHOLD",
    "DEFAULT_SUBTASK_TIMEOUT",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_ACP_TIMEOUT",
    "BROWSER_NAV_TIMEOUT",
    "BROWSER_CLICK_TIMEOUT",
    "OPENCLAW_GATEWAY_PORT",
    "MCP_PROTOCOL_VERSION",
    "DEFAULT_MAX_SEQ_LENGTH",
    "EMBEDDING_MAX_INPUT",
    "TRUNCATE_SHORT",
    "TRUNCATE_MEDIUM",
    "TRUNCATE_LONG",
    "UDS_SOCKET_MODE",
    "JSONRPC_PARSE_ERROR",
    "JSONRPC_INTERNAL_ERROR",
    "PROJECTS_DIR",
    "LOGS_DIR",
    "REQUIRED_DIRS",
    "ensure_home",
    "DEFAULT_GATEWAY_PORT",
    "DEFAULT_GATEWAY_URL",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_HERMES_URL",
    "DEFAULT_LLM_IDLE_TIMEOUT",
]


# ── Paths ──

CAVEMAN_HOME = Path(os.environ.get("CAVEMAN_HOME", "~/.caveman")).expanduser()

# Sub-directories
MEMORY_DIR = CAVEMAN_HOME / "memory"
SKILLS_DIR = CAVEMAN_HOME / "skills"
WIKI_DIR = CAVEMAN_HOME / "wiki"
PLUGINS_DIR = CAVEMAN_HOME / "plugins"
TRAJECTORIES_DIR = CAVEMAN_HOME / "trajectories"
SESSIONS_DIR = CAVEMAN_HOME / "sessions"
TRAINING_DIR = CAVEMAN_HOME / "training"
TRAINING_SFT_DIR = TRAINING_DIR / "output"
TRAINING_RL_DIR = TRAINING_DIR / "rl_output"
HUB_CACHE_DIR = CAVEMAN_HOME / "hub_cache"

# Files
CONFIG_PATH = CAVEMAN_HOME / "config.yaml"
MEMORY_DB_PATH = MEMORY_DIR / "caveman.db"

# Socket paths
UDS_SOCK = "/tmp/caveman-bridge.sock"
OPENCLAW_SOCK = "/tmp/openclaw-mcp.sock"

# ── Provider defaults ──

DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS_ANTHROPIC = 8192
DEFAULT_MAX_TOKENS_OPENAI = 4096
DEFAULT_CONTEXT_WINDOW = 200_000
OPENAI_CONTEXT_WINDOW = 128_000

# ── Agent defaults ──

DEFAULT_MAX_ITERATIONS = 50
DEFAULT_COMPRESSION_THRESHOLD = 0.75
DEFAULT_SUBTASK_TIMEOUT = 300  # seconds
DEFAULT_SYSTEM_PROMPT = (
    "You are Caveman, an AI agent that learns, executes, and evolves.\n"
    "You have tools for bash, file ops, web search, and more.\n"
    "Think step-by-step. Be concise but thorough.\n"
    "Memory is automatic — Shield saves session state after each task."
)
DEFAULT_ACP_TIMEOUT = 300  # seconds

# ── Network / Browser ──

BROWSER_NAV_TIMEOUT = 30_000  # ms
BROWSER_CLICK_TIMEOUT = 10_000  # ms
OPENCLAW_GATEWAY_PORT = 18789
MCP_PROTOCOL_VERSION = "2024-11-05"

# ── Training ──

DEFAULT_MAX_SEQ_LENGTH = 4096

# ── Embedding ──

EMBEDDING_MAX_INPUT = 8000  # chars to truncate before sending to embedding API

# ── Display / Logging ──

TRUNCATE_SHORT = 100  # characters for log/display truncation
TRUNCATE_MEDIUM = 200  # for result previews
TRUNCATE_LONG = 300  # for turn summaries

# ── Security ──

UDS_SOCKET_MODE = 0o600  # restrictive permissions for Unix socket
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INTERNAL_ERROR = -32603

# ── Sessions (Shield) ──

PROJECTS_DIR = CAVEMAN_HOME / "projects"

# ── Logs ──

LOGS_DIR = CAVEMAN_HOME / "logs"


# ── Home directory initialization ──

REQUIRED_DIRS = [
    CAVEMAN_HOME,
    CAVEMAN_HOME / "workspace",
    CAVEMAN_HOME / "workspace" / "agents",
    MEMORY_DIR,
    SKILLS_DIR,
    PLUGINS_DIR,
    CAVEMAN_HOME / "cron",
    TRAJECTORIES_DIR,
    SESSIONS_DIR,
    TRAINING_DIR,
    PROJECTS_DIR,
    WIKI_DIR,
    HUB_CACHE_DIR,
    LOGS_DIR,
]


def ensure_home() -> None:
    """Create all required Caveman directories if they don't exist."""
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)

# Bridge defaults
DEFAULT_GATEWAY_PORT = 18789
DEFAULT_GATEWAY_URL = f"ws://127.0.0.1:{DEFAULT_GATEWAY_PORT}"

# Ollama defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_HERMES_URL = "http://localhost:8000"

# LLM idle timeout — abort if no token received for this long (seconds)
DEFAULT_LLM_IDLE_TIMEOUT = 120  # 2 minutes between tokens
