"""Config validation — fail fast on bad configuration.

Instead of silently using wrong config values, validate at startup and give
clear error messages. Catches typos, wrong types, missing required fields.
"""
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


from caveman.errors import ConfigError


# ── Schema definition ──
# Each field: {type, required, default, choices, range, description}
# `range` accepts (min, max); either side may be None. A None max means
# "no arbitrary upper bound" — important for long-compounding autonomous work.

CONFIG_SCHEMA: dict[str, dict] = {
    "agent.default_model": {
        "type": str,
        "default": "claude-opus-4-6",
        "description": "Default LLM model",
    },
    "agent.max_iterations": {
        "type": int,
        "default": 50,
        "range": (1, None),
        "description": "Max agent loop iterations (no arbitrary upper bound; tune by task budget)",
    },
    "providers.anthropic.api_key": {
        "type": str,
        "default": "",
        "description": "Anthropic API key",
    },
    "providers.anthropic.base_url": {
        "type": str,
        "description": "Custom Anthropic API base URL",
    },
    "providers.anthropic.max_tokens": {
        "type": int,
        "default": 8192,
        "range": (100, 200000),
        "description": "Max output tokens for Anthropic",
    },
    "providers.openai.api_key": {
        "type": str,
        "default": "",
        "description": "OpenAI API key",
    },
    "providers.openai.max_tokens": {
        "type": int,
        "default": 4096,
        "range": (100, 128000),
        "description": "Max output tokens for OpenAI",
    },
    "providers.openai.reasoning_effort": {
        "type": str,
        "default": "high",
        "choices": ["minimal", "low", "medium", "high"],
        "description": "Reasoning effort for OpenAI reasoning models",
    },
    "memory.backend": {
        "type": str,
        "default": "local",
        "choices": ["local", "sqlite", "redis"],
        "description": "Memory storage backend",
    },
    "memory.local_dir": {
        "type": str,
        "description": "Memory storage directory",
    },
    "memory.embedding_backend": {
        "type": str,
        "default": "auto",
        "choices": ["auto", "openai", "ollama", "fastembed", "local", "none"],
        "description": "Embedding provider",
    },
    "memory.quality_gate.use_llm": {
        "type": bool,
        "default": False,
        "description": "Use the active provider as an LLM judge for memory write quality after heuristic filters",
    },
    "memory.quality_gate.mode": {
        "type": str,
        "default": "heuristic",
        "choices": ["heuristic", "llm", "off"],
        "description": "Memory write quality gate mode",
    },
    "memory.quality_gate.timeout_seconds": {
        "type": int,
        "default": 8,
        "range": (1, 60),
        "description": "Timeout budget for LLM quality judging",
    },
    "skills.local_dir": {
        "type": str,
        "description": "Skills directory",
    },
    "skills.auto_create": {
        "type": bool,
        "default": True,
        "description": "Auto-create skills from trajectories",
    },
    "trajectory.enabled": {
        "type": bool,
        "default": True,
        "description": "Enable trajectory recording",
    },
    "security.secret_scanning": {
        "type": bool,
        "default": True,
        "description": "Scan for secrets in tool output",
    },
    "bridges.openclaw.enabled": {
        "type": bool,
        "default": False,
        "description": "Enable OpenClaw MCP bridge",
    },
    "bridges.openclaw.port": {
        "type": int,
        "default": 18789,
        "range": (1024, 65535),
        "description": "OpenClaw gateway port",
    },
    # Gateway
    "gateway.discord.enabled": {
        "type": bool,
        "default": False,
        "description": "Enable Discord gateway",
    },
    "gateway.discord.token": {
        "type": str,
        "description": "Discord bot token",
    },
    "gateway.discord.prefix": {
        "type": str,
        "default": "!cave",
        "description": "Command prefix for Discord",
    },
    "gateway.telegram.enabled": {
        "type": bool,
        "default": False,
        "description": "Enable Telegram gateway",
    },
    "gateway.telegram.token": {
        "type": str,
        "description": "Telegram bot token",
    },
    "gateway.limits.max_tool_calls": {
        "type": int,
        "nullable": True,
        "range": (1, None),
        "description": "Optional explicit per-task tool-call budget; omit/null for no arbitrary tool-count cap",
    },
    # Compression
    "compression.threshold": {
        "type": float,
        "default": 0.75,
        "range": (0.3, 0.95),
        "description": "Context utilization threshold for compression",
    },
    "compression.tail_budget": {
        "type": int,
        "default": 200000,
        "range": (10000, 1000000),
        "description": "Token budget for tail protection during compression",
    },
    # Delegation
    "delegation.max_iterations": {
        "type": int,
        "default": 30,
        "range": (1, None),
        "description": "Max iterations for delegated subtasks (no arbitrary upper bound; caller controls budget)",
    },
    "delegation.max_concurrent": {
        "type": int,
        "default": 3,
        "range": (1, 10),
        "description": "Max concurrent delegated tasks",
    },
    # Cron
    "cron.enabled": {
        "type": bool,
        "default": True,
        "description": "Enable cron scheduler",
    },
    "cron.max_concurrent": {
        "type": int,
        "default": 3,
        "range": (1, 10),
        "description": "Max concurrent cron jobs",
    },
    # Sandbox
    "sandbox.backend": {
        "type": str,
        "default": "subprocess",
        "choices": ["subprocess", "docker"],
        "description": "Sandbox execution backend",
    },
    "sandbox.docker_image": {
        "type": str,
        "default": "python:3.12-slim",
        "description": "Docker image for sandbox execution",
    },
}


_MISSING = object()


def _get_nested(config: dict, path: str) -> Any:
    """Get a value from a nested dict using dot notation."""
    keys = path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def validate_config(config: dict, strict: bool = True) -> list[str]:
    """Validate config against schema. Returns list of warnings.

    Args:
        config: Loaded config dict
        strict: If True, raise ConfigError on any issue. If False, just warn.

    Returns: List of warning messages (empty = all good)
    """
    warnings: list[str] = []

    for path, schema in CONFIG_SCHEMA.items():
        value = _get_nested(config, path)

        if value is _MISSING:
            continue  # Optional fields that aren't set

        # Null explicitly disables optional limits (e.g. no arbitrary tool-count cap).
        if value is None and schema.get("nullable", False):
            continue

        # Type check
        expected_type = schema.get("type")
        if expected_type and not isinstance(value, expected_type):
            msg = f"Config '{path}': expected {expected_type.__name__}, got {type(value).__name__} ({value!r})"
            warnings.append(msg)
            continue

        # Range check
        value_range = schema.get("range")
        if value_range and isinstance(value, (int, float)):
            lo, hi = value_range
            below_min = lo is not None and value < lo
            above_max = hi is not None and value > hi
            if below_min or above_max:
                if hi is None:
                    bound = f">= {lo}"
                elif lo is None:
                    bound = f"<= {hi}"
                else:
                    bound = f"[{lo}, {hi}]"
                msg = f"Config '{path}': value {value} out of range {bound}"
                warnings.append(msg)

        # Choices check
        choices = schema.get("choices")
        if choices and value not in choices:
            msg = f"Config '{path}': value {value!r} not in {choices}"
            warnings.append(msg)

    # Check for unknown top-level keys
    known_top = {"agent", "providers", "memory", "skills", "trajectory", "security", "bridges", "gateway", "caveman", "compression", "engines", "locale", "delegation", "cron", "sandbox"}
    for key in config:
        if key not in known_top:
            warnings.append(f"Unknown config key: '{key}' (typo?)")

    if strict and warnings:
        raise ConfigError("Config validation failed:\n" + "\n".join(f"  - {w}" for w in warnings))

    for w in warnings:
        logger.warning("Config: %s", w)

    return warnings


def get_config_help() -> str:
    """Generate human-readable config documentation from schema."""
    lines = ["# Caveman Configuration Reference", ""]
    current_section = ""
    for path, schema in sorted(CONFIG_SCHEMA.items()):
        section = path.split(".")[0]
        if section != current_section:
            current_section = section
            lines.append(f"\n## {section}")
        desc = schema.get("description", "")
        typ = schema.get("type", Any).__name__ if "type" in schema else "any"
        default = schema.get("default", "")
        choices = schema.get("choices", [])

        line = f"  {path}: {typ}"
        if default:
            line += f" (default: {default})"
        if choices:
            line += f" — one of: {choices}"
        if desc:
            line += f"  # {desc}"
        lines.append(line)
    return "\n".join(lines)
