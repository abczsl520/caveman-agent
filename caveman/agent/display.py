"""Display callbacks — UI output functions used by the agent loop.

This module exists to break the loop→tui→factory→loop circular dependency.
The agent loop needs display functions, but shouldn't depend on the CLI layer.

In TUI mode: these get overridden with rich-formatted versions.
In headless/gateway mode: these defaults are fine (logging only).
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# --- Overridable display callbacks ---
# These are module-level functions that can be monkey-patched by the TUI.
# Default: log-only (no terminal UI dependency).


def show_tool_call(tool_name: str, args: dict) -> None:
    """Display a tool call being made."""
    logger.info("Tool call: %s %s", tool_name, str(args)[:80])


def show_tool_result(tool_name: str, result: str, success: bool = True) -> None:
    """Display a tool result."""
    icon = "✓" if success else "✗"
    logger.info("%s %s: %s", icon, tool_name, result[:100])


def show_memory_nudge() -> None:
    """Display memory nudge activity."""
    logger.debug("Memory nudge running...")


def show_skill_nudge() -> None:
    """Display skill nudge activity."""
    logger.debug("Skill check triggered...")


def show_error(message: str) -> None:
    """Display an error."""
    logger.error("Agent error: %s", message)


def show_thinking() -> None:
    """Display thinking indicator."""
    logger.debug("Thinking...")
