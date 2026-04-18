"""Unified command dispatcher — routes commands to handlers across surfaces."""
from __future__ import annotations

import importlib
from typing import Any

from caveman.commands.types import CommandContext
from caveman.commands.registry import resolve_command


def parse_command(raw: str) -> tuple[str | None, str]:
    """Parse raw input into (command_name, args).

    Returns (None, "") for non-command input.

    >>> parse_command("/model claude-sonnet-4-6")
    ('model', 'claude-sonnet-4-6')
    >>> parse_command("/help")
    ('help', '')
    >>> parse_command("not a command")
    (None, '')
    """
    raw = raw.strip()
    if not raw.startswith("/"):
        return (None, "")
    parts = raw.split(maxsplit=1)
    name = parts[0][1:].lower()  # strip leading /
    if not name:
        return (None, "")
    args = parts[1] if len(parts) > 1 else ""
    return (name, args)


async def dispatch(
    raw: str,
    agent: Any,
    surface: str = "cli",
    session_key: str = "",
    respond_fn: Any = None,
    locale: str = "en",
) -> str | None:
    """Dispatch a slash command to its handler.

    Returns:
        "exit" if the command signals quit,
        "handled" if processed,
        None if not a recognized command.
    """
    name, args = parse_command(raw)
    if not name:
        return None

    cmd_def = resolve_command(name)

    # Simple locale helper (same logic as ctx.t, avoids duplication)
    def _t(en: str, zh: str = "") -> str:
        return zh if locale.startswith("zh") and zh else en

    _respond = respond_fn or (lambda msg: None)

    if cmd_def is None:
        _respond(_t(f"Unknown command: /{name}. Try /help", f"未知命令: /{name}。试试 /帮助"))
        return "handled"

    if cmd_def.cli_only and surface != "cli":
        from caveman.commands.registry import ZH_COMMAND_NAMES
        dn = ZH_COMMAND_NAMES.get(cmd_def.name, cmd_def.name) if locale.startswith("zh") else cmd_def.name
        _respond(_t(f"/{dn} is CLI only.", f"/{dn} 仅限 CLI。"))
        return "handled"
    if cmd_def.gateway_only and surface in ("cli",):
        from caveman.commands.registry import ZH_COMMAND_NAMES
        dn = ZH_COMMAND_NAMES.get(cmd_def.name, cmd_def.name) if locale.startswith("zh") else cmd_def.name
        _respond(_t(f"/{dn} is gateway only.", f"/{dn} 仅限网关。"))
        return "handled"

    ctx = CommandContext(
        command=cmd_def.name,
        args=args,
        agent=agent,
        surface=surface,
        locale=locale,
        session_key=session_key,
        respond=_respond,
    )

    handler_fn = _resolve_handler(cmd_def.handler)
    if handler_fn is None:
        _respond(_t(f"Handler not found: /{cmd_def.name}", f"处理器未找到: /{cmd_def.name}"))
        return "handled"

    if cmd_def.name == "quit":
        _respond("👋")
        return "exit"

    await handler_fn(ctx)
    return "handled"


def _resolve_handler(handler_path: str):
    """Resolve 'session.handle_new' → actual function."""
    parts = handler_path.rsplit(".", 1)
    if len(parts) != 2:
        return None
    module_name, func_name = parts
    try:
        mod = importlib.import_module(f"caveman.commands.handlers.{module_name}")
        return getattr(mod, func_name, None)
    except ImportError:
        return None
