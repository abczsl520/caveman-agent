"""Auto-completion for slash commands."""
from __future__ import annotations

from caveman.commands.registry import COMMAND_REGISTRY, resolve_command


class CommandCompleter:
    """Provides tab-completion for slash commands, subcommands, and args."""

    def __init__(self) -> None:
        self._commands: list[str] = []
        self._rebuild()

    def _rebuild(self) -> None:
        """Rebuild completion list from registry."""
        names: set[str] = set()
        for cmd in COMMAND_REGISTRY:
            names.add(cmd.name)
            names.update(cmd.aliases)
        self._commands = sorted(names)

    def complete(self, text: str) -> list[str]:
        """Return completions for partial input.

        Supports:
        - Command name prefix: /mod → [/model]
        - Subcommand: /model <tab> → subcommands
        - Fuzzy: /eng → [/engines]
        """
        if not text.lstrip().startswith("/"):
            return []

        has_space = " " in text.strip() or text.rstrip() != text
        stripped = text.strip()
        parts = stripped.split(maxsplit=1)
        cmd_text = parts[0][1:]  # strip /

        if not has_space:
            # Completing command name
            return [f"/{c}" for c in self._commands if c.startswith(cmd_text)]

        # Completing subcommand/args
        cmd_def = resolve_command(cmd_text)
        if cmd_def and cmd_def.subcommands:
            prefix = parts[1] if len(parts) > 1 else ""
            return [s for s in cmd_def.subcommands if s.startswith(prefix)]

        return []

    def readline_completer(self, text: str, state: int):
        """readline-compatible completer function."""
        if state == 0:
            self._matches = self.complete(text)
        if state < len(self._matches):
            return self._matches[state]
        return None
