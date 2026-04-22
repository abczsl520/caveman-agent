"""Caveman Import System — modular importers for external memory sources."""
from .base import BaseImporter, ImportItem, ImportManifest, ImportResult
from .openclaw import OpenClawImporter
from .openclaw_sessions import OpenClawSessionImporter
from .hermes import HermesImporter
from .claude_code import ClaudeCodeImporter
from .codex import CodexImporter
from .directory import DirectoryImporter

IMPORTERS: dict[str, type[BaseImporter]] = {
    "openclaw": OpenClawImporter,
    "openclaw-sessions": OpenClawSessionImporter,
    "hermes": HermesImporter,
    "claude-code": ClaudeCodeImporter,
    "codex": CodexImporter,
    "directory": DirectoryImporter,
}

__all__ = [
    "BaseImporter", "ImportItem", "ImportManifest", "ImportResult",
    "OpenClawImporter", "OpenClawSessionImporter", "HermesImporter",
    "ClaudeCodeImporter", "CodexImporter", "DirectoryImporter", "IMPORTERS",
]
