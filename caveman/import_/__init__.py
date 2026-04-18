"""Caveman Import System — modular importers for external memory sources."""
from .base import BaseImporter, ImportItem, ImportManifest, ImportResult
from .openclaw import OpenClawImporter
from .hermes import HermesImporter
from .claude_code import ClaudeCodeImporter
from .codex import CodexImporter
from .directory import DirectoryImporter

IMPORTERS: dict[str, type[BaseImporter]] = {
    "openclaw": OpenClawImporter,
    "hermes": HermesImporter,
    "claude-code": ClaudeCodeImporter,
    "codex": CodexImporter,
    "directory": DirectoryImporter,
}

__all__ = [
    "BaseImporter", "ImportItem", "ImportManifest", "ImportResult",
    "OpenClawImporter", "HermesImporter", "ClaudeCodeImporter",
    "CodexImporter", "DirectoryImporter", "IMPORTERS",
]
