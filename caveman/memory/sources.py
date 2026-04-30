"""Canonical memory source taxonomy and source governance policy.

Centralizing these labels prevents decay, dashboard diagnostics, migrations and
importers from drifting into subtly different source spellings.
"""
from __future__ import annotations

__all__ = [
    "IMPORT_SOURCE_PREFIX",
    "SOURCE_ALIASES",
    "SOURCE_POLICY_LOW_SIGNAL_IMPORTS",
    "canonicalize_memory_source",
]

IMPORT_SOURCE_PREFIX = "import:"

SOURCE_ALIASES: dict[str, str] = {
    "import:openclaw_sessions": "import:openclaw",
    "import:openclaw-sessions": "import:openclaw",
    "openclaw_sessions": "import:openclaw",
    "openclaw-session": "import:openclaw-session",
    "openclaw": "import:openclaw",
    "hermes": "import:hermes",
    "hermes-skill-ref": "import:hermes-skill-ref",
}

SOURCE_POLICY_LOW_SIGNAL_IMPORTS = frozenset({
    "import:openclaw",
    "import:openclaw-session",
    "import:hermes",
    "import:hermes-skill-ref",
})


def canonicalize_memory_source(source: object) -> str:
    """Return the canonical source label for known legacy spelling variants."""
    if not isinstance(source, str):
        return ""
    cleaned = source.strip()
    if not cleaned:
        return ""
    return SOURCE_ALIASES.get(cleaned.lower(), cleaned)
