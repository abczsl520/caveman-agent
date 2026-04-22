"""Context References — compat shim.

Thin wrapper around context_refs.py for backward compatibility.
New code should use caveman.agent.context_refs directly.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

__all__ = [
    "Reference",
    "detect_references",
    "resolve_file_reference",
    "resolve_references",
    "build_context_from_references",
]

logger = logging.getLogger("caveman.agent.context_references")

# ── Reference Patterns (bare @file.py detection — different from context_refs) ──

_FILE_REF_RE = re.compile(
    r"(?:^|\s)(?:@|file:)([^\s]+\.(?:py|js|ts|md|txt|yaml|yml|json|toml|cfg|ini|sh|bash|zsh|rs|go|java|c|cpp|h|hpp|rb|php|swift|kt))",
    re.MULTILINE,
)
_URL_REF_RE = re.compile(r"https?://[^\s<>\"']+")
_LINE_REF_RE = re.compile(r":(\d+)(?:-(\d+))?$")


@dataclass
class Reference:
    """A resolved reference."""
    type: str  # file | url | symbol
    raw: str
    resolved_path: str = ""
    content: str = ""
    line_start: int = 0
    line_end: int = 0
    error: str = ""


def detect_references(text: str) -> List[Reference]:
    """Detect all references in text (bare @file.py and URLs)."""
    refs = []
    for match in _FILE_REF_RE.finditer(text):
        raw = match.group(1)
        ref = Reference(type="file", raw=raw)
        line_match = _LINE_REF_RE.search(raw)
        if line_match:
            ref.line_start = int(line_match.group(1))
            ref.line_end = int(line_match.group(2) or line_match.group(1))
            ref.raw = raw[:line_match.start()]
        refs.append(ref)
    for match in _URL_REF_RE.finditer(text):
        refs.append(Reference(type="url", raw=match.group()))
    return refs


def resolve_file_reference(ref: Reference, cwd: Optional[str] = None) -> Reference:
    """Resolve a file reference to its content."""
    search_paths = [Path(cwd or os.getcwd())]
    for subdir in ("src", "lib", "app", "tests", "caveman"):
        search_paths.append(search_paths[0] / subdir)
    for base in search_paths:
        path = base / ref.raw
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                if ref.line_start > 0:
                    lines = content.split("\n")
                    end = ref.line_end or ref.line_start
                    content = "\n".join(lines[ref.line_start - 1:end])
                if len(content) > 50_000:
                    content = content[:50_000] + "\n\n[... truncated ...]"
                ref.resolved_path = str(path)
                ref.content = content
                return ref
            except Exception as e:
                ref.error = str(e)
                return ref
    ref.error = f"File not found: {ref.raw}"
    return ref


def resolve_references(
    text: str,
    cwd: Optional[str] = None,
    resolve_urls: bool = False,
) -> List[Reference]:
    """Detect and resolve all references in text."""
    refs = detect_references(text)
    resolved = []
    for ref in refs:
        if ref.type == "file":
            resolved.append(resolve_file_reference(ref, cwd))
        elif ref.type == "url" and resolve_urls:
            resolved.append(ref)
        else:
            resolved.append(ref)
    return resolved


def build_context_from_references(refs: List[Reference]) -> str:
    """Build context text from resolved references."""
    sections = []
    for ref in refs:
        if ref.error:
            sections.append(f"<!-- Reference {ref.raw}: {ref.error} -->")
            continue
        if not ref.content:
            continue
        if ref.type == "file":
            line_info = ""
            if ref.line_start:
                line_info = f" (lines {ref.line_start}-{ref.line_end})"
            sections.append(
                f"## {ref.resolved_path or ref.raw}{line_info}\n"
                f"```\n{ref.content}\n```"
            )
        elif ref.type == "url":
            sections.append(f"## {ref.raw}\n{ref.content}")
    return "\n\n".join(sections)
