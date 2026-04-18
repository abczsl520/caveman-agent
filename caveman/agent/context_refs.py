"""Context references — @file, @url, @diff inline expansion.

Inspired by Hermes context_references.py (MIT, Nous Research).
Parses @file:path, @url:url, @diff references in user messages
and expands them into context blocks.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_QUOTED_VALUE = r'(?:`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\')'
REFERENCE_RE = re.compile(
    rf"(?<![\w/])@(?:"
    rf"(?P<simple>diff|staged)\b"
    rf"|(?P<kind>file|folder|url):(?P<value>{_QUOTED_VALUE}(?::\d+(?:-\d+)?)?|\S+)"
    rf")"
)

_SENSITIVE_DIRS = frozenset({
    ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure",
})
_SENSITIVE_FILES = frozenset({
    ".ssh/id_rsa", ".ssh/id_ed25519", ".ssh/config",
    ".ssh/authorized_keys", ".netrc", ".pgpass", ".npmrc", ".pypirc",
})

# Approximate chars per token
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ContextRef:
    """A parsed context reference."""
    raw: str
    kind: str  # file | folder | url | diff | staged
    target: str
    start: int
    end: int
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class ExpansionResult:
    """Result of expanding context references."""
    message: str
    original: str
    refs: list[ContextRef] = field(default_factory=list)
    injected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tokens_added: int = 0
    blocked: bool = False


def parse_refs(message: str) -> list[ContextRef]:
    """Parse @file/@url/@diff references from a message."""
    if not message:
        return []

    refs = []
    for m in REFERENCE_RE.finditer(message):
        simple = m.group("simple")
        if simple:
            refs.append(ContextRef(
                raw=m.group(0), kind=simple, target="",
                start=m.start(), end=m.end(),
            ))
            continue

        kind = m.group("kind")
        value = (m.group("value") or "").strip(",.;!?")
        target = value.strip("`\"'")

        line_start = line_end = None
        if kind == "file" and ":" in target:
            parts = target.rsplit(":", 1)
            if parts[1].replace("-", "").isdigit():
                target = parts[0]
                line_range = parts[1]
                if "-" in line_range:
                    s, e = line_range.split("-", 1)
                    line_start, line_end = int(s), int(e)
                else:
                    line_start = line_end = int(line_range)

        refs.append(ContextRef(
            raw=m.group(0), kind=kind, target=target,
            start=m.start(), end=m.end(),
            line_start=line_start, line_end=line_end,
        ))

    return refs


def _is_sensitive(path: Path, home: Path) -> bool:
    """Check if a path points to sensitive files."""
    try:
        rel = path.resolve().relative_to(home)
    except ValueError:
        return False
    rel_str = str(rel)
    if any(rel_str.startswith(d) for d in _SENSITIVE_DIRS):
        return True
    if rel_str in _SENSITIVE_FILES:
        return True
    return False


def _read_file(path: Path, line_start: Optional[int], line_end: Optional[int]) -> str:
    """Read file content, optionally slicing by line range."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if line_start is not None:
        lines = text.splitlines(keepends=True)
        start = max(0, line_start - 1)
        end = line_end if line_end else start + 1
        text = "".join(lines[start:end])
    return text


def _read_folder(path: Path, max_files: int = 20) -> str:
    """List folder contents (non-recursive, limited)."""
    all_entries = sorted(path.iterdir())
    lines = []
    for entry in all_entries[:max_files]:
        prefix = "📁" if entry.is_dir() else "📄"
        lines.append(f"{prefix} {entry.name}")
    if len(all_entries) > max_files:
        lines.append(f"... and {len(all_entries) - max_files} more")
    return "\n".join(lines)


def _git_diff(staged: bool = False) -> str:
    """Get git diff output."""
    cmd = ["git", "diff"]
    if staged:
        cmd.append("--staged")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        return result.stdout[:50_000] if result.stdout else "(no changes)"
    except Exception as e:
        return f"(git diff failed: {e})"


def expand_refs(
    message: str,
    *,
    cwd: str | Path = ".",
    token_budget: int = 50_000,
) -> ExpansionResult:
    """Parse and expand all context references in a message.

    Returns the message with references replaced by content blocks.
    Respects token budget to avoid prompt overflow.
    """
    refs = parse_refs(message)
    if not refs:
        return ExpansionResult(message=message, original=message)

    cwd = Path(cwd).resolve()
    home = Path.home()
    result = ExpansionResult(message=message, original=message, refs=refs)
    tokens_used = 0
    replacements: list[tuple[int, int, str]] = []

    for ref in refs:
        content = ""
        label = ref.raw

        if ref.kind == "file":
            path = (cwd / ref.target).resolve()
            if _is_sensitive(path, home):
                result.warnings.append(f"Blocked sensitive file: {ref.target}")
                result.blocked = True
                continue
            if not path.is_file():
                result.warnings.append(f"File not found: {ref.target}")
                continue
            try:
                content = _read_file(path, ref.line_start, ref.line_end)
                label = f"[File: {ref.target}]"
            except Exception as e:
                result.warnings.append(f"Error reading {ref.target}: {e}")
                continue

        elif ref.kind == "folder":
            path = (cwd / ref.target).resolve()
            if not path.is_dir():
                result.warnings.append(f"Folder not found: {ref.target}")
                continue
            content = _read_folder(path)
            label = f"[Folder: {ref.target}]"

        elif ref.kind in ("diff", "staged"):
            content = _git_diff(staged=(ref.kind == "staged"))
            label = f"[Git {'staged ' if ref.kind == 'staged' else ''}diff]"

        elif ref.kind == "url":
            # URL fetching requires async — skip in sync expansion
            result.warnings.append(f"URL expansion not available in sync mode: {ref.target}")
            continue

        if not content:
            continue

        # Token budget check
        est_tokens = len(content) // _CHARS_PER_TOKEN
        if tokens_used + est_tokens > token_budget:
            # Truncate to fit
            remaining = (token_budget - tokens_used) * _CHARS_PER_TOKEN
            if remaining > 200:
                content = content[:remaining] + "\n... (truncated)"
                est_tokens = remaining // _CHARS_PER_TOKEN
            else:
                result.warnings.append(f"Token budget exceeded, skipping {ref.raw}")
                continue

        tokens_used += est_tokens
        block = f"\n{label}\n```\n{content}\n```\n"
        result.injected.append(block)
        replacements.append((ref.start, ref.end, block))

    # Apply replacements in reverse order to preserve positions
    new_message = message
    for start, end, block in reversed(sorted(replacements)):
        new_message = new_message[:start] + block + new_message[end:]

    result.message = new_message
    result.tokens_added = tokens_used
    return result
