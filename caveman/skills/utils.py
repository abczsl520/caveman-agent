"""Skill utilities — frontmatter parsing, platform filtering, discovery.

Ported from Hermes skill_utils.py (MIT, Nous Research).
Adapted for Caveman's skill system.
"""
from __future__ import annotations

__all__ = [
    "parse_frontmatter",
    "skill_matches_platform",
    "extract_skill_description",
    "iter_skill_files",
    "discover_skills",
]

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PLATFORM_MAP = {
    "macos": "darwin",
    "linux": "linux",
    "windows": "win32",
}

_EXCLUDED_DIRS = frozenset((".git", ".github", ".hub", "__pycache__", "node_modules"))

# Lazy YAML loader
_yaml_load_fn = None


def _yaml_load(content: str) -> Any:
    """Parse YAML with lazy import and CSafeLoader preference."""
    global _yaml_load_fn
    if _yaml_load_fn is None:
        try:
            import yaml
            loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader
            _yaml_load_fn = lambda v: yaml.load(v, Loader=loader)  # noqa: E731
        except ImportError:
            _yaml_load_fn = lambda v: {}  # noqa: E731
    return _yaml_load_fn(content)


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown string.

    Returns (frontmatter_dict, remaining_body).
    """
    if not content.startswith("---"):
        return {}, content

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    yaml_content = content[3:end_match.start() + 3]
    body = content[end_match.end() + 3:]

    try:
        parsed = _yaml_load(yaml_content)
        if isinstance(parsed, dict):
            return parsed, body
    except Exception:
        # Fallback: simple key:value parsing
        fm: dict[str, Any] = {}
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fm[key.strip()] = value.strip()
        return fm, body

    return {}, body


def skill_matches_platform(frontmatter: dict[str, Any]) -> bool:
    """Check if a skill is compatible with the current OS.

    Skills declare platform requirements via ``platforms`` list in frontmatter:
        platforms: [macos, linux]

    If absent or empty, the skill is compatible with all platforms.
    """
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = sys.platform
    for platform in platforms:
        normalized = str(platform).lower().strip()
        mapped = PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
    return False


def extract_skill_description(frontmatter: dict[str, Any]) -> str:
    """Extract a truncated description from frontmatter."""
    raw = frontmatter.get("description", "")
    if not raw:
        return ""
    desc = str(raw).strip().strip("'\"")
    return desc[:57] + "..." if len(desc) > 60 else desc


def extract_conditions(frontmatter: dict[str, Any]) -> dict[str, list]:
    """Extract conditional activation fields from frontmatter metadata."""
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return {"requires_tools": [], "fallback_for_tools": []}
    caveman = metadata.get("caveman") or metadata.get("hermes") or {}
    if not isinstance(caveman, dict):
        return {"requires_tools": [], "fallback_for_tools": []}
    return {
        "requires_tools": caveman.get("requires_tools", []),
        "fallback_for_tools": caveman.get("fallback_for_tools", []),
    }


def iter_skill_files(
    skills_dir: Path, filename: str = "SKILL.md",
) -> list[Path]:
    """Walk skills_dir yielding sorted paths matching filename."""
    matches: list[Path] = []
    if not skills_dir.is_dir():
        return matches
    for root, dirs, files in os.walk(skills_dir):
        dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
        if filename in files:
            matches.append(Path(root) / filename)
    matches.sort(key=lambda p: str(p.relative_to(skills_dir)))
    return matches


def discover_skills(
    skills_dirs: list[Path] | None = None,
    disabled: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover all skills from directories, returning metadata dicts.

    Each dict has: name, description, path, platforms, conditions.
    """
    if skills_dirs is None:
        from caveman.paths import SKILLS_DIR
        skills_dirs = [SKILLS_DIR]

    disabled = disabled or set()
    results: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for skills_dir in skills_dirs:
        for skill_file in iter_skill_files(skills_dir):
            try:
                raw = skill_file.read_text(encoding="utf-8")
                fm, body = parse_frontmatter(raw)
            except Exception:
                continue

            name = fm.get("name") or skill_file.parent.name
            name = str(name)
            if name in disabled or name in seen_names:
                continue
            if not skill_matches_platform(fm):
                continue

            seen_names.add(name)
            results.append({
                "name": name,
                "description": extract_skill_description(fm),
                "path": str(skill_file),
                "platforms": fm.get("platforms", []),
                "conditions": extract_conditions(fm),
                "version": fm.get("version", ""),
            })

    return results
