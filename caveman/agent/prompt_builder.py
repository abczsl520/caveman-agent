"""Prompt Builder — system prompt construction with context files.

Builds the system prompt from SOUL.md, AGENTS.md, project context,
skills manifest, and environment hints. Extracted from Hermes
agent/prompt_builder.py (1025 lines).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

__all__ = [
    "CONTEXT_FILE_MAX_CHARS",
    "find_git_root",
    "strip_yaml_frontmatter",
    "truncate_content",
    "scan_context_content",
    "load_context_file",
    "load_soul_md",
    "build_environment_hints",
    "SkillEntry",
    "build_skills_manifest",
    "build_skills_prompt",
    "PromptConfig",
    "build_system_prompt",
]


logger = logging.getLogger("caveman.agent.prompt_builder")

CONTEXT_FILE_MAX_CHARS = 20_000
_YAML_FM_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


# ── Context File Discovery ──

def find_git_root(start: Path) -> Optional[Path]:
    """Walk up to find the git root directory."""
    current = start.resolve()
    for _ in range(20):
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def strip_yaml_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    return _YAML_FM_RE.sub("", content, count=1)


def truncate_content(content: str, filename: str, max_chars: int = CONTEXT_FILE_MAX_CHARS) -> str:
    """Truncate content with a warning marker."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n\n[... {filename} truncated at {max_chars:,} chars ...]"


def scan_context_content(content: str, filename: str) -> str:
    """Process a context file: strip frontmatter, truncate."""
    content = strip_yaml_frontmatter(content)
    content = truncate_content(content, filename)
    return content.strip()


# ── Context File Loaders ──

_CONTEXT_FILE_NAMES = {
    "caveman": [".caveman.md", "CAVEMAN.md"],
    "hermes": [".hermes.md", "HERMES.md"],
    "agents": ["AGENTS.md", "agents.md"],
    "claude": ["CLAUDE.md", "claude.md"],
    "cursor": [".cursorrules"],
}


def load_context_file(cwd: Path, category: str) -> Optional[str]:
    """Load a context file by category."""
    names = _CONTEXT_FILE_NAMES.get(category, [])
    search_dirs = [cwd]

    # For caveman/hermes, also search up to git root
    if category in ("caveman", "hermes"):
        git_root = find_git_root(cwd)
        if git_root and git_root != cwd:
            search_dirs.append(git_root)

    for search_dir in search_dirs:
        for name in names:
            path = search_dir / name
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                    processed = scan_context_content(content, name)
                    if processed:
                        return f"## {path}\n{processed}"
                except Exception as e:
                    logger.debug("Failed to read %s: %s", path, e)
    return None


def load_soul_md(home_dir: Optional[Path] = None) -> Optional[str]:
    """Load SOUL.md from caveman home directory."""
    home = home_dir or Path.home() / ".caveman"
    for name in ("SOUL.md", "soul.md"):
        path = home / name
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                return scan_context_content(content, name)
            except Exception as exc:
                logger.debug("load_soul_md: suppressed %s", exc)
    return None


# ── Environment Hints ──

def build_environment_hints() -> str:
    """Build environment hints for the system prompt."""
    import platform
    import shutil

    hints = []
    hints.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    hints.append(f"Python: {platform.python_version()}")
    hints.append(f"CWD: {os.getcwd()}")

    # Check for common tools
    for tool in ("git", "node", "npm", "docker", "ffmpeg"):
        if shutil.which(tool):
            hints.append(f"{tool}: available")

    # Shell
    shell = os.environ.get("SHELL", "")
    if shell:
        hints.append(f"Shell: {Path(shell).name}")

    return "Environment:\n" + "\n".join(f"  {h}" for h in hints)


# ── Skills Manifest ──

@dataclass
class SkillEntry:
    """A skill in the manifest."""
    name: str
    description: str = ""
    location: str = ""
    triggers: List[str] = field(default_factory=list)
    category: str = ""


def build_skills_manifest(skills_dir: Optional[Path] = None) -> List[SkillEntry]:
    """Build a manifest of available skills."""
    skills_dir = skills_dir or Path.home() / ".caveman" / "skills"
    if not skills_dir.exists():
        return []

    entries = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            # Extract description from first paragraph or frontmatter
            desc = _extract_skill_description(content)
            triggers = _extract_skill_triggers(content)
            entries.append(SkillEntry(
                name=skill_dir.name,
                description=desc,
                location=str(skill_md),
                triggers=triggers,
            ))
        except Exception as e:
            logger.debug("Failed to parse skill %s: %s", skill_dir.name, e)

    return entries


def _extract_skill_description(content: str) -> str:
    """Extract description from SKILL.md."""
    # Try frontmatter
    fm_match = _YAML_FM_RE.match(content)
    if fm_match:
        fm = fm_match.group()
        for line in fm.split("\n"):
            if line.strip().startswith("description:"):
                return line.split(":", 1)[1].strip().strip("'\"")

    # First non-empty, non-heading line
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("---"):
            return line[:200]
    return ""


def _extract_skill_triggers(content: str) -> List[str]:
    """Extract trigger words from SKILL.md."""
    triggers = []
    for line in content.split("\n"):
        lower = line.lower().strip()
        if "trigger" in lower and ":" in lower:
            # Extract quoted strings
            for match in re.finditer(r"['\"]([^'\"]+)['\"]", line):
                triggers.append(match.group(1))
    return triggers


def build_skills_prompt(skills: List[SkillEntry]) -> str:
    """Build the skills section of the system prompt."""
    if not skills:
        return ""

    lines = ["<available_skills>"]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{skill.name}</name>")
        lines.append(f"    <description>{skill.description}</description>")
        lines.append(f"    <location>{skill.location}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


# ── Full System Prompt Builder ──

@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    cwd: str = ""
    home_dir: Optional[Path] = None
    skills_dir: Optional[Path] = None
    include_env: bool = True
    include_skills: bool = True
    include_soul: bool = True
    skip_soul_in_context: bool = True  # Avoid double-loading
    model: str = ""
    extra_sections: List[str] = field(default_factory=list)


def build_system_prompt(config: Optional[PromptConfig] = None) -> str:
    """Build the complete system prompt."""
    config = config or PromptConfig()
    cwd = Path(config.cwd or os.getcwd()).resolve()
    sections = []

    # 1. Soul/Identity
    if config.include_soul:
        soul = load_soul_md(config.home_dir)
        if soul:
            sections.append(soul)

    # 2. Project context (priority: caveman > hermes > agents > claude > cursor)
    project_ctx = None
    for category in ("caveman", "hermes", "agents", "claude", "cursor"):
        project_ctx = load_context_file(cwd, category)
        if project_ctx:
            break
    if project_ctx:
        sections.append(project_ctx)

    # 3. Environment hints
    if config.include_env:
        sections.append(build_environment_hints())

    # 3b. Subdirectory hints (codebase navigation)
    if config.include_env and cwd.exists():
        from caveman.agent.subdirectory_hints import generate_hints, format_hints_for_prompt
        hints = generate_hints(str(cwd), max_depth=2)
        if hints:
            hint_text = format_hints_for_prompt(hints, max_lines=20)
            if hint_text:
                sections.append(hint_text)

    # 4. Skills manifest
    if config.include_skills:
        skills = build_skills_manifest(config.skills_dir)
        skills_prompt = build_skills_prompt(skills)
        if skills_prompt:
            sections.append(skills_prompt)

    # 5. Extra sections
    for extra in config.extra_sections:
        if extra.strip():
            sections.append(extra)

    if not sections:
        return ""

    return "\n\n".join(sections)
