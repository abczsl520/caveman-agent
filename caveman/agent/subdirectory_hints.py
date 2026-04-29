"""Subdirectory Hints — codebase navigation assistance.

Generates intelligent hints about directory structure to help the agent
navigate large codebases efficiently without reading every file.

Extracted from Hermes agent/subdirectory_hints.py (224 lines).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

__all__ = [
    "DirHint",
    "generate_hints",
    "format_hints_for_prompt",
]


logger = logging.getLogger("caveman.agent.subdirectory_hints")

# Directories to always skip
_SKIP_DIRS: Set[str] = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".next", "dist", "build", ".cache", ".tox", ".mypy_cache",
    "target", "vendor", ".gradle", ".idea", ".vscode",
    "coverage", ".nyc_output", "egg-info",
}

# File extensions that indicate project type
_LANG_INDICATORS = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".js": "JavaScript",
    ".rs": "Rust",
    ".go": "Go",
    ".java": "Java",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".cs": "C#",
    ".cpp": "C++",
    ".c": "C",
}

# Config files that reveal project structure
_CONFIG_FILES = {
    "package.json": "Node.js project",
    "pyproject.toml": "Python project",
    "Cargo.toml": "Rust project",
    "go.mod": "Go module",
    "pom.xml": "Java/Maven project",
    "build.gradle": "Java/Gradle project",
    "Gemfile": "Ruby project",
    "Makefile": "Build system",
    "Dockerfile": "Containerized",
    "docker-compose.yml": "Multi-container",
    ".env": "Environment config",
    "tsconfig.json": "TypeScript project",
}


@dataclass
class DirHint:
    """Hint about a directory's purpose."""
    path: str
    purpose: str = ""
    languages: List[str] = field(default_factory=list)
    file_count: int = 0
    key_files: List[str] = field(default_factory=list)
    subdirs: List[str] = field(default_factory=list)


def generate_hints(
    root: str,
    *,
    max_depth: int = 3,
    max_files_per_dir: int = 20,
) -> List[DirHint]:
    """Generate navigation hints for a directory tree."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    hints: List[DirHint] = []
    _scan_dir(root_path, root_path, hints, depth=0, max_depth=max_depth, max_files=max_files_per_dir)
    return hints


def _scan_dir(
    path: Path,
    root: Path,
    hints: List[DirHint],
    depth: int,
    max_depth: int,
    max_files: int,
) -> None:
    """Recursively scan directory and generate hints."""
    if depth > max_depth:
        return

    if path.name in _SKIP_DIRS:
        return

    try:
        entries = list(path.iterdir())
    except PermissionError:
        return

    files = [e for e in entries if e.is_file()]
    dirs = [e for e in entries if e.is_dir() and e.name not in _SKIP_DIRS]

    # Detect languages
    languages: Set[str] = set()
    for f in files:
        lang = _LANG_INDICATORS.get(f.suffix.lower())
        if lang:
            languages.add(lang)

    # Detect purpose from config files
    purpose = ""
    key_files = []
    for f in files[:max_files]:
        if f.name in _CONFIG_FILES:
            purpose = _CONFIG_FILES[f.name]
            key_files.append(f.name)
        elif f.name in ("README.md", "README.rst", "README.txt"):
            key_files.append(f.name)
        elif f.name in ("main.py", "index.ts", "index.js", "app.py", "server.py"):
            key_files.append(f.name)

    # Infer purpose from directory name
    if not purpose:
        purpose = _infer_purpose(path.name)

    rel_path = str(path.relative_to(root)) if path != root else "."

    hints.append(DirHint(
        path=rel_path,
        purpose=purpose,
        languages=sorted(languages),
        file_count=len(files),
        key_files=key_files[:5],
        subdirs=[d.name for d in dirs[:10]],
    ))

    # Recurse into subdirectories
    for d in dirs:
        _scan_dir(d, root, hints, depth + 1, max_depth, max_files)


def _infer_purpose(name: str) -> str:
    """Infer directory purpose from its name."""
    purposes = {
        "src": "Source code",
        "lib": "Library code",
        "tests": "Test suite",
        "test": "Test suite",
        "docs": "Documentation",
        "scripts": "Utility scripts",
        "config": "Configuration",
        "utils": "Utilities",
        "helpers": "Helper functions",
        "models": "Data models",
        "views": "View layer",
        "controllers": "Controllers",
        "routes": "API routes",
        "middleware": "Middleware",
        "services": "Service layer",
        "components": "UI components",
        "pages": "Page components",
        "api": "API endpoints",
        "db": "Database",
        "migrations": "DB migrations",
        "static": "Static assets",
        "public": "Public assets",
        "templates": "Templates",
        "plugins": "Plugins",
        "extensions": "Extensions",
    }
    return purposes.get(name.lower(), "")


def format_hints_for_prompt(hints: List[DirHint], max_lines: int = 30) -> str:
    """Format hints as a concise string for the system prompt."""
    lines = ["Project structure:"]
    for hint in hints[:max_lines]:
        parts = [f"  {hint.path}/"]
        if hint.purpose:
            parts.append(f" — {hint.purpose}")
        if hint.languages:
            parts.append(f" [{', '.join(hint.languages)}]")
        if hint.file_count > 0:
            parts.append(f" ({hint.file_count} files)")
        lines.append("".join(parts))
    return "\n".join(lines)
