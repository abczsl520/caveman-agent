"""Caveman Changelog — auto-generate from git log."""
from __future__ import annotations
import subprocess
from collections import defaultdict


def generate_changelog(n: int = 20) -> str:
    """Generate changelog from recent git commits."""
    result = subprocess.run(
        ["git", "log", f"-{n}", "--pretty=format:%h|%s|%ai"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        return "Error: not a git repository"

    categories: dict[str, list[str]] = defaultdict(list)
    for line in result.stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        sha, msg, date = parts
        date_short = date[:10]

        # Categorize by conventional commit prefix
        if msg.startswith("feat"):
            cat = "✨ Features"
        elif msg.startswith("fix"):
            cat = "🐛 Fixes"
        elif msg.startswith("test"):
            cat = "🧪 Tests"
        elif msg.startswith("chore") or msg.startswith("docs"):
            cat = "📝 Chores"
        else:
            cat = "🔧 Other"

        categories[cat].append(f"  {sha} {msg} ({date_short})")

    parts = [f"Caveman Changelog (last {n} commits)\n"]
    for cat in ["✨ Features", "🐛 Fixes", "🧪 Tests", "📝 Chores", "🔧 Other"]:
        if cat in categories:
            parts.append(f"{cat}:")
            parts.extend(categories[cat])
            parts.append("")

    return "\n".join(parts)
