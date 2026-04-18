#!/usr/bin/env python3
"""Auto-append CHANGELOG.md from recent git commits.

Usage: python scripts/auto-changelog.py [--since HASH]

Reads git log since last CHANGELOG entry, groups by flywheel round,
and prepends new entries. Idempotent — skips already-logged commits.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"


def git_log(since: str | None = None) -> list[dict]:
    """Get commits as [{hash, subject, body}]."""
    cmd = ["git", "log", "--format=%H|%s|%b---END---"]
    if since:
        cmd.append(f"{since}..HEAD")
    else:
        cmd.extend(["-50"])  # last 50 if no anchor
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        return []
    commits = []
    for block in result.stdout.split("---END---"):
        block = block.strip()
        if not block:
            continue
        parts = block.split("|", 2)
        if len(parts) >= 2:
            commits.append({
                "hash": parts[0][:7],
                "subject": parts[1].strip(),
                "body": parts[2].strip() if len(parts) > 2 else "",
            })
    return commits


def extract_round(subject: str) -> str | None:
    """Extract 'Round N' from commit subject."""
    m = re.search(r'[Rr]ound\s*(\d+)', subject)
    return f"Round {m.group(1)}" if m else None


def existing_hashes() -> set[str]:
    """Get commit hashes already in CHANGELOG."""
    if not CHANGELOG.exists():
        return set()
    return set(re.findall(r'\b([0-9a-f]{7})\b', CHANGELOG.read_text()))


def main():
    since = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--since" else None
    commits = git_log(since)
    if not commits:
        print("No new commits.")
        return

    known = existing_hashes()
    new_commits = [c for c in commits if c["hash"] not in known]
    if not new_commits:
        print("CHANGELOG already up to date.")
        return

    # Group by round
    grouped: dict[str, list[dict]] = {}
    for c in reversed(new_commits):  # oldest first
        rnd = extract_round(c["subject"]) or "Ungrouped"
        grouped.setdefault(rnd, []).append(c)

    # Build new entries
    lines = []
    for rnd, commits_in_round in grouped.items():
        lines.append(f"\n## {rnd}")
        for c in commits_in_round:
            # Clean subject: remove round reference for cleaner display
            subj = re.sub(r'\s*\(?[Rr]ound\s*\d+\)?\s*', ' ', c["subject"]).strip()
            subj = re.sub(r'\s*—\s*$', '', subj).strip()
            lines.append(f"- `{c['hash']}` {subj}")
    new_section = "\n".join(lines) + "\n"

    # Prepend after header
    if CHANGELOG.exists():
        content = CHANGELOG.read_text()
        # Insert after the first two lines (title + blank)
        header_end = content.find("\n\n")
        if header_end > 0:
            updated = content[:header_end + 2] + new_section + content[header_end + 2:]
        else:
            updated = content + "\n" + new_section
    else:
        updated = "# Changelog\n\nAll notable changes to Caveman, organized by flywheel round.\n" + new_section

    CHANGELOG.write_text(updated)
    print(f"Added {len(new_commits)} commits to CHANGELOG.md")


if __name__ == "__main__":
    main()
