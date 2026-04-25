#!/usr/bin/env python3
"""Validate external user trial evidence for v1.0 gate (#33).

This does not fabricate real trials. It checks a JSON/JSONL evidence file and
exits 0 only when at least 3 non-author users have entries for Day 1 through Day
7 with feedback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_DAYS = set(range(1, 8))


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    if not isinstance(data, list):
        raise ValueError("trial evidence must be a list or {records: [...]} object")
    return data


def validate(records: list[dict], min_users: int = 3) -> tuple[bool, dict]:
    users: dict[str, set[int]] = {}
    feedback_counts: dict[str, int] = {}
    rejected: list[str] = []

    for idx, rec in enumerate(records):
        user_id = str(rec.get("user_id", "")).strip()
        day = rec.get("day")
        author = bool(rec.get("is_author", False))
        feedback = str(rec.get("feedback", "")).strip()
        if not user_id:
            rejected.append(f"record {idx}: missing user_id")
            continue
        if author:
            rejected.append(f"record {idx}: author/self trial is not external evidence")
            continue
        if not isinstance(day, int) or day not in REQUIRED_DAYS:
            rejected.append(f"record {idx}: invalid day {day!r}")
            continue
        if len(feedback) < 10:
            rejected.append(f"record {idx}: feedback too short")
            continue
        users.setdefault(user_id, set()).add(day)
        feedback_counts[user_id] = feedback_counts.get(user_id, 0) + 1

    complete_users = sorted(uid for uid, days in users.items() if days >= REQUIRED_DAYS)
    summary = {
        "total_records": len(records),
        "external_users_seen": len(users),
        "complete_users": complete_users,
        "complete_user_count": len(complete_users),
        "required_users": min_users,
        "required_days": sorted(REQUIRED_DAYS),
        "rejected": rejected,
    }
    return len(complete_users) >= min_users, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Caveman external trial evidence")
    parser.add_argument("path", help="JSON or JSONL evidence file")
    parser.add_argument("--min-users", type=int, default=3)
    args = parser.parse_args()

    records = load_records(Path(args.path))
    ok, summary = validate(records, min_users=args.min_users)
    summary["status"] = "passed" if ok else "failed"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
