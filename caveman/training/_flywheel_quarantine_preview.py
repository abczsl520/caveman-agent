"""Quarantine restore-preview helpers for FlywheelDashboard."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any


def collect_restorable_quarantine_preview(cur: Any) -> dict[str, dict[str, int]]:
    """Return source/reason impact for currently quarantined memories."""
    by_source: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    rows = cur.execute(
        "SELECT metadata_json FROM memories "
        "WHERE json_valid(metadata_json) "
        "AND json_extract(metadata_json, '$.governance_state') = 'quarantined'"
    ).fetchall()
    for row in rows:
        try:
            meta = json.loads(str(row[0] or "{}"))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        by_source[str(meta.get("source") or "<missing>")] += 1
        by_reason[str(meta.get("quarantine_reason") or "<missing>")] += 1
    return {
        "restorable_quarantine_by_source": dict(sorted(by_source.items())),
        "restorable_quarantine_by_reason": dict(sorted(by_reason.items())),
    }
