# Internal helpers for flywheel dashboard; not public API.
from __future__ import annotations

import json
from typing import Any

from caveman.training._flywheel_dashboard_values import _count_value, _number_value


def _memory_columns(cur: Any) -> set[str]:
    """Return available memory table columns for legacy-schema tolerant diagnostics."""
    return {str(row[1]) for row in cur.execute("PRAGMA table_info(memories)").fetchall()}


def _memory_breakdown_label(value: object, max_length: int = 80) -> str:
    """Return a compact display label for dashboard diagnostics."""
    label = str(value or "<missing>").replace("\n", " ").replace("\r", " ")
    if len(label) > max_length:
        return label[: max_length - 1] + "…"
    return label


def _memory_breakdown_row(
    label: str,
    total: int,
    trust_sum: float,
    never_recalled: int,
    helpful: int,
) -> dict[str, Any]:
    """Build one memory diagnostic row with rounded rates."""
    return {
        "label": label,
        "total": total,
        "avg_trust": round(trust_sum / total, 3) if total else 0.0,
        "never_recalled": never_recalled,
        "never_recalled_pct": round(never_recalled / total, 3) if total else 0.0,
        "helpful": helpful,
        "helpful_pct": round(helpful / total, 3) if total else 0.0,
    }


def _collect_memory_type_breakdown(cur: Any) -> list[dict[str, Any]]:
    """Break memory health down by type so global averages do not hide skew."""
    rows = cur.execute(
        "SELECT type, COUNT(*) AS total, COALESCE(SUM(trust_score), 0) AS trust_sum, "
        "SUM(CASE WHEN COALESCE(retrieval_count, 0) = 0 THEN 1 ELSE 0 END) AS never_recalled, "
        "SUM(CASE WHEN COALESCE(helpful_count, 0) > 0 THEN 1 ELSE 0 END) AS helpful "
        "FROM memories GROUP BY type ORDER BY total DESC, helpful DESC, type ASC"
    ).fetchall()
    return [
        _memory_breakdown_row(
            _memory_breakdown_label(row[0]),
            int(row[1] or 0),
            float(row[2] or 0.0),
            int(row[3] or 0),
            int(row[4] or 0),
        )
        for row in rows
    ]


def _collect_memory_source_breakdown(cur: Any, limit: int = 12) -> list[dict[str, Any]]:
    """Break memory health down by metadata.source, tolerating malformed metadata."""
    rows = cur.execute(
        "SELECT metadata_json, trust_score, COALESCE(retrieval_count, 0), "
        "COALESCE(helpful_count, 0) FROM memories"
    ).fetchall()
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            meta = json.loads(row[0] or "{}")
        except (TypeError, json.JSONDecodeError):
            meta = {}
        source = "<missing>"
        if isinstance(meta, dict):
            raw_source = meta.get("source")
            if raw_source:
                source = _memory_breakdown_label(raw_source)
        bucket = grouped.setdefault(source, {"total": 0, "trust_sum": 0.0, "never": 0, "helpful": 0})
        bucket["total"] += 1
        bucket["trust_sum"] += _number_value(row[1], 0.0)
        bucket["never"] += int(_count_value(row[2]) == 0)
        bucket["helpful"] += int(_count_value(row[3]) > 0)
    return [
        _memory_breakdown_row(
            label,
            int(bucket["total"]),
            float(bucket["trust_sum"]),
            int(bucket["never"]),
            int(bucket["helpful"]),
        )
        for label, bucket in sorted(
            grouped.items(),
            key=lambda item: (
                -int(item[1]["total"]),
                -int(item[1]["helpful"]),
                item[0],
            ),
        )[:limit]
    ]
