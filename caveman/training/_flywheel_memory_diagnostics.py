# Internal helpers for flywheel dashboard; not public API.
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from caveman.memory.decay import (
    _DECAY_RATE_PER_DAY,
    _DECAY_START_DAYS,
    _HIGH_TRUST_SLOWDOWN,
    _HIGH_TRUST_THRESHOLD,
    _PRUNE_AGE_DAYS,
    _SOURCE_POLICY_MIN_AGE_DAYS,
    _SOURCE_POLICY_TRUST_THRESHOLD,
)
from caveman.memory.sources import IMPORT_SOURCE_PREFIX, SOURCE_POLICY_LOW_SIGNAL_IMPORTS, canonicalize_memory_source
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
    active: int | None = None,
    quarantined: int | None = None,
    eligible_for_source_policy: int | None = None,
) -> dict[str, Any]:
    """Build one memory diagnostic row with rounded rates."""
    row = {
        "label": label,
        "total": total,
        "avg_trust": round(trust_sum / total, 3) if total else 0.0,
        "never_recalled": never_recalled,
        "never_recalled_pct": round(never_recalled / total, 3) if total else 0.0,
        "helpful": helpful,
        "helpful_pct": round(helpful / total, 3) if total else 0.0,
    }
    if active is not None:
        row["active"] = active
    if quarantined is not None:
        row["quarantined"] = quarantined
    if eligible_for_source_policy is not None:
        row["eligible_for_source_policy"] = eligible_for_source_policy
    return row


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


def _source_and_governance(metadata_json: object) -> tuple[str, str, str]:
    try:
        meta = json.loads(str(metadata_json or "{}"))
    except (TypeError, json.JSONDecodeError):
        meta = {}
    source = "<missing>"
    source_identity = "<missing>"
    state = "active"
    if isinstance(meta, dict):
        raw_source = canonicalize_memory_source(meta.get("source"))
        if raw_source:
            source_identity = raw_source
            source = _memory_breakdown_label(raw_source)
        state = str(meta.get("governance_state", "active")).lower()
    return source, source_identity, state


def _memory_age_days(created_at: object, now: datetime) -> int:
    try:
        created = datetime.fromisoformat(str(created_at))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return _PRUNE_AGE_DAYS
    return (now - created).days


def _source_policy_display_trust(trust: float, age_days: int, retrieval_count: int) -> float:
    if age_days < _DECAY_START_DAYS:
        return trust
    rate = _DECAY_RATE_PER_DAY
    if trust >= _HIGH_TRUST_THRESHOLD:
        rate /= _HIGH_TRUST_SLOWDOWN
    if retrieval_count > 10:
        rate *= 0.5
    decay_amount = rate * min(age_days - _DECAY_START_DAYS, 30)
    return max(0.0, trust - decay_amount)


def _source_policy_eligible(
    source: str,
    governance_state: str,
    trust: float,
    retrieval_count: int,
    helpful_count: int,
    age_days: int,
) -> bool:
    return (
        source in SOURCE_POLICY_LOW_SIGNAL_IMPORTS
        and governance_state != "quarantined"
        and age_days >= _SOURCE_POLICY_MIN_AGE_DAYS
        and age_days < _PRUNE_AGE_DAYS
        and _source_policy_display_trust(trust, age_days, retrieval_count) <= _SOURCE_POLICY_TRUST_THRESHOLD
        and retrieval_count == 0
        and helpful_count == 0
    )


def _collect_source_rows(cur: Any) -> list[dict[str, Any]]:
    rows = cur.execute(
        "SELECT metadata_json, trust_score, COALESCE(retrieval_count, 0), "
        "COALESCE(helpful_count, 0), created_at FROM memories"
    ).fetchall()
    grouped: dict[str, dict[str, Any]] = {}
    now = datetime.now(timezone.utc)
    for row in rows:
        source_label, source_identity, governance_state = _source_and_governance(row[0])
        trust = _number_value(row[1], 0.0)
        retrieval_count = _count_value(row[2])
        helpful_count = _count_value(row[3])
        age_days = _memory_age_days(row[4], now)
        bucket = grouped.setdefault(
            source_identity,
            {
                "label": source_label,
                "identity": source_identity,
                "total": 0,
                "trust_sum": 0.0,
                "never": 0,
                "helpful": 0,
                "active": 0,
                "quarantined": 0,
                "eligible": 0,
            },
        )
        bucket["total"] += 1
        bucket["trust_sum"] += trust
        bucket["never"] += int(retrieval_count == 0)
        bucket["helpful"] += int(helpful_count > 0)
        is_quarantined = governance_state == "quarantined"
        bucket["quarantined"] += int(is_quarantined)
        bucket["active"] += int(not is_quarantined)
        bucket["eligible"] += int(
            _source_policy_eligible(source_identity, governance_state, trust, retrieval_count, helpful_count, age_days)
        )
    return [
        _memory_breakdown_row(
            str(bucket["label"]),
            int(bucket["total"]),
            float(bucket["trust_sum"]),
            int(bucket["never"]),
            int(bucket["helpful"]),
            int(bucket["active"]),
            int(bucket["quarantined"]),
            int(bucket["eligible"]),
        ) | {"identity": bucket["identity"]}
        for bucket in grouped.values()
    ]


def _collect_memory_source_breakdown(cur: Any, limit: int = 12) -> list[dict[str, Any]]:
    """Break memory health down by metadata.source, tolerating malformed metadata."""
    return [
        {k: v for k, v in row.items() if k != "identity"}
        for row in sorted(
            _collect_source_rows(cur),
            key=lambda row: (-int(row["total"]), -int(row["helpful"]), row["label"]),
        )[:limit]
    ]


def _collect_memory_source_governance(cur: Any, limit: int = 8) -> list[dict[str, Any]]:
    """Return actionable source-governance rows across all sources, not just displayed top sources."""
    actions = []
    for row in _collect_source_rows(cur):
        eligible = int(row.get("eligible_for_source_policy", 0) or 0)
        quarantined = int(row.get("quarantined", 0) or 0)
        if eligible == 0 and quarantined == 0:
            continue
        total = int(row.get("total", 0) or 0)
        active = int(row.get("active", 0) or 0)
        never_pct = float(row.get("never_recalled_pct", 0.0) or 0.0)
        helpful_pct = float(row.get("helpful_pct", 0.0) or 0.0)
        actions.append({
            "label": row["label"],
            "total": total,
            "active": active,
            "quarantined": quarantined,
            "eligible_for_source_policy": eligible,
            "noise_score": round(max(0.0, never_pct * (1.0 - helpful_pct)), 3),
            "recall_candidate_reduction_pct": round(quarantined / total, 3) if total else 0.0,
        })
    return sorted(
        actions,
        key=lambda row: (
            -int(row["eligible_for_source_policy"]),
            -int(row["quarantined"]),
            -float(row["noise_score"]),
            -int(row["total"]),
            row["label"],
        ),
    )[:limit]


def _collect_memory_source_policy_drift(cur: Any, min_rows: int = 3, limit: int = 8) -> list[dict[str, Any]]:
    """Find unmanaged import sources that look like low-signal bulk imports."""
    drift = []
    for row in _collect_source_rows(cur):
        identity = str(row.get("identity") or row.get("label") or "")
        total = int(row.get("total", 0) or 0)
        active = int(row.get("active", 0) or 0)
        never_pct = float(row.get("never_recalled_pct", 0.0) or 0.0)
        helpful_pct = float(row.get("helpful_pct", 0.0) or 0.0)
        avg_trust = float(row.get("avg_trust", 0.0) or 0.0)
        if (
            identity.startswith(IMPORT_SOURCE_PREFIX)
            and identity not in SOURCE_POLICY_LOW_SIGNAL_IMPORTS
            and total >= min_rows
            and active > 0
            and never_pct >= 0.9
            and helpful_pct == 0.0
            and avg_trust <= 0.1
        ):
            drift.append({
                "label": row["label"],
                "total": total,
                "active": active,
                "never_recalled_pct": row["never_recalled_pct"],
                "helpful_pct": row["helpful_pct"],
                "avg_trust": row["avg_trust"],
                "reason": "unmanaged_low_signal_import",
            })
    return sorted(drift, key=lambda row: (-int(row["total"]), row["label"]))[:limit]
