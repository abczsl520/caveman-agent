from __future__ import annotations

from typing import Any

from caveman.operator_output import operator_literal


def _operator_literal(value: object, max_length: int = 160) -> str:
    """Return a repr-style literal safe for terminal/operator reports."""
    return operator_literal(value, max_length=max_length)


def _format_restorable_quarantine(by_source: dict[str, int], by_reason: dict[str, int]) -> list[str]:
    lines = []
    if by_source:
        lines.append(
            "   Restorable quarantine: "
            + ", ".join(f"{_operator_literal(source, max_length=120)}={count}" for source, count in by_source.items())
        )
    if by_reason:
        lines.append(
            "   Restorable quarantine reasons: "
            + ", ".join(f"{_operator_literal(reason, max_length=120)}={count}" for reason, count in by_reason.items())
        )
    return lines


def _format_source_policy_drift(mem: dict[str, Any]) -> list[str]:
    source_policy_drift = mem.get("source_policy_drift", [])
    if not source_policy_drift:
        return []
    lines = ["   Source policy drift:"]
    lines.extend(
        f"      label={_operator_literal(row.get('label', '<missing>'))}: unmanaged low-signal import source "
        f"(n={row.get('total', 0)}, never={float(row.get('never_recalled_pct', 0.0)):.0%}, "
        f"helpful={float(row.get('helpful_pct', 0.0)):.0%}, "
        f"candidate={_operator_literal(row.get('candidate_policy_entry', row.get('label', '<missing>')))})"
        for row in source_policy_drift[:5]
    )
    return lines
