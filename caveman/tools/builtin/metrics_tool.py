"""Metrics tool — query agent performance and flywheel stats.

Lets Caveman (and users) inspect token usage, timing percentiles,
flywheel history, and session costs. Essential for self-optimization.
"""
from __future__ import annotations

from caveman.tools.registry import tool


@tool(
    name="metrics",
    description=(
        "Query agent performance metrics: token usage, timing percentiles, "
        "flywheel stats, session costs. Use to understand efficiency and "
        "identify optimization targets."
    ),
    params={
        "category": {
            "type": "string",
            "description": (
                "What to query: 'agent' (timings/counters), 'flywheel' (round history), "
                "'provider' (token usage/costs), 'all' (everything)"
            ),
        },
    },
    required=[],
)
async def metrics_query(category: str = "all", _context: dict | None = None) -> dict:
    """Query performance metrics."""
    ctx = _context or {}
    result = {}

    if category in ("agent", "all"):
        m = ctx.get("metrics")
        if m and hasattr(m, "summary"):
            result["agent"] = m.summary()
        else:
            result["agent"] = {"note": "No metrics collector in context"}

    if category in ("flywheel", "all"):
        try:
            from caveman.cli.flywheel import FlywheelStats
            result["flywheel"] = FlywheelStats().summary()
        except Exception as e:
            result["flywheel"] = {"error": str(e)}

    if category in ("provider", "all"):
        # Try to get provider stats from the loop
        # The provider tracks _total_input_tokens, _total_output_tokens
        result["provider"] = {"note": "Use /status or check session meta for token stats"}

    return {"ok": True, "metrics": result}
