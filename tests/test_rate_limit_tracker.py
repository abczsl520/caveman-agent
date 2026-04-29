"""Regression tests for rate-limit duration parsing."""
from __future__ import annotations

import pytest

from caveman.agent.rate_limit_tracker import _parse_duration


def test_parse_duration_prefers_milliseconds_over_minutes() -> None:
    """A value like 500ms must not be parsed as 500 minutes."""
    assert _parse_duration("500ms") == pytest.approx(0.5)


def test_parse_duration_handles_compound_units() -> None:
    assert _parse_duration("1h2m3s") == pytest.approx(3723.0)


def test_parse_duration_handles_compound_milliseconds() -> None:
    assert _parse_duration("1m500ms") == pytest.approx(60.5)
