"""Global test fixtures."""
import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _bypass_quality_gate():
    """Bypass quality gate in tests so existing test content works.

    Quality gate is tested explicitly in test_quality_gate.py.
    Other tests focus on their own concerns without needing realistic content.
    """
    with patch("caveman.memory.quality_gate.check_quality", return_value=None):
        with patch("caveman.memory.quality_gate.truncate_if_needed", side_effect=lambda x: x):
            yield
