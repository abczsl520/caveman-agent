"""Tests for behavior_rules — single source of truth for agent conventions."""
import pytest
from caveman.agent.behavior_rules import get_rule, _RULES


class TestBehaviorRules:
    """Ensure behavior_rules stays consistent and complete."""

    def test_no_closing_format_rules_are_exposed(self):
        assert get_rule("CLOSING_FORMAT") is None
        assert get_rule("AGENT_CLOSING_FORMAT") is None

    def test_output_style_exists_and_nonempty(self):
        val = get_rule("OUTPUT_STYLE")
        assert val is not None
        assert len(val) > 50, "OUTPUT_STYLE should be substantive"

    def test_output_style_key_principles(self):
        """Lock down key principles so they aren't accidentally removed."""
        style = get_rule("OUTPUT_STYLE")
        assert "\u5ba2\u5957" in style, "must mention no pleasantries"
        assert "\u4ee3\u7801\u5757" in style or "```" in style, "must mention code blocks"
        assert "\u603b\u7ed3" in style, "must mention summary"
        assert "✅" not in style, "output style must not encourage completion emoji"

    def test_unknown_rule_returns_none(self):
        assert get_rule("NONEXISTENT_RULE") is None

    def test_rules_dict_not_empty(self):
        assert len(_RULES) >= 1, "should have output style rules"

    def test_legacy_closing_format_lives_only_in_output_validator(self):
        """Legacy markers may exist only as strip tokens, not behavior rules."""
        from caveman.agent import output_validator
        assert output_validator.CLOSING_LINE == "✅---本轮已完成---✅"
        assert get_rule("CLOSING_FORMAT") is None
