"""Tests for behavior_rules — single source of truth for agent conventions."""
import pytest
from caveman.agent.behavior_rules import get_rule, _RULES


class TestBehaviorRules:
    """Ensure behavior_rules stays consistent and complete."""

    def test_closing_format_exists(self):
        val = get_rule("CLOSING_FORMAT")
        assert val is not None
        assert "\u2705" in val

    def test_agent_closing_format_exists(self):
        val = get_rule("AGENT_CLOSING_FORMAT")
        assert val is not None
        assert val != get_rule("CLOSING_FORMAT"), "agent should have distinct closing"

    def test_output_style_exists_and_nonempty(self):
        val = get_rule("OUTPUT_STYLE")
        assert val is not None
        assert len(val) > 50, "OUTPUT_STYLE should be substantive"

    def test_output_style_key_principles(self):
        """Lock down key principles so they aren't accidentally removed."""
        style = get_rule("OUTPUT_STYLE")
        assert "\u5ba2\u5957" in style, "must mention no pleasantries"
        assert "\u4ee3\u7801\u5757" in style or "```" in style, "must mention code blocks"
        assert "\u603b\u7ed3" in style, "must mention brief summary"

    def test_unknown_rule_returns_none(self):
        assert get_rule("NONEXISTENT_RULE") is None

    def test_rules_dict_not_empty(self):
        assert len(_RULES) >= 3, "should have at least 3 rules"

    def test_closing_format_used_by_output_validator(self):
        """output_validator must read from behavior_rules, not hardcode."""
        import inspect
        from caveman.agent import output_validator
        src = inspect.getsource(output_validator)
        assert "get_rule" in src or "behavior_rules" in src, \
            "output_validator must import from behavior_rules"

    def test_closing_format_used_by_lifecycle(self):
        """conversation_lifecycle must reference behavior_rules."""
        import inspect
        from caveman.agent import conversation_lifecycle
        src = inspect.getsource(conversation_lifecycle)
        assert "behavior_rules" in src or "get_rule" in src or "CLOSING_FORMAT" in src, \
            "conversation_lifecycle must reference closing format"
