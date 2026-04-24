"""Tests for output_validator — closing format enforcement."""
from caveman.agent.output_validator import enforce_closing_format, CLOSING_LINE


class TestEnforceClosingFormat:
    def test_already_correct(self):
        text = f"做完了。\n\n{CLOSING_LINE}"
        assert enforce_closing_format(text, True) == text

    def test_bare_checkmark_fixed(self):
        text = "做完了。\n\n✅"
        result = enforce_closing_format(text, True)
        assert result.endswith(CLOSING_LINE)
        assert "做完了。" in result

    def test_wrong_pattern_fixed(self):
        text = "搞定了 ✅完成✅"
        result = enforce_closing_format(text, True)
        assert CLOSING_LINE in result

    def test_missing_closing_appended(self):
        text = "全部修复完毕，测试通过。"
        result = enforce_closing_format(text, True)
        assert result.endswith(CLOSING_LINE)

    def test_no_close_when_not_needed(self):
        text = "还在跑，等一下。"
        assert enforce_closing_format(text, False) == text

    def test_empty_text(self):
        assert enforce_closing_format("", True) == ""

    def test_checkmark_in_middle_not_touched(self):
        text = "步骤 ✅ 完成\n还有下一步要做"
        result = enforce_closing_format(text, True)
        # Should append closing since no closing at end
        assert result.endswith(CLOSING_LINE)
