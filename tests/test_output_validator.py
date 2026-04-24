"""Tests for output_validator — closing format enforcement."""
from caveman.agent.conversation_lifecycle import ConversationState
from caveman.agent.output_validator import (
    CLOSING_LINE,
    enforce_closing_format,
    final_sentence_is_question,
    should_use_closing_marker,
)


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

    def test_question_not_closed_even_when_requested(self):
        text = "你要我继续把服务也重启吗？"
        assert enforce_closing_format(text, True) == text

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


class TestQuestionDetection:
    def test_chinese_question(self):
        assert final_sentence_is_question("需要我继续吗？") is True

    def test_english_question(self):
        assert final_sentence_is_question("Should I continue?") is True

    def test_statement(self):
        assert final_sentence_is_question("已经处理完。") is False


class TestShouldUseClosingMarker:
    def test_simple_answer_does_not_close(self):
        state = ConversationState(turn_count=1, tool_call_count=0, iteration_count=0)
        assert should_use_closing_marker(state=state, final_text="在呢，元宝～", surface="discord") is False

    def test_simple_one_tool_lookup_does_not_close(self):
        state = ConversationState(turn_count=1, tool_call_count=1, iteration_count=1)
        assert should_use_closing_marker(state=state, final_text="当前目录有 3 个文件。", surface="discord") is False

    def test_question_does_not_close_even_for_complex_work(self):
        state = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        assert should_use_closing_marker(state=state, final_text="要我现在部署吗？", surface="discord") is False

    def test_complex_work_closes(self):
        state = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        assert should_use_closing_marker(state=state, final_text="已修复并通过测试。", surface="discord") is True

    def test_medium_real_work_closes(self):
        state = ConversationState(turn_count=2, tool_call_count=3, iteration_count=1)
        assert should_use_closing_marker(state=state, final_text="已完成排查和修复。", surface="discord") is True
