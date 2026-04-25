"""Tests for output_validator — terminal marker suppression."""
from caveman.agent.conversation_lifecycle import ConversationState
from caveman.agent.output_validator import (
    CLOSING_LINE,
    enforce_closing_format,
    final_sentence_is_question,
    final_text_looks_truncated,
    should_use_closing_marker,
)


class TestEnforceClosingFormat:
    def test_canonical_marker_is_stripped_even_when_requested(self):
        text = f"做完了。\n\n{CLOSING_LINE}"
        assert enforce_closing_format(text, True) == "做完了。"

    def test_bare_checkmark_stripped_even_when_requested(self):
        text = "做完了。\n\n✅"
        result = enforce_closing_format(text, True)
        assert result == "做完了。"
        assert CLOSING_LINE not in result

    def test_wrong_pattern_stripped(self):
        text = "搞定了 ✅完成✅"
        result = enforce_closing_format(text, True)
        assert CLOSING_LINE not in result
        assert "✅" not in result

    def test_missing_closing_not_appended_without_explicit_attempt(self):
        text = "全部修复完毕，测试通过。"
        result = enforce_closing_format(text, True)
        assert result == text
        assert CLOSING_LINE not in result

    def test_question_not_closed_even_when_requested(self):
        text = "你要我继续把服务也重启吗？"
        assert enforce_closing_format(text, True) == text

    def test_question_with_accidental_closing_is_stripped(self):
        text = f"你要我继续把服务也重启吗？\n\n{CLOSING_LINE}"
        assert enforce_closing_format(text, True) == "你要我继续把服务也重启吗？"

    def test_simple_reply_with_accidental_closing_is_stripped(self):
        text = f"在呢，元宝。\n\n{CLOSING_LINE}"
        assert enforce_closing_format(text, False) == "在呢，元宝。"

    def test_simple_reply_with_bare_checkmark_is_stripped(self):
        text = "在呢，元宝。\n\n✅"
        assert enforce_closing_format(text, False) == "在呢，元宝。"

    def test_empty_text(self):
        assert enforce_closing_format("", True) == ""

    def test_checkmark_in_middle_not_touched(self):
        text = "步骤 ✅ 完成\n还有下一步要做"
        result = enforce_closing_format(text, True)
        # No explicit trailing closing attempt, so the validator must not invent
        # a terminal completion signal from ordinary content.
        assert result == text
        assert CLOSING_LINE not in result


class TestTruncatedFinalDetection:
    def test_observed_prd_half_line_is_truncated(self):
        text = """
元宝，我仔细对照了，不是只扫文件名，确实读了对应实现、CLI、测试和 PRD 证据链。

📌 已更新 `docs/PRD.md`

这次主要把几项“PRD 还写未完成/部分完成，但代码其实已经落地”的内容标记清楚了：

- `#21 last_accessed`
  - 已有独立列 `last_accessed TEXT`
  - 已有 v2 migration
  - 已有旧 metadata 回填
  - PR
"""
        assert final_text_looks_truncated(text) is True

    def test_normal_chinese_summary_without_punctuation_is_not_truncated(self):
        text = "这次已完成 PRD 对照审计，更新了状态，并跑过相关测试"
        assert final_text_looks_truncated(text) is False

    def test_unclosed_code_fence_is_truncated(self):
        text = "这里是结果：\n\n```python\nprint('hello')\n"
        assert final_text_looks_truncated(text) is True


class TestQuestionDetection:
    def test_chinese_question(self):
        assert final_sentence_is_question("需要我继续吗？") is True

    def test_english_question(self):
        assert final_sentence_is_question("Should I continue?") is True

    def test_chinese_question_with_closing_marker(self):
        text = f"要我继续吗？\n\n{CLOSING_LINE}"
        assert final_sentence_is_question(text) is True

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

    def test_complex_work_does_not_close(self):
        state = ConversationState(turn_count=8, tool_call_count=30, iteration_count=5)
        assert should_use_closing_marker(state=state, final_text="已修复并通过测试。", surface="discord") is False

    def test_medium_real_work_does_not_close(self):
        state = ConversationState(turn_count=2, tool_call_count=3, iteration_count=1)
        assert should_use_closing_marker(state=state, final_text="已完成排查和修复。", surface="discord") is False
