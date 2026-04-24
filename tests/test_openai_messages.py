"""Tests for OpenAI message format converter."""
from caveman.providers.openai_messages import convert_to_openai_messages


def test_plain_string_passthrough():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    result = convert_to_openai_messages(msgs)
    assert result == msgs


def test_system_message():
    msgs = [{"role": "system", "content": "You are helpful."}]
    result = convert_to_openai_messages(msgs)
    assert result[0] == {"role": "system", "content": "You are helpful."}


def test_assistant_with_tool_use_blocks():
    """Anthropic-style assistant message with tool_use → OpenAI tool_calls."""
    msgs = [{"role": "assistant", "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "call_1", "name": "file_read",
         "input": {"path": "/tmp/x"}},
    ]}]
    result = convert_to_openai_messages(msgs)
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "Let me check."
    assert len(result[0]["tool_calls"]) == 1
    tc = result[0]["tool_calls"][0]
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "file_read"


def test_assistant_tool_use_no_text():
    """Tool call with no text → content is None (valid for OpenAI)."""
    msgs = [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_2", "name": "bash", "input": {"cmd": "ls"}},
    ]}]
    result = convert_to_openai_messages(msgs)
    assert result[0]["content"] is None
    assert len(result[0]["tool_calls"]) == 1


def test_tool_result_blocks():
    """Anthropic tool_result blocks → OpenAI tool messages."""
    msgs = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "file contents"},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "ok"},
    ]}]
    result = convert_to_openai_messages(msgs)
    assert len(result) == 2
    assert result[0]["role"] == "tool"
    assert result[0]["tool_call_id"] == "call_1"
    assert result[0]["content"] == "file contents"
    assert result[1]["role"] == "tool"
    assert result[1]["tool_call_id"] == "call_2"


def test_null_content_becomes_empty_string():
    msgs = [{"role": "user", "content": None}]
    result = convert_to_openai_messages(msgs)
    assert result[0]["content"] == ""


def test_already_openai_format_idempotent():
    """Messages already in OpenAI format pass through."""
    msgs = [
        {"role": "assistant", "content": "thinking...", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "bash", "arguments": '{"cmd":"ls"}'}},
        ]},
        {"role": "tool", "content": "output", "tool_call_id": "c1"},
    ]
    result = convert_to_openai_messages(msgs)
    assert result[0]["tool_calls"] == msgs[0]["tool_calls"]
    assert result[1] == msgs[1]


def test_mixed_conversation():
    """Full conversation with text, tool calls, and results."""
    msgs = [
        {"role": "user", "content": "read /tmp/x"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Reading file."},
            {"type": "tool_use", "id": "c1", "name": "file_read",
             "input": {"path": "/tmp/x"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "hello world"},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "The file contains: hello world"},
        ]},
    ]
    result = convert_to_openai_messages(msgs)
    assert len(result) == 4
    assert result[0] == {"role": "user", "content": "read /tmp/x"}
    assert result[1]["tool_calls"][0]["function"]["name"] == "file_read"
    assert result[2] == {"role": "tool", "content": "hello world", "tool_call_id": "c1"}
    assert result[3] == {"role": "assistant", "content": "The file contains: hello world"}
