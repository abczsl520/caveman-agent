"""Regression coverage for import/session parsing type boundaries."""
from __future__ import annotations

import json
from pathlib import Path

from caveman.import_.session_parser import parse_session
from caveman.import_.workspace_adapter import adapt_workspace_content, validate_adapted_content


def test_parse_session_normalizes_malformed_tool_call_fields(tmp_path: Path) -> None:
    session = tmp_path / "session-topic-123.jsonl"
    rows = [
        {"type": "session", "id": 123, "timestamp": "2026-01-01T00:00:00Z", "cwd": "/tmp"},
        {"type": "message", "message": {"role": "user", "content": "please run something"}},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will run it."},
                    {"type": "toolCall", "name": None, "arguments": "not-a-dict", "id": 42},
                ],
            },
        },
    ]
    session.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    parsed = parse_session(session)

    assert parsed is not None
    assert parsed.metadata.session_id == "123"
    tool_call = parsed.turns[0].tool_calls[0]
    assert tool_call.name == ""
    assert tool_call.arguments == {}
    assert tool_call.call_id == "42"


def test_parse_session_skips_wrong_shaped_json_lines(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    rows: list[object] = [
        [],
        {"type": "message", "message": "bad"},
        {"type": "message", "message": {"role": "user", "content": "please parse this"}},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": 123},
                    {"type": "thinking", "thinking": None},
                    {"type": "text", "text": "safe text"},
                ],
            },
        },
    ]
    session.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    parsed = parse_session(session)

    assert parsed is not None
    assert parsed.turns[0].assistant_texts == ["safe text"]
    assert parsed.turns[0].thinking_text == ""


def test_workspace_adapter_allows_unavailable_tool_mapping() -> None:
    adapted = adapt_workspace_content("AGENTS.md", "Use `tts` after the message tool")

    assert "`tts` (not available)" in adapted
    assert "progress tool" in adapted
    assert validate_adapted_content("AGENTS.md", adapted) == []
