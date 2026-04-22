"""OpenClaw Session JSONL Parser — structured extraction of conversation data.

Parses the JSONL format used by OpenClaw's session storage:
  - Each line is a JSON object with a `type` field
  - Types: session, message, model_change, thinking_level_change, custom
  - Messages have role: user | assistant | toolResult
  - Assistant content blocks: text, toolCall, thinking
  - toolCall has: name, arguments, id
  - toolResult has: text content

This parser extracts structured conversation turns from raw JSONL,
preserving the semantic relationships between messages.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ToolCall",
    "ConversationTurn",
    "SessionMetadata",
    "ParsedSession",
    "parse_session",
    "scan_sessions",
]


logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A single tool invocation within an assistant turn."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    result_text: str = ""

    @property
    def is_message_send(self) -> bool:
        """Was this a message sent to the user (Discord/Telegram)?"""
        return (
            self.name == "message"
            and self.arguments.get("action") == "send"
        )

    @property
    def sent_message(self) -> str:
        """The text sent to the user, if this is a message send."""
        if self.is_message_send:
            return self.arguments.get("message", "")
        return ""

    @property
    def is_read_op(self) -> bool:
        """Is this a file read or code read operation?"""
        return self.name in ("read", "readCode", "readFile")

    @property
    def is_exec(self) -> bool:
        """Is this a command execution?"""
        return self.name in ("exec", "bash")

    @property
    def is_write_op(self) -> bool:
        """Is this a file write/edit operation?"""
        return self.name in ("write", "edit", "writeFile", "editFile")

    @property
    def is_search(self) -> bool:
        """Is this a search operation (web or memory)?"""
        return self.name in (
            "web_search", "web_fetch", "memory_search",
            "memory_get", "search",
        )


@dataclass
class ConversationTurn:
    """A single turn in the conversation: user prompt + assistant response."""
    turn_index: int
    timestamp: str = ""
    user_text: str = ""
    assistant_texts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking_text: str = ""

    @property
    def assistant_prose(self) -> str:
        """All non-tool assistant text joined."""
        return "\n\n".join(t for t in self.assistant_texts if t.strip())

    @property
    def messages_sent(self) -> list[str]:
        """All messages sent to the user via message tool."""
        return [
            tc.sent_message for tc in self.tool_calls
            if tc.is_message_send and tc.sent_message
        ]

    @property
    def total_text_length(self) -> int:
        """Total character count of meaningful text in this turn."""
        return (
            len(self.user_text)
            + sum(len(t) for t in self.assistant_texts)
            + sum(len(tc.sent_message) for tc in self.tool_calls if tc.is_message_send)
        )

    @property
    def has_substance(self) -> bool:
        """Does this turn contain meaningful content worth extracting?"""
        # A turn has substance if:
        # 1. Assistant wrote significant prose (not just "OK" or tool calls)
        # 2. Or sent substantial messages to the user
        min_chars = 100
        prose_len = len(self.assistant_prose)
        msg_len = sum(len(m) for m in self.messages_sent)
        return (prose_len >= min_chars) or (msg_len >= min_chars)


@dataclass
class SessionMetadata:
    """Metadata extracted from the session header."""
    session_id: str = ""
    timestamp: str = ""
    cwd: str = ""
    provider: str = ""
    model_id: str = ""
    topic_id: str = ""  # Discord topic/thread ID if present

    @property
    def date(self) -> str:
        """Extract date string (YYYY-MM-DD) from timestamp."""
        if self.timestamp:
            try:
                return self.timestamp[:10]
            except (ValueError, IndexError):
                pass  # intentional: ValueError/IndexError suppressed
        return ""


@dataclass
class ParsedSession:
    """A fully parsed session with metadata and conversation turns."""
    source_path: Path
    metadata: SessionMetadata
    turns: list[ConversationTurn] = field(default_factory=list)

    @property
    def total_user_messages(self) -> int:
        return sum(1 for t in self.turns if t.user_text)

    @property
    def total_assistant_texts(self) -> int:
        return sum(len(t.assistant_texts) for t in self.turns)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)

    @property
    def substantive_turns(self) -> list[ConversationTurn]:
        """Turns that contain meaningful content."""
        return [t for t in self.turns if t.has_substance]

    @property
    def topic_hint(self) -> str:
        """Best guess at what this session is about.

        Uses the first user message or thread starter as a hint.
        """
        for turn in self.turns:
            if turn.user_text:
                text = turn.user_text.strip()
                # Strip OpenClaw metadata prefix
                for prefix in (
                    "[Thread starter - for context]",
                    "Conversation info (untrusted metadata):",
                ):
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                # Strip JSON metadata blocks
                if text.startswith("```json"):
                    end = text.find("```", 7)
                    if end > 0:
                        text = text[end + 3:].strip()
                # Return first meaningful line
                for line in text.split("\n"):
                    line = line.strip()
                    if line and len(line) > 5:
                        return line[:200]
        return ""

    @property
    def summary_stats(self) -> str:
        """Human-readable summary of session contents."""
        sub = len(self.substantive_turns)
        return (
            f"{self.total_user_messages} user msgs, "
            f"{self.total_assistant_texts} assistant texts, "
            f"{self.total_tool_calls} tool calls, "
            f"{sub} substantive turns"
        )


def parse_session(path: Path) -> ParsedSession | None:
    """Parse an OpenClaw session JSONL file into structured data.

    Returns None if the file is unreadable or empty.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        logger.warning("Failed to read session %s: %s", path, e)
        return None

    if not lines:
        return None

    metadata = SessionMetadata()
    # Pending tool results keyed by tool call ID
    pending_results: dict[str, str] = {}
    turns: list[ConversationTurn] = []
    current_turn: ConversationTurn | None = None
    turn_index = 0

    # Extract topic ID from filename if present
    fname = path.stem
    if "-topic-" in fname:
        parts = fname.split("-topic-")
        if len(parts) == 2:
            metadata.topic_id = parts[1].split(".")[0]

    for line_text in lines:
        line_text = line_text.strip()
        if not line_text:
            continue
        try:
            obj = json.loads(line_text)
        except json.JSONDecodeError:
            continue

        obj_type = obj.get("type", "")

        # --- Session header ---
        if obj_type == "session":
            metadata.session_id = obj.get("id", "")
            metadata.timestamp = obj.get("timestamp", "")
            metadata.cwd = obj.get("cwd", "")
            continue

        # --- Model info ---
        if obj_type == "model_change":
            metadata.provider = obj.get("provider", "")
            metadata.model_id = obj.get("modelId", "")
            continue

        # --- Messages ---
        if obj_type != "message":
            continue

        msg = obj.get("message", {})
        role = msg.get("role", "")
        timestamp = obj.get("timestamp", "")
        content = msg.get("content", [])

        if role == "user":
            # Start a new turn
            turn_index += 1
            current_turn = ConversationTurn(
                turn_index=turn_index,
                timestamp=timestamp,
            )
            turns.append(current_turn)
            # Extract user text
            current_turn.user_text = _extract_text_from_content(content)

        elif role == "assistant" and current_turn is not None:
            # Process assistant content blocks
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type", "")

                    if block_type == "text":
                        text = block.get("text", "").strip()
                        if text:
                            current_turn.assistant_texts.append(text)

                    elif block_type == "thinking":
                        thinking = block.get("thinking", "").strip()
                        if thinking:
                            current_turn.thinking_text += thinking + "\n"

                    elif block_type == "toolCall":
                        tc = ToolCall(
                            name=block.get("name", block.get("toolName", "")),
                            arguments=block.get("arguments", block.get("input", {})),
                            call_id=block.get("id", ""),
                        )
                        current_turn.tool_calls.append(tc)

        elif role == "toolResult" and current_turn is not None:
            # Match tool results back to their calls
            result_text = _extract_text_from_content(content)
            # Try to match by parent message ID
            parent_id = obj.get("parentId", "")
            # Store for potential matching
            if result_text:
                # Find the most recent unmatched tool call
                for tc in reversed(current_turn.tool_calls):
                    if not tc.result_text:
                        tc.result_text = result_text
                        break

    # Build the parsed session
    session = ParsedSession(
        source_path=path,
        metadata=metadata,
        turns=turns,
    )
    return session


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from a message content field.

    Content can be:
    - A string
    - A list of content blocks [{"type": "text", "text": "..."}]
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return ""


def scan_sessions(sessions_dir: Path) -> list[Path]:
    """Find all active session JSONL files (exclude deleted/reset)."""
    if not sessions_dir.is_dir():
        return []

    results = []
    for f in sorted(sessions_dir.iterdir()):
        if not f.name.endswith(".jsonl"):
            continue
        # Skip deleted and reset files
        if ".deleted." in f.name or ".reset." in f.name:
            continue
        results.append(f)
    return results
