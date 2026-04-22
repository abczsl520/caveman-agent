"""Abstract base class for pluggable memory providers.

.. deprecated:: 0.3.0
    This module is dead code. The production MemoryManager lives in
    caveman.memory.manager. Will be removed in 0.4.0.

Ported from Hermes agent/memory_provider.py — full lifecycle hooks.

Memory providers give the agent persistent recall across sessions. One
external provider is active at a time alongside the always-on built-in
memory. The MemoryManager enforces this limit.

Lifecycle (called by MemoryManager):
  initialize()           — connect, create resources, warm up
  system_prompt_block()  — static text for the system prompt
  prefetch(query)        — background recall before each turn
  sync_turn(user, asst)  — async write after each turn
  get_tool_schemas()     — tool schemas to expose to the model
  handle_tool_call()     — dispatch a tool call
  shutdown()             — clean exit

Optional hooks (override to opt in):
  on_turn_start(turn, message, **kwargs)
  on_session_end(messages)
  on_pre_compress(messages) -> str
  on_memory_write(action, target, content)
  on_delegation(task, result, **kwargs)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'builtin', 'honcho', 'hindsight')."""

    # ── Core lifecycle (implement these) ───────────────────────────────────

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if configured, has credentials, and is ready.

        Called during agent init. Should not make network calls.
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize for a session.

        Called once at agent startup. May create resources, establish
        connections, start background threads.

        kwargs always include:
          - caveman_home (str): Active home directory path.
          - platform (str): "cli", "telegram", "discord", "cron", etc.

        kwargs may also include:
          - agent_context (str): "primary", "subagent", "cron", "flush"
          - agent_identity (str): Profile name
          - parent_session_id (str): For subagents
          - user_id (str): Platform user identifier
        """

    def system_prompt_block(self) -> str:
        """Return text to include in the system prompt.

        For STATIC provider info. Prefetched recall is injected separately.
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn.

        Called before each API call. Return formatted text to inject,
        or empty string if nothing relevant.
        """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background recall for the NEXT turn."""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist a completed turn to the backend.

        Should be non-blocking — queue for background processing.
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this provider exposes.

        Each schema follows OpenAI function calling format.
        Return empty list if context-only (no tools).
        """

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call for one of this provider's tools.

        Must return a JSON string (the tool result).
        """
        raise NotImplementedError(f"{self.name} does not handle {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # ── Optional hooks (override to opt in) ────────────────────────────────

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at the start of each turn with the user message.

        kwargs may include: remaining_tokens, model, platform, tool_count.
        """

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends (explicit exit or timeout).

        messages is the full conversation history.
        NOT called after every turn — only at actual session boundaries.
        """

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called before context compression discards old messages.

        Return text to include in compression summary prompt so the
        compressor preserves provider-extracted insights.
        """
        return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Called when the built-in memory tool writes an entry.

        action: 'add', 'replace', or 'remove'
        target: 'memory' or 'user'
        content: the entry content
        """

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """Called on the PARENT agent when a subagent completes.

        task: the delegation prompt
        result: the subagent's final response
        """

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields this provider needs for setup.

        Each field: {key, description, secret, required, default, choices, url, env_var}
        Return empty list if no config needed.
        """
        return []

    def save_config(self, values: Dict[str, Any], caveman_home: str) -> None:
        """Write non-secret config to the provider's native location.

        Called by setup after collecting user inputs.
        """


class BuiltinMemoryProvider(MemoryProvider):
    """Always-on built-in memory provider (MEMORY.md / USER.md).

    Cannot be removed. Provides the base memory tools.
    """

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._caveman_home = kwargs.get("caveman_home", "")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_search",
                "description": "Search long-term memory for relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_store",
                "description": "Store important information to long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to remember"},
                        "target": {"type": "string", "enum": ["memory", "user"], "default": "memory"},
                    },
                    "required": ["content"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        import json
        if tool_name == "memory_search":
            # Delegate to actual memory search
            try:
                from caveman.memory.manager import MemoryManager as _MM
                mm = _MM()
                results = mm.search(args.get("query", ""), limit=5)
                return json.dumps({"results": results}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e)})
        elif tool_name == "memory_store":
            try:
                from caveman.memory.manager import MemoryManager as _MM
                mm = _MM()
                mm.add(args.get("content", ""), target=args.get("target", "memory"))
                return json.dumps({"success": True})
            except Exception as e:
                return json.dumps({"error": str(e)})
        return super().handle_tool_call(tool_name, args, **kwargs)

    def system_prompt_block(self) -> str:
        return ""

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass
