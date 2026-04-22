"""Dispatch from Config — configuration-driven message routing.

Extracted from OpenClaw dispatch-from-config.ts (1046 lines).
Routes incoming messages through hooks, directives, and model dispatch.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("caveman.gateway.dispatch")


@dataclass
class SendPolicy:
    """Controls whether and how replies are sent."""
    allow: bool = True
    reason: str = ""
    suppress_tools: bool = False
    suppress_typing: bool = False


@dataclass
class DispatchContext:
    """Full context for message dispatch."""
    session_key: str = ""
    sender_id: str = ""
    channel_id: str = ""
    chat_type: str = "dm"  # dm | group | thread
    is_group: bool = False
    is_forum: bool = False
    was_mentioned: bool = False
    command_text: str = ""
    body: str = ""
    provider: str = ""
    model: str = ""
    surface: str = ""
    inbound_audio: bool = False
    tts_auto: bool = False


@dataclass
class DispatchResult:
    """Result of message dispatch."""
    ok: bool = True
    reply_text: str = ""
    reply_queued: bool = False
    phase: str = "completed"
    reason: str = ""
    tool_calls: int = 0
    tokens_used: int = 0
    duration_ms: float = 0


class HookRunner:
    """Runs before_dispatch and reply_dispatch hooks."""

    def __init__(self):
        self._before_hooks: List[Callable] = []
        self._reply_hooks: List[Callable] = []

    def add_before_hook(self, hook: Callable) -> None:
        self._before_hooks.append(hook)

    def add_reply_hook(self, hook: Callable) -> None:
        self._reply_hooks.append(hook)

    def has_hooks(self, phase: str) -> bool:
        if phase == "before_dispatch":
            return bool(self._before_hooks)
        if phase == "reply_dispatch":
            return bool(self._reply_hooks)
        return False

    async def run_before_dispatch(self, ctx: DispatchContext) -> Optional[Dict]:
        """Run before_dispatch hooks. Returns {handled: True, text: ...} to short-circuit."""
        for hook in self._before_hooks:
            try:
                result = hook(ctx) if not asyncio.iscoroutinefunction(hook) else await hook(ctx)
                if result and result.get("handled"):
                    return result
            except Exception as e:
                logger.warning("before_dispatch hook failed: %s", e)
        return None

    async def run_reply_dispatch(self, ctx: DispatchContext) -> Optional[Dict]:
        """Run reply_dispatch hooks."""
        for hook in self._reply_hooks:
            try:
                result = hook(ctx) if not asyncio.iscoroutinefunction(hook) else await hook(ctx)
                if result and result.get("handled"):
                    return result
            except Exception as e:
                logger.warning("reply_dispatch hook failed: %s", e)
        return None


def resolve_send_policy(
    config: Dict[str, Any],
    session_key: str = "",
    channel: str = "",
    chat_type: str = "dm",
) -> SendPolicy:
    """Resolve send policy from config."""
    policies = config.get("send_policies", {})

    # Check channel-specific policy
    channel_policy = policies.get(f"channel:{channel}")
    if channel_policy == "deny":
        return SendPolicy(allow=False, reason="channel_denied")

    # Check chat type policy
    type_policy = policies.get(f"type:{chat_type}")
    if type_policy == "deny":
        return SendPolicy(allow=False, reason="chat_type_denied")

    # Check session-specific
    session_policy = policies.get(f"session:{session_key}")
    if session_policy == "deny":
        return SendPolicy(allow=False, reason="session_denied")

    return SendPolicy(allow=True)


class MessageDispatcher:
    """Dispatches messages through the full pipeline.

    Pipeline: preflight → hooks → directives → send_policy → agent → reply
    """

    def __init__(
        self,
        config: Dict[str, Any],
        hook_runner: Optional[HookRunner] = None,
        agent_fn: Optional[Callable] = None,
        send_fn: Optional[Callable] = None,
    ):
        self._config = config
        self._hooks = hook_runner or HookRunner()
        self._agent_fn = agent_fn
        self._send_fn = send_fn
        self._queued_replies: List[Dict] = []

    async def dispatch(self, ctx: DispatchContext) -> DispatchResult:
        """Full dispatch pipeline."""
        start = time.monotonic()
        result = DispatchResult()

        # 1. Send policy check
        policy = resolve_send_policy(
            self._config, ctx.session_key, ctx.channel_id, ctx.chat_type,
        )
        if not policy.allow:
            result.reason = policy.reason
            result.ok = False
            return result

        # 2. Before dispatch hooks
        if self._hooks.has_hooks("before_dispatch"):
            hook_result = await self._hooks.run_before_dispatch(ctx)
            if hook_result and hook_result.get("handled"):
                result.reply_text = hook_result.get("text", "")
                result.reason = "before_dispatch_handled"
                if result.reply_text and self._send_fn:
                    await self._send_fn(ctx.channel_id, result.reply_text)
                    result.reply_queued = True
                result.duration_ms = (time.monotonic() - start) * 1000
                return result

        # 3. Reply dispatch hooks
        if self._hooks.has_hooks("reply_dispatch"):
            hook_result = await self._hooks.run_reply_dispatch(ctx)
            if hook_result and hook_result.get("handled"):
                result.reason = "reply_dispatch_handled"
                result.reply_queued = hook_result.get("queued", False)
                result.duration_ms = (time.monotonic() - start) * 1000
                return result

        # 4. Agent dispatch
        if self._agent_fn:
            try:
                agent_result = await self._agent_fn(ctx)
                result.reply_text = agent_result.get("text", "")
                result.tool_calls = agent_result.get("tool_calls", 0)
                result.tokens_used = agent_result.get("tokens", 0)

                if result.reply_text and self._send_fn:
                    await self._send_fn(ctx.channel_id, result.reply_text)
                    result.reply_queued = True

            except Exception as e:
                logger.error("Agent dispatch failed: %s", e)
                result.ok = False
                result.reason = f"agent_error: {e}"

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

from caveman.gateway.dispatch_depth import (  # noqa: F401,E402  # depth wiring
    StreamChunk,
    BlockReplyConfig,
    TTSConfig,
    StreamingDispatcher,
)

__all__ = [
    "SendPolicy",
    "DispatchContext",
    "DispatchResult",
    "HookRunner",
    "resolve_send_policy",
    "MessageDispatcher",
    "StreamChunk",
    "BlockReplyConfig",
    "TTSConfig",
    "StreamingDispatcher",
]

