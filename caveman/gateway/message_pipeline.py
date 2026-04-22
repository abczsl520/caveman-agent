"""Inbound message pipeline — the heart of Caveman's message processing.

Inspired by OpenClaw's auto-reply architecture (45K LOC), distilled to essentials.

Pipeline stages:
1. Normalize — clean text, extract metadata, sanitize
2. Dedupe — prevent duplicate message processing
3. Preprocess — link understanding, media detection, hooks
4. Route — command detection, group activation, directive parsing
5. Execute — agent run or command handler
6. Deliver — format reply, chunk for platform, send

Each stage is a pure function taking a MessageContext and returning
a modified context or a terminal result. Stages can be composed,
reordered, or skipped via config.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

__all__ = [
    "MessageAction",
    "MessageContext",
    "normalize_message",
    "DedupeCache",
    "dedupe_message",
    "preprocess_message",
    "route_message",
    "execute_message",
    "prepare_delivery",
    "process_inbound",
]


logger = logging.getLogger(__name__)


class MessageAction(str, Enum):
    """What to do with a message after routing."""
    AGENT_RUN = "agent_run"       # Send to agent loop
    COMMAND = "command"            # Handle as slash command
    SILENT = "silent"             # No response (heartbeat, etc.)
    QUEUED = "queued"             # Queued for later processing
    REJECTED = "rejected"         # Rejected (dedupe, rate limit, etc.)
    ERROR = "error"               # Processing error


@dataclass
class MessageContext:
    """Unified message context flowing through the pipeline."""
    # Identity
    message_id: str = ""
    session_id: str = ""
    platform: str = ""
    chat_id: str = ""
    thread_id: str = ""

    # Sender
    sender_id: str = ""
    sender_name: str = ""
    sender_label: str = ""

    # Content
    body: str = ""
    raw_body: str = ""
    media_paths: list[str] = field(default_factory=list)
    media_urls: list[str] = field(default_factory=list)

    # Chat context
    chat_type: str = "dm"  # dm, group, channel
    is_group: bool = False
    is_mention: bool = False
    is_reply_to_bot: bool = False
    thread_starter: str = ""

    # Routing
    action: MessageAction = MessageAction.AGENT_RUN
    command_name: str = ""
    command_args: str = ""
    directives: dict[str, Any] = field(default_factory=dict)

    # Pipeline metadata
    received_at: float = 0.0
    processed_at: float = 0.0
    link_context: str = ""
    normalized: bool = False
    deduped: bool = False

    # Result
    reply_text: str = ""
    reply_media: list[str] = field(default_factory=list)
    error: str = ""


# ── Stage 1: Normalize ──

def normalize_message(ctx: MessageContext) -> MessageContext:
    """Clean and normalize inbound message text."""
    if ctx.normalized:
        return ctx

    # Normalize newlines
    ctx.body = ctx.body.replace("\r\n", "\n").replace("\r", "\n")
    ctx.raw_body = ctx.body

    # Strip system tags that could be injected
    ctx.body = _sanitize_system_tags(ctx.body)

    # Normalize whitespace (but preserve intentional formatting)
    ctx.body = re.sub(r"\n{4,}", "\n\n\n", ctx.body)

    # Set defaults
    if not ctx.received_at:
        ctx.received_at = time.time()
    if not ctx.sender_label:
        ctx.sender_label = ctx.sender_name or ctx.sender_id or "unknown"

    ctx.is_group = ctx.chat_type in ("group", "channel", "supergroup")
    ctx.normalized = True
    return ctx


def _sanitize_system_tags(text: str) -> str:
    """Remove injected system-like tags from user text."""
    # Remove anything that looks like [system:...] or [[...]]
    text = re.sub(r"\[system:[^\]]*\]", "", text)
    text = re.sub(r"\[\[(?:reply_to|system|admin)[^\]]*\]\]", "", text)
    return text.strip()


# ── Stage 2: Dedupe ──

class DedupeCache:
    """Time-based message deduplication."""

    def __init__(self, ttl: float = 1200, max_size: int = 5000) -> None:
        self._cache: dict[str, float] = {}
        self._ttl = ttl
        self._max_size = max_size

    def is_duplicate(self, ctx: MessageContext) -> bool:
        """Check if this message was already processed."""
        key = self._build_key(ctx)
        if not key:
            return False

        self._evict_expired()

        if key in self._cache:
            return True

        self._cache[key] = time.time()
        return False

    def _build_key(self, ctx: MessageContext) -> str:
        """Build dedup key from message identity."""
        if not ctx.message_id or not ctx.platform:
            return ""
        parts = [ctx.platform, ctx.message_id]
        if ctx.session_id:
            parts.append(ctx.session_id)
        return ":".join(parts)

    def _evict_expired(self) -> None:
        now = time.time()
        if len(self._cache) > self._max_size:
            # Remove oldest half
            sorted_keys = sorted(self._cache, key=self._cache.get)
            for k in sorted_keys[:len(sorted_keys) // 2]:
                del self._cache[k]
        # Remove expired
        expired = [k for k, t in self._cache.items() if now - t > self._ttl]
        for k in expired:
            del self._cache[k]


_global_dedupe = DedupeCache()


def dedupe_message(ctx: MessageContext, cache: DedupeCache | None = None) -> MessageContext:
    """Check for duplicate messages."""
    c = cache or _global_dedupe
    if c.is_duplicate(ctx):
        ctx.action = MessageAction.REJECTED
        ctx.error = "duplicate message"
        logger.debug("Dedupe: rejected %s", ctx.message_id)
    ctx.deduped = True
    return ctx


# ── Stage 3: Preprocess ──

async def preprocess_message(ctx: MessageContext) -> MessageContext:
    """Run preprocessing hooks: link understanding, media detection."""
    # Link understanding
    if ctx.body and "http" in ctx.body:
        try:
            from caveman.gateway.link_understanding import understand_links
            urls, link_ctx = await understand_links(ctx.body, max_links=2, timeout=10)
            if link_ctx:
                ctx.link_context = link_ctx
        except Exception as e:
            logger.debug("Link understanding failed: %s", e)

    return ctx


# ── Stage 4: Route ──

# Command prefix patterns
_COMMAND_RE = re.compile(r"^/(\w+)(?:\s+(.*))?$", re.S)


def route_message(
    ctx: MessageContext,
    bot_name: str = "",
    group_activation: str = "mention",  # "always", "mention", "reply", "never"
) -> MessageContext:
    """Determine what to do with the message."""
    if ctx.action == MessageAction.REJECTED:
        return ctx

    body = ctx.body.strip()

    # Check for slash commands
    cmd_match = _COMMAND_RE.match(body)
    if cmd_match:
        ctx.action = MessageAction.COMMAND
        ctx.command_name = cmd_match.group(1).lower()
        ctx.command_args = (cmd_match.group(2) or "").strip()
        return ctx

    # Group activation logic
    if ctx.is_group:
        should_respond = _should_respond_in_group(ctx, bot_name, group_activation)
        if not should_respond:
            ctx.action = MessageAction.SILENT
            return ctx

    # Parse inline directives (e.g., @model:opus, @verbose)
    ctx.directives = _parse_directives(body)
    if ctx.directives:
        ctx.body = _strip_directives(body)

    # Default: send to agent
    ctx.action = MessageAction.AGENT_RUN
    return ctx


def _should_respond_in_group(
    ctx: MessageContext,
    bot_name: str,
    activation: str,
) -> bool:
    """Determine if bot should respond in a group chat."""
    if activation == "always":
        return True
    if activation == "never":
        return False

    # Check mention
    if ctx.is_mention:
        return True

    # Check reply to bot
    if ctx.is_reply_to_bot:
        return True

    # Check if bot name is mentioned in text
    if bot_name and bot_name.lower() in ctx.body.lower():
        return True

    # Check if it's a thread where bot is participating
    if ctx.thread_id:
        return True  # In threads, always respond

    return False


def _parse_directives(text: str) -> dict[str, Any]:
    """Parse inline directives like @model:opus @verbose @think."""
    directives: dict[str, Any] = {}
    for match in re.finditer(r"@(\w+)(?::(\S+))?", text):
        key = match.group(1).lower()
        value = match.group(2) or True
        if key in ("model", "verbose", "think", "reasoning", "elevated", "queue"):
            directives[key] = value
    return directives


def _strip_directives(text: str) -> str:
    """Remove parsed directives from message text."""
    return re.sub(r"@(?:model|verbose|think|reasoning|elevated|queue)(?::\S+)?", "", text).strip()


# ── Stage 5: Execute ──

async def execute_message(
    ctx: MessageContext,
    agent_fn: Callable[[MessageContext], Awaitable[str]] | None = None,
    command_fn: Callable[[str, str, MessageContext], Awaitable[str]] | None = None,
) -> MessageContext:
    """Execute the routed action."""
    if ctx.action == MessageAction.REJECTED or ctx.action == MessageAction.SILENT:
        return ctx

    try:
        if ctx.action == MessageAction.COMMAND and command_fn:
            ctx.reply_text = await command_fn(ctx.command_name, ctx.command_args, ctx)
        elif ctx.action == MessageAction.AGENT_RUN and agent_fn:
            ctx.reply_text = await agent_fn(ctx)
        else:
            ctx.reply_text = ""
            ctx.action = MessageAction.ERROR
            ctx.error = "no handler for action"
    except Exception as e:
        ctx.action = MessageAction.ERROR
        ctx.error = str(e)[:500]
        logger.error("Execute error: %s", e)

    ctx.processed_at = time.time()
    return ctx


# ── Stage 6: Deliver ──

def prepare_delivery(
    ctx: MessageContext,
    max_length: int = 2000,
) -> list[str]:
    """Prepare reply for delivery: chunk, format, etc."""
    if not ctx.reply_text:
        return []

    from caveman.gateway.display_config import split_message
    return split_message(ctx.reply_text, max_length)


# ── Full Pipeline ──

async def process_inbound(
    ctx: MessageContext,
    agent_fn: Callable[[MessageContext], Awaitable[str]] | None = None,
    command_fn: Callable[[str, str, MessageContext], Awaitable[str]] | None = None,
    bot_name: str = "",
    group_activation: str = "mention",
    dedupe_cache: DedupeCache | None = None,
) -> MessageContext:
    """Run the full inbound message pipeline.

    This is the main entry point for processing incoming messages.
    """
    # Stage 1: Normalize
    ctx = normalize_message(ctx)

    # Stage 2: Dedupe
    ctx = dedupe_message(ctx, cache=dedupe_cache)
    if ctx.action == MessageAction.REJECTED:
        return ctx

    # Stage 3: Preprocess
    ctx = await preprocess_message(ctx)

    # Stage 4: Route
    ctx = route_message(ctx, bot_name=bot_name, group_activation=group_activation)
    if ctx.action == MessageAction.SILENT:
        return ctx

    # Stage 5: Execute
    ctx = await execute_message(ctx, agent_fn=agent_fn, command_fn=command_fn)

    return ctx
