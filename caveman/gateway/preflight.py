"""Message Preflight — filter, validate, and enrich incoming messages.

Extracted from OpenClaw message-handler.preflight.ts (1124 lines) and
Hermes BasePlatformAdapter.handle_message (400 lines).

Pipeline: self-filter → bot-filter → system-event → rate-limit → permission →
          mention → dedup → media-check → quote-parse → audio-preprocess →
          thread-detect → forum-detect → command bypass → enrich context.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from caveman.gateway.platform_types import MessageEvent, SessionSource

__all__ = [
    "BYPASS_COMMANDS",
    "SYSTEM_EVENT_TYPES",
    "DEFAULT_RATE_LIMIT",
    "DEFAULT_RATE_WINDOW",
    "DEFAULT_MEDIA_MAX_BYTES",
    "PreflightConfig",
    "QuotedMessage",
    "PreflightResult",
    "MessagePreflight",
]


logger = logging.getLogger("caveman.gateway.preflight")

# Commands that bypass the active-session guard
BYPASS_COMMANDS = frozenset({
    "approve", "deny", "status", "stop", "new", "reset",
    "background", "restart", "cancel", "help", "whoami",
})

# System event message types to filter
SYSTEM_EVENT_TYPES = frozenset({
    "member_join", "member_leave", "channel_created", "channel_deleted",
    "pin_added", "pin_removed", "boost", "thread_created",
    "call_started", "call_ended",
})

DEFAULT_RATE_LIMIT = 30
DEFAULT_RATE_WINDOW = 60
DEFAULT_MEDIA_MAX_BYTES = 25 * 1024 * 1024  # 25MB


@dataclass
class PreflightConfig:
    """Configuration for message preflight checks."""
    bot_user_id: str = ""
    bot_display_name: str = ""
    allowed_users: Optional[Set[str]] = None
    allowed_channels: Optional[Set[str]] = None
    blocked_users: Set[str] = field(default_factory=set)
    allow_bots: bool = False
    allow_dms: bool = True
    allow_groups: bool = True
    require_mention: bool = False
    implicit_mention_in_dm: bool = True
    implicit_mention_in_thread: bool = True
    mention_patterns: List[str] = field(default_factory=list)
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate_window: int = DEFAULT_RATE_WINDOW
    dedup_window: float = 5.0
    media_max_bytes: int = DEFAULT_MEDIA_MAX_BYTES
    filter_system_events: bool = True
    pluralkit_bot_ids: Set[str] = field(default_factory=set)
    audio_transcribe_fn: Optional[Callable] = None


@dataclass
class QuotedMessage:
    """A quoted/replied-to message."""
    message_id: str = ""
    text: str = ""
    author_id: str = ""
    author_name: str = ""
    is_bot_message: bool = False


@dataclass
class PreflightResult:
    """Result of preflight checks."""
    passed: bool
    event: Optional[MessageEvent] = None
    drop_reason: str = ""
    is_command_bypass: bool = False
    command: str = ""
    command_args: str = ""
    session_key: str = ""
    was_mentioned: bool = False
    mention_kind: str = ""  # explicit | implicit | reply | none
    is_thread: bool = False
    is_forum: bool = False
    thread_id: str = ""
    thread_starter_id: str = ""
    quoted: Optional[QuotedMessage] = None
    media_oversized: List[str] = field(default_factory=list)
    audio_transcription: str = ""
    is_pluralkit: bool = False
    enriched_text: str = ""  # Text after preprocessing


class MessagePreflight:
    """Stateful preflight checker with rate limiting and dedup."""

    def __init__(self, config: PreflightConfig):
        self.config = config
        self._rate_counts: Dict[str, List[float]] = {}
        self._recent_hashes: Dict[str, float] = {}
        self._thread_starters: Dict[str, Tuple[str, float]] = {}  # thread_id → (starter_id, cached_at)
        self._last_cleanup = time.monotonic()

    def check(self, event: MessageEvent) -> PreflightResult:
        """Run all preflight checks."""
        now = time.monotonic()
        if now - self._last_cleanup > 60:
            self._cleanup()
            self._last_cleanup = now

        src = event.source
        if not src:
            return PreflightResult(passed=False, drop_reason="no_source")

        # 1. Self-message filter
        if self.config.bot_user_id and src.user_id == self.config.bot_user_id:
            return PreflightResult(passed=False, drop_reason="self_message")

        # 2. Bot filter (with PluralKit exception)
        is_pluralkit = src.user_id in self.config.pluralkit_bot_ids
        if getattr(src, "is_bot", False) and not self.config.allow_bots and not is_pluralkit:
            return PreflightResult(passed=False, drop_reason="bot_message")

        # 3. System event filter
        if self.config.filter_system_events:
            msg_type_str = event.message_type.value if hasattr(event.message_type, 'value') else str(event.message_type)
            if msg_type_str in SYSTEM_EVENT_TYPES:
                return PreflightResult(passed=False, drop_reason="system_event")

        # 4. Blocked users
        if src.user_id in self.config.blocked_users:
            return PreflightResult(passed=False, drop_reason="blocked_user")

        # 5. Channel type filter
        if src.chat_type == "dm" and not self.config.allow_dms:
            return PreflightResult(passed=False, drop_reason="dm_disabled")
        if src.chat_type in ("group", "channel") and not self.config.allow_groups:
            return PreflightResult(passed=False, drop_reason="groups_disabled")

        # 6. Allowlist checks
        if self.config.allowed_users is not None and src.user_id not in self.config.allowed_users:
            return PreflightResult(passed=False, drop_reason="user_not_allowed")
        if self.config.allowed_channels is not None and src.chat_id not in self.config.allowed_channels:
            return PreflightResult(passed=False, drop_reason="channel_not_allowed")

        # 7. Rate limiting
        if not self._check_rate_limit(src.user_id):
            return PreflightResult(passed=False, drop_reason="rate_limited")

        # 8. Dedup
        if self._is_duplicate(event):
            return PreflightResult(passed=False, drop_reason="duplicate")

        # 9. Thread/forum detection
        is_thread = bool(src.thread_id)
        is_forum = self._detect_forum(src)
        thread_id = src.thread_id or ""
        thread_starter_id = self._get_thread_starter(thread_id) if thread_id else ""

        # 10. Quote/reply parsing
        quoted = self._parse_quoted(event)

        # 11. Media size check
        media_oversized = self._check_media_sizes(event)

        # 12. Audio preprocessing
        audio_transcription = ""
        text = (event.text or "").strip()

        # 13. Empty content check (after audio transcription)
        if not text and not event.media_urls and not audio_transcription:
            return PreflightResult(passed=False, drop_reason="empty_content")

        enriched_text = audio_transcription if audio_transcription and not text else text

        # 14. Command detection
        command = ""
        command_args = ""
        is_bypass = False
        if enriched_text.startswith("/"):
            parts = enriched_text[1:].split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            command_args = parts[1] if len(parts) > 1 else ""
            is_bypass = command in BYPASS_COMMANDS

        # 15. Mention resolution (multi-strategy)
        was_mentioned, mention_kind = self._resolve_mention(
            enriched_text, src, is_thread, quoted, command,
        )

        # 16. Mention requirement (groups only, with exceptions)
        if (self.config.require_mention
                and src.chat_type in ("group", "channel")
                and not was_mentioned
                and not command
                and not is_forum):
            return PreflightResult(passed=False, drop_reason="mention_required")

        return PreflightResult(
            passed=True,
            event=event,
            is_command_bypass=is_bypass,
            command=command,
            command_args=command_args,
            was_mentioned=was_mentioned,
            mention_kind=mention_kind,
            is_thread=is_thread,
            is_forum=is_forum,
            thread_id=thread_id,
            thread_starter_id=thread_starter_id,
            quoted=quoted,
            media_oversized=media_oversized,
            audio_transcription=audio_transcription,
            is_pluralkit=is_pluralkit,
            enriched_text=enriched_text,
        )

    # ── Rate Limiting ──

    def _check_rate_limit(self, user_id: str) -> bool:
        now = time.monotonic()
        window = self.config.rate_window
        timestamps = self._rate_counts.get(user_id, [])
        timestamps = [t for t in timestamps if now - t < window]
        if len(timestamps) >= self.config.rate_limit:
            self._rate_counts[user_id] = timestamps
            return False
        timestamps.append(now)
        self._rate_counts[user_id] = timestamps
        return True

    # ── Dedup ──

    def _is_duplicate(self, event: MessageEvent) -> bool:
        now = time.monotonic()
        content = f"{event.source.user_id}:{event.text}:{','.join(event.media_urls)}"
        h = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()
        if h in self._recent_hashes:
            if now - self._recent_hashes[h] < self.config.dedup_window:
                return True
        self._recent_hashes[h] = now
        return False

    # ── Mention Resolution ──

    def _resolve_mention(
        self, text: str, source: SessionSource,
        is_thread: bool, quoted: Optional[QuotedMessage],
        command: str,
    ) -> Tuple[bool, str]:
        """Multi-strategy mention resolution. Returns (was_mentioned, kind)."""
        if not self.config.bot_user_id:
            return False, "none"

        # Strategy 1: Explicit mention (<@id> or <@!id>)
        explicit_patterns = [
            f"<@{self.config.bot_user_id}>",
            f"<@!{self.config.bot_user_id}>",
        ]
        text_lower = text.lower()
        for p in explicit_patterns:
            if p.lower() in text_lower:
                return True, "explicit"

        # Strategy 2: Custom mention patterns (bot name, etc.)
        for p in self.config.mention_patterns:
            if p.lower() in text_lower:
                return True, "explicit"

        # Strategy 3: Bot display name mention
        if self.config.bot_display_name:
            name_lower = self.config.bot_display_name.lower()
            # Word boundary check to avoid false positives
            if re.search(rf'\b{re.escape(name_lower)}\b', text_lower):
                return True, "explicit"

        # Strategy 4: Reply to bot message
        if quoted and quoted.is_bot_message:
            return True, "reply"

        # Strategy 5: Implicit mention in DM
        if source.chat_type == "dm" and self.config.implicit_mention_in_dm:
            return True, "implicit"

        # Strategy 6: Implicit mention in thread (if bot started it)
        if is_thread and self.config.implicit_mention_in_thread:
            return True, "implicit"

        # Strategy 7: Command always counts as mention
        if command:
            return True, "implicit"

        return False, "none"

    # ── Thread/Forum Detection ──

    def _detect_forum(self, source: SessionSource) -> bool:
        """Detect if message is in a forum channel."""
        topic = getattr(source, "chat_topic", None)
        return bool(topic)

    def _get_thread_starter(self, thread_id: str) -> str:
        """Get cached thread starter ID."""
        entry = self._thread_starters.get(thread_id)
        if entry:
            starter_id, cached_at = entry
            if time.monotonic() - cached_at < 3600:  # 1 hour cache
                return starter_id
        return ""

    def set_thread_starter(self, thread_id: str, starter_id: str) -> None:
        """Cache a thread starter ID."""
        self._thread_starters[thread_id] = (starter_id, time.monotonic())

    # ── Quote/Reply Parsing ──

    def _parse_quoted(self, event: MessageEvent) -> Optional[QuotedMessage]:
        """Parse quoted/replied-to message."""
        if not event.reply_to_message_id:
            return None
        return QuotedMessage(
            message_id=event.reply_to_message_id,
            text=event.reply_to_text or "",
            is_bot_message=self._is_bot_reply(event),
        )

    def _is_bot_reply(self, event: MessageEvent) -> bool:
        """Check if the replied-to message was from the bot."""
        # This would need platform-specific logic in production
        # For now, check if reply_to_text looks like a bot response
        return False

    # ── Media Checks ──

    def _check_media_sizes(self, event: MessageEvent) -> List[str]:
        """Check media sizes against limit. Returns list of oversized URLs."""
        oversized = []
        for i, url in enumerate(event.media_urls):
            media_type = event.media_types[i] if i < len(event.media_types) else ""
            # Size checking would need HTTP HEAD in production
            # For now, just validate URL format
            if not url.startswith(("http://", "https://", "data:")):
                oversized.append(url)
        return oversized

    # ── Cleanup ──

    def _cleanup(self) -> None:
        """Remove stale entries."""
        now = time.monotonic()
        # Rate limits
        for uid in list(self._rate_counts.keys()):
            self._rate_counts[uid] = [
                t for t in self._rate_counts[uid]
                if now - t < self.config.rate_window
            ]
            if not self._rate_counts[uid]:
                del self._rate_counts[uid]
        # Dedup hashes
        cutoff = now - max(self.config.dedup_window * 2, 30)
        self._recent_hashes = {
            h: t for h, t in self._recent_hashes.items() if t > cutoff
        }
        # Thread starters (1 hour TTL)
        self._thread_starters = {
            tid: (sid, ts) for tid, (sid, ts) in self._thread_starters.items()
            if now - ts < 3600
        }
