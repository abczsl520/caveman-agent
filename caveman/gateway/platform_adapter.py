"""Base Platform Adapter — unified abstraction for all messaging platforms.

Subclasses implement connect/disconnect/send. Base handles: typing, retry,
media extraction, interrupt, session tracking, message splitting.
"""
from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Set

from caveman.gateway.platform_types import (
    MessageEvent,
    MessageHandler,
    MessageType,
    Platform,
    PlatformConfig,
    ProcessingOutcome,
    SendResult,
    SessionSource,
    build_session_key,
)
from caveman.gateway.platform_delivery import (
    RETRYABLE_PATTERNS,
    extract_images,
    extract_media,
    is_animation_url,
    truncate_message,
)

logger = logging.getLogger("caveman.gateway")

# Media file extensions for routing
_AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

class BasePlatformAdapter(ABC):
    """Base class for all platform adapters.

    Subclasses implement platform-specific logic for connecting, receiving,
    and sending messages. The base class handles the full message lifecycle:
    typing indicators, retry logic, media extraction, interrupt support,
    and session tracking.
    """

    def __init__(self, config: PlatformConfig, platform: Platform):
        self.config = config
        self.platform = platform
        self._message_handler: Optional[MessageHandler] = None
        self._running = False

        # Session tracking for interrupt support
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._pending_messages: Dict[str, MessageEvent] = {}
        self._background_tasks: Set[asyncio.Task] = set()

        # Typing control
        self._typing_paused: Set[str] = set()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return self.platform.value.title()

    @property
    def is_connected(self) -> bool:
        return self._running

    # ── Configuration ───────────────────────────────────────────────────────

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Set the handler for incoming messages."""
        self._message_handler = handler

    # ── Abstract methods (MUST implement) ───────────────────────────────────

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the platform. Returns True on success."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the platform."""
        ...

    @abstractmethod
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message. Returns SendResult."""
        ...

    # ── Optional overrides (sensible defaults) ──────────────────────────────

    async def edit_message(
        self, chat_id: str, message_id: str, content: str,
    ) -> SendResult:
        """Edit a previously sent message. Default: not supported."""
        return SendResult(success=False, error="Not supported")

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        """Send typing indicator. Override if platform supports it."""

    async def stop_typing(self, chat_id: str) -> None:
        """Stop typing indicator. Override for platforms with persistent typing."""

    async def send_image(
        self, chat_id: str, image_url: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image by URL. Default: send URL as text."""
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to)

    async def send_image_file(
        self, chat_id: str, image_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a local image file. Default: send path as text."""
        text = f"🖼️ {caption}" if caption else f"🖼️ Image: {Path(image_path).name}"
        return await self.send(chat_id, text, reply_to)

    async def send_voice(
        self, chat_id: str, audio_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send audio as voice message. Default: send path as text."""
        text = f"🔊 {caption}" if caption else f"🔊 Audio: {Path(audio_path).name}"
        return await self.send(chat_id, text, reply_to)

    async def send_video(
        self, chat_id: str, video_path: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a video. Default: send path as text."""
        text = f"🎬 {caption}" if caption else f"🎬 Video: {Path(video_path).name}"
        return await self.send(chat_id, text, reply_to)

    async def send_document(
        self, chat_id: str, file_path: str,
        caption: Optional[str] = None, file_name: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        """Send a file/document. Default: send path as text."""
        name = file_name or Path(file_path).name
        text = f"📎 {caption}\n{name}" if caption else f"📎 File: {name}"
        return await self.send(chat_id, text, reply_to)

    async def send_animation(
        self, chat_id: str, animation_url: str,
        caption: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send animated GIF. Default: falls back to send_image."""
        return await self.send_image(chat_id, animation_url, caption, reply_to, metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get chat metadata. Default: minimal info."""
        return {"name": chat_id, "type": "unknown"}

    def format_message(self, content: str) -> str:
        """Platform-specific message formatting. Default: pass-through."""
        return content

    # ── Lifecycle hooks (override for platform-specific reactions) ───────────

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Called when message processing begins (e.g., add 👀 reaction)."""

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome,
    ) -> None:
        """Called when processing completes (e.g., add ✅/❌ reaction)."""

    # ── Message handling (DO NOT override) ──────────────────────────────────

    async def handle_message(self, event: MessageEvent) -> None:
        """Process an incoming message. Spawns background task for non-blocking."""
        if not self._message_handler:
            return

        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
        )

        # Session already active — handle interrupt or queue
        if session_key in self._active_sessions:
            # Bypass commands that must execute immediately
            cmd = event.get_command()
            if cmd in ("approve", "deny", "stop", "new", "reset", "status"):
                try:
                    response = await self._message_handler(event)
                    if response:
                        meta = {"thread_id": event.source.thread_id} if event.source and event.source.thread_id else None
                        await self._send_with_retry(event.source.chat_id, response, event.message_id, meta)
                except Exception as e:
                    logger.error("[%s] Command bypass failed: %s", self.name, e)
                return

            # Photo bursts: merge without interrupt
            if event.message_type == MessageType.PHOTO:
                self._merge_pending(session_key, event)
                return

            # Default: queue and signal interrupt
            self._pending_messages[session_key] = event
            self._active_sessions[session_key].set()
            return

        # Mark session active BEFORE spawning task (prevents race)
        self._active_sessions[session_key] = asyncio.Event()

        task = asyncio.create_task(self._process_message_background(event, session_key))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_message_background(
        self, event: MessageEvent, session_key: str,
    ) -> None:
        """Background task: typing → handler → deliver response → cleanup."""
        chat_id = event.source.chat_id
        thread_meta = {"thread_id": event.source.thread_id} if event.source and event.source.thread_id else None

        # Start typing indicator
        typing_task = asyncio.create_task(self._keep_typing(chat_id, metadata=thread_meta))

        try:
            await self._run_hook("on_processing_start", event)

            # Call the agent handler
            response = await self._message_handler(event)

            if response:
                await self._deliver_response(chat_id, response, event.message_id, thread_meta)

            outcome = ProcessingOutcome.SUCCESS
            await self._run_hook("on_processing_complete", event, outcome)

            # Process any pending message queued during our run
            if session_key in self._pending_messages:
                pending = self._pending_messages.pop(session_key)
                if session_key in self._active_sessions:
                    del self._active_sessions[session_key]
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass  # intentional: Exception suppressed
                await self._process_message_background(pending, session_key)
                return

        except asyncio.CancelledError:
            await self._run_hook("on_processing_complete", event, ProcessingOutcome.CANCELLED)
            raise
        except Exception as e:
            await self._run_hook("on_processing_complete", event, ProcessingOutcome.FAILURE)
            logger.error("[%s] Error processing message: %s", self.name, e, exc_info=True)
            try:
                await self.send(chat_id, f"⚠️ Error: {type(e).__name__}: {str(e)[:300]}", metadata=thread_meta)
            except Exception as exc:
                logger.debug("_process_message_background: suppressed %s", exc)
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass  # intentional: Exception suppressed
            try:
                await self.stop_typing(chat_id)
            except Exception as exc:
                logger.debug("unknown: suppressed %s", exc)
            if session_key in self._active_sessions:
                del self._active_sessions[session_key]

    # ── Response delivery ───────────────────────────────────────────────────

    async def _deliver_response(
        self, chat_id: str, response: str,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Deliver a response: redact secrets, extract media, split, send."""
        try:
            from caveman.gateway.redaction import redact_all
            response = redact_all(response)
        except Exception as exc:
            logger.debug("_deliver_response: suppressed %s", exc)
        media_files, response = extract_media(response)
        images, text_content = extract_images(response)

        if text_content.strip():
            chunks = truncate_message(text_content, self._max_message_length)
            for i, chunk in enumerate(chunks):
                r2 = reply_to if i == 0 and self.config.reply_to_mode != "off" else None
                await self._send_with_retry(chat_id, chunk, r2, metadata)

        for image_url, alt_text in images:
            if is_animation_url(image_url):
                await self.send_animation(chat_id, image_url, alt_text or None, metadata=metadata)
            else:
                await self.send_image(chat_id, image_url, alt_text or None, metadata=metadata)

        for media_path, _is_voice in media_files:
            ext = Path(media_path).suffix.lower()
            if ext in _AUDIO_EXTS:
                await self.send_voice(chat_id, media_path, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                await self.send_video(chat_id, media_path, metadata=metadata)
            elif ext in _IMAGE_EXTS:
                await self.send_image_file(chat_id, media_path, metadata=metadata)
            else:
                await self.send_document(chat_id, media_path, metadata=metadata)

    # ── Retry logic ─────────────────────────────────────────────────────────

    async def _send_with_retry(
        self, chat_id: str, content: str,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 2, base_delay: float = 2.0,
    ) -> SendResult:
        """Send with automatic retry for transient network errors."""
        result = await self.send(chat_id, content, reply_to, metadata)
        if result.success:
            return result

        error_str = (result.error or "").lower()
        is_network = result.retryable or any(p in error_str for p in RETRYABLE_PATTERNS)

        if not is_network:
            # Try plain-text fallback for formatting errors
            fallback = await self.send(chat_id, f"(plain text fallback)\n\n{content[:3500]}", reply_to, metadata)
            return fallback if fallback.success else result

        # Retry with exponential backoff
        for attempt in range(1, max_retries + 1):
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            logger.warning("[%s] Send retry %d/%d in %.1fs: %s", self.name, attempt, max_retries, delay, result.error)
            await asyncio.sleep(delay)
            result = await self.send(chat_id, content, reply_to, metadata)
            if result.success:
                return result

        # All retries failed — notify user
        try:
            await self.send(chat_id, "⚠️ Message delivery failed after retries. Please try again.")
        except Exception as exc:
            logger.debug("_send_with_retry: suppressed %s", exc)
        return result

    # ── Typing indicator ────────────────────────────────────────────────────

    async def _keep_typing(self, chat_id: str, interval: float = 4.0, metadata: Any = None) -> None:
        """Continuously send typing indicator until cancelled."""
        try:
            while True:
                if chat_id not in self._typing_paused:
                    await self.send_typing(chat_id, metadata)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass  # intentional: Exception suppressed

    def pause_typing(self, chat_id: str) -> None:
        """Pause typing for a chat (e.g., during approval waits)."""
        self._typing_paused.add(chat_id)

    def resume_typing(self, chat_id: str) -> None:
        """Resume typing for a chat."""
        self._typing_paused.discard(chat_id)

    # ── Message length ────────────────────────────────────────────────────

    @property
    def _max_message_length(self) -> int:
        """Platform-specific max message length. Override in subclasses."""
        return 4096

    # ── Helpers ─────────────────────────────────────────────────────────────

    def build_source(
        self, chat_id: str, *,
        chat_name: Optional[str] = None, chat_type: str = "dm",
        user_id: Optional[str] = None, user_name: Optional[str] = None,
        thread_id: Optional[str] = None, chat_topic: Optional[str] = None,
    ) -> SessionSource:
        """Helper to build a SessionSource for this platform."""
        return SessionSource(
            platform=self.platform,
            chat_id=str(chat_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            user_name=user_name,
            thread_id=str(thread_id) if thread_id else None,
            chat_topic=chat_topic.strip() if chat_topic else None,
        )

    def has_pending_interrupt(self, session_key: str) -> bool:
        """Check if there's a pending interrupt for a session."""
        return session_key in self._active_sessions and self._active_sessions[session_key].is_set()

    def get_pending_message(self, session_key: str) -> Optional[MessageEvent]:
        """Get and clear any pending message for a session."""
        return self._pending_messages.pop(session_key, None)

    async def cancel_background_tasks(self) -> None:
        """Cancel all in-flight background tasks (for shutdown)."""
        tasks = [t for t in self._background_tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._pending_messages.clear()
        self._active_sessions.clear()

    def _merge_pending(self, session_key: str, event: MessageEvent) -> None:
        """Merge a photo event into pending (for album/burst support)."""
        existing = self._pending_messages.get(session_key)
        if existing and existing.message_type == MessageType.PHOTO and event.message_type == MessageType.PHOTO:
            existing.media_urls.extend(event.media_urls)
            existing.media_types.extend(event.media_types)
            if event.text:
                existing.text = f"{existing.text}\n\n{event.text}".strip() if existing.text else event.text
        else:
            self._pending_messages[session_key] = event

    async def _run_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Run a lifecycle hook without letting failures break message flow."""
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            return
        try:
            await hook(*args, **kwargs)
        except Exception as e:
            logger.debug("[%s] Hook %s failed: %s", self.name, hook_name, e)
