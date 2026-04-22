"""Message Processing Engine — handle messages end-to-end.

Extracted from Hermes _process_message_background (270 lines) and
OpenClaw processDiscordMessage (900 lines).

Handles: session management → typing → agent call → media extraction →
         response delivery → interrupt handling → error recovery.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from caveman.gateway.platform_types import (
    MessageEvent, MessageType, ProcessingOutcome,
)

logger = logging.getLogger("caveman.gateway.processor")

# File type routing (from Hermes)
_AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})

# Media extraction patterns
_MEDIA_TAG_RE = re.compile(r"MEDIA:\s*(\S+)")
_IMAGE_MD_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)]+)\)")
_LOCAL_FILE_RE = re.compile(r"(?:^|\s)(/[\w./\-]+\.(?:png|jpg|jpeg|gif|webp|mp4|pdf|ogg|mp3))\b")
_INTERNAL_DIRECTIVES = re.compile(r"\[\[audio_as_voice\]\]|MEDIA:\s*\S+")


class MessageProcessor:
    """Processes messages through the full pipeline.

    Integrates with BasePlatformAdapter for sending responses.
    """

    def __init__(self, adapter: Any):
        self._adapter = adapter
        self._active_sessions: Dict[str, asyncio.Event] = {}
        self._pending_messages: Dict[str, MessageEvent] = {}
        self._background_tasks: Set[asyncio.Task] = set()

    @property
    def active_session_count(self) -> int:
        return len(self._active_sessions)

    def has_active_session(self, session_key: str) -> bool:
        return session_key in self._active_sessions

    # ── Main Entry Point ──

    async def process(self, event: MessageEvent, session_key: str) -> None:
        """Process a message. Handles session locking and interrupt."""
        from caveman.gateway.preflight import BYPASS_COMMANDS

        # Active session? Check for command bypass or queue interrupt
        if session_key in self._active_sessions:
            cmd = event.get_command() if hasattr(event, "get_command") else ""
            if not cmd and (event.text or "").startswith("/"):
                cmd = (event.text or "")[1:].split()[0].lower()

            if cmd in BYPASS_COMMANDS:
                logger.debug("Command /%s bypassing active session %s", cmd, session_key)
                await self._dispatch_direct(event)
                return

            # Photo bursts: queue without interrupt (from Hermes)
            if event.message_type == MessageType.PHOTO:
                logger.debug("Queuing photo for session %s", session_key)
                self._merge_pending(session_key, event)
                return

            # Default: interrupt the running agent
            logger.debug("Interrupt: new message for active session %s", session_key)
            self._pending_messages[session_key] = event
            interrupt = self._active_sessions.get(session_key)
            if interrupt:
                interrupt.set()
            return

        # Mark session active BEFORE spawning task (race prevention from Hermes)
        self._active_sessions[session_key] = asyncio.Event()

        task = asyncio.create_task(self._run(event, session_key))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run(self, event: MessageEvent, session_key: str) -> None:
        """Background task: typing → handler → media → delivery → cleanup."""
        thread_meta = (
            {"thread_id": event.source.thread_id}
            if event.source and event.source.thread_id else None
        )

        # Start typing indicator
        typing_task = asyncio.create_task(
            self._keep_typing(event.source.chat_id if event.source else "", thread_meta)
        )

        try:
            await self._emit_hook("on_processing_start", event)

            # Call the message handler
            handler = getattr(self._adapter, "_message_handler", None)
            if not handler:
                return

            response = await handler(event)

            if not response:
                logger.debug("Handler returned empty response for %s",
                             event.source.chat_id if event.source else "?")
                await self._emit_hook("on_processing_complete", event, ProcessingOutcome.SUCCESS)
                return

            # Extract and deliver response
            await self._deliver_response(
                response=response,
                event=event,
                thread_meta=thread_meta,
            )

            await self._emit_hook("on_processing_complete", event, ProcessingOutcome.SUCCESS)

            # Process pending message from interrupt
            if session_key in self._pending_messages:
                pending = self._pending_messages.pop(session_key)
                logger.debug("Processing queued message from interrupt")
                self._active_sessions.pop(session_key, None)
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass  # intentional: Exception suppressed
                await self._run(pending, session_key)
                return

        except asyncio.CancelledError:
            await self._emit_hook("on_processing_complete", event, ProcessingOutcome.CANCELLED)
            raise
        except Exception as e:
            await self._emit_hook("on_processing_complete", event, ProcessingOutcome.FAILURE)
            logger.error("Error processing message: %s", e, exc_info=True)
            # Send error to user (from Hermes — never leave user with silence)
            try:
                error_type = type(e).__name__
                detail = str(e)[:300] if str(e) else "no details"
                chat_id = event.source.chat_id if event.source else ""
                if chat_id:
                    await self._adapter.send(
                        chat_id,
                        f"Sorry, I encountered an error ({error_type}).\n{detail}\n"
                        "Try again or use /reset to start a fresh session.",
                        metadata=thread_meta,
                    )
            except Exception as exc:
                logger.debug("unknown: suppressed %s", exc)
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass  # intentional: Exception suppressed
            # Stop platform typing
            if hasattr(self._adapter, "stop_typing") and event.source:
                try:
                    await self._adapter.stop_typing(event.source.chat_id)
                except Exception as exc:
                    logger.debug("unknown: suppressed %s", exc)
            self._active_sessions.pop(session_key, None)

    # ── Response Delivery ──

    async def _deliver_response(
        self, response: str, event: MessageEvent,
        thread_meta: Optional[Dict] = None,
    ) -> None:
        """Extract media, send text, then send media attachments."""
        chat_id = event.source.chat_id if event.source else ""
        if not chat_id:
            return

        # 1. Extract MEDIA: tags
        media_files = extract_media_tags(response)
        response = _MEDIA_TAG_RE.sub("", response)

        # 2. Extract markdown images
        images, response = extract_images(response)

        # 3. Extract local file paths
        local_files, response = extract_local_files(response)

        # 4. Clean internal directives
        response = _INTERNAL_DIRECTIVES.sub("", response).strip()

        # 5. Send text
        if response:
            await self._adapter._send_with_retry(
                chat_id=chat_id,
                content=response,
                reply_to=event.message_id,
                metadata=thread_meta,
            )

        # 6. Send images
        for url, alt in images:
            try:
                if _is_animation_url(url):
                    await self._adapter.send_animation(
                        chat_id=chat_id, animation_url=url,
                        caption=alt or None, metadata=thread_meta,
                    )
                else:
                    await self._adapter.send_image(
                        chat_id=chat_id, image_url=url,
                        caption=alt or None, metadata=thread_meta,
                    )
            except Exception as e:
                logger.warning("Failed to send image: %s", e)

        # 7. Send media files (route by type)
        for path, _is_voice in media_files:
            await self._send_media_file(chat_id, path, thread_meta)

        # 8. Send local files
        for path in local_files:
            await self._send_media_file(chat_id, path, thread_meta)

    async def _send_media_file(
        self, chat_id: str, file_path: str,
        thread_meta: Optional[Dict] = None,
    ) -> None:
        """Route a file to the correct send method by extension."""
        ext = Path(file_path).suffix.lower()
        try:
            if ext in _AUDIO_EXTS:
                await self._adapter.send_voice(
                    chat_id=chat_id, audio_path=file_path, metadata=thread_meta)
            elif ext in _VIDEO_EXTS:
                await self._adapter.send_video(
                    chat_id=chat_id, video_path=file_path, metadata=thread_meta)
            elif ext in _IMAGE_EXTS:
                await self._adapter.send_image_file(
                    chat_id=chat_id, image_path=file_path, metadata=thread_meta)
            else:
                await self._adapter.send_document(
                    chat_id=chat_id, file_path=file_path, metadata=thread_meta)
        except Exception as e:
            logger.warning("Failed to send media %s: %s", file_path, e)

    # ── Helpers ──

    async def _dispatch_direct(self, event: MessageEvent) -> None:
        """Dispatch a command directly (bypass session queue)."""
        handler = getattr(self._adapter, "_message_handler", None)
        if not handler:
            return
        try:
            response = await handler(event)
            if response and event.source:
                thread_meta = (
                    {"thread_id": event.source.thread_id}
                    if event.source.thread_id else None
                )
                await self._adapter._send_with_retry(
                    chat_id=event.source.chat_id,
                    content=response,
                    reply_to=event.message_id,
                    metadata=thread_meta,
                )
        except Exception as e:
            logger.error("Command dispatch failed: %s", e, exc_info=True)

    def _merge_pending(self, session_key: str, event: MessageEvent) -> None:
        """Merge photo burst into pending message (from Hermes)."""
        existing = self._pending_messages.get(session_key)
        if existing and existing.media_urls:
            existing.media_urls.extend(event.media_urls)
            if event.text and event.text not in (existing.text or ""):
                existing.text = f"{existing.text or ''}\n{event.text}".strip()
        else:
            self._pending_messages[session_key] = event

    async def _keep_typing(
        self, chat_id: str, metadata: Optional[Dict] = None,
        interval: float = 4.0,
    ) -> None:
        """Continuously send typing indicator."""
        if not chat_id:
            return
        while True:
            try:
                if hasattr(self._adapter, "send_typing"):
                    await self._adapter.send_typing(chat_id, metadata)
            except Exception as exc:
                logger.debug("_keep_typing: suppressed %s", exc)
            await asyncio.sleep(interval)

    async def _emit_hook(self, hook_name: str, *args: Any) -> None:
        """Call adapter hook if it exists."""
        hook = getattr(self._adapter, hook_name, None)
        if hook:
            try:
                await hook(*args)
            except Exception:
                logger.debug("Hook %s failed", hook_name, exc_info=True)

    async def cancel_all(self) -> None:
        """Cancel all background tasks."""
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._active_sessions.clear()
        self._pending_messages.clear()


# ── Standalone extraction functions ──

def extract_media_tags(content: str) -> List[Tuple[str, bool]]:
    """Extract MEDIA:<path> tags. Returns [(path, is_voice)]."""
    results = []
    for match in _MEDIA_TAG_RE.finditer(content):
        path = match.group(1)
        is_voice = path.endswith((".ogg", ".opus"))
        results.append((path, is_voice))
    return results


def extract_images(content: str) -> Tuple[List[Tuple[str, str]], str]:
    """Extract markdown images. Returns ([(url, alt)], cleaned_text)."""
    images = []
    for match in _IMAGE_MD_RE.finditer(content):
        images.append((match.group(2), match.group(1)))
    cleaned = _IMAGE_MD_RE.sub("", content).strip()
    return images, cleaned


def extract_local_files(content: str) -> Tuple[List[str], str]:
    """Extract local file paths. Returns ([paths], cleaned_text)."""
    files = []
    for match in _LOCAL_FILE_RE.finditer(content):
        path = match.group(1)
        if Path(path).exists():
            files.append(path)
    cleaned = content
    for f in files:
        cleaned = cleaned.replace(f, "").strip()
    return files, cleaned


def _is_animation_url(url: str) -> bool:
    """Check if URL points to an animated image."""
    lower = url.lower().split("?")[0]
    return lower.endswith(".gif") or "giphy.com" in lower or "tenor.com" in lower

from caveman.gateway.processor_depth import (  # noqa: F401,E402  # depth wiring
    ToolProgressEvent,
    RetryConfig,
    StreamingProcessor,
)

__all__ = [
    "MessageProcessor",
    "extract_media_tags",
    "extract_images",
    "extract_local_files",
    "ToolProgressEvent",
    "RetryConfig",
    "StreamingProcessor",
]

