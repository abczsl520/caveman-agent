"""Shared gateway platform send/delivery helpers."""
from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

from caveman.gateway.platform_delivery import (
    RETRYABLE_PATTERNS,
    extract_images,
    extract_media,
    is_animation_url,
    truncate_message,
)
from caveman.gateway.platform_types import SendResult

logger = logging.getLogger("caveman.gateway")
_AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})


async def deliver_response(
    adapter: Any,
    chat_id: str,
    response: str,
    reply_to: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Deliver a response: redact secrets, extract media, split, send."""
    try:
        from caveman.gateway.redaction import redact_all
        response = redact_all(response)
    except Exception as exc:
        logger.debug("deliver_response: suppressed %s", exc)
    media_files, response = extract_media(response)
    images, text_content = extract_images(response)

    if text_content.strip():
        chunks = truncate_message(text_content, adapter._max_message_length)
        for i, chunk in enumerate(chunks):
            r2 = reply_to if i == 0 and adapter.config.reply_to_mode != "off" else None
            await send_with_retry(adapter, chat_id, chunk, r2, metadata)

    for image_url, alt_text in images:
        if is_animation_url(image_url):
            await adapter.send_animation(chat_id, image_url, alt_text or None, metadata=metadata)
        else:
            await adapter.send_image(chat_id, image_url, alt_text or None, metadata=metadata)

    for media_path, _is_voice in media_files:
        ext = Path(media_path).suffix.lower()
        if ext in _AUDIO_EXTS:
            await adapter.send_voice(chat_id, media_path, metadata=metadata)
        elif ext in _VIDEO_EXTS:
            await adapter.send_video(chat_id, media_path, metadata=metadata)
        elif ext in _IMAGE_EXTS:
            await adapter.send_image_file(chat_id, media_path, metadata=metadata)
        else:
            await adapter.send_document(chat_id, media_path, metadata=metadata)


async def send_with_retry(
    adapter: Any,
    chat_id: str,
    content: str,
    reply_to: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
    base_delay: float = 2.0,
) -> SendResult:
    """Send with automatic retry for transient network errors."""
    result = await adapter.send(chat_id, content, reply_to, metadata)
    if result.success:
        return result

    error_str = (result.error or "").lower()
    is_network = result.retryable or any(p in error_str for p in RETRYABLE_PATTERNS)

    if not is_network:
        fallback = await adapter.send(chat_id, f"(plain text fallback)\n\n{content[:3500]}", reply_to, metadata)
        return fallback if fallback.success else result

    for attempt in range(1, max_retries + 1):
        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
        logger.warning("[%s] Send retry %d/%d in %.1fs: %s", adapter.name, attempt, max_retries, delay, result.error)
        await asyncio.sleep(delay)
        result = await adapter.send(chat_id, content, reply_to, metadata)
        if result.success:
            return result

    try:
        await adapter.send(chat_id, "⚠️ 消息投递重试后仍失败；这不是任务完成信号，已记录日志，请继续排查投递链路。")
    except Exception as exc:
        logger.debug("send_with_retry: suppressed %s", exc)
    return result
