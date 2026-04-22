"""Platform delivery utilities — media extraction, retry, message splitting.

Extracted from platform_adapter.py to keep it under 450 lines.
These are static/standalone utilities used by BasePlatformAdapter.
"""
from __future__ import annotations

import re
from typing import List, Tuple

__all__ = [
    "RETRYABLE_PATTERNS",
    "extract_media",
    "extract_images",
    "is_animation_url",
    "truncate_message",
]


# Error patterns that indicate transient network failures (safe to retry)
RETRYABLE_PATTERNS = (
    "connecterror", "connectionerror", "connectionreset",
    "connectionrefused", "connecttimeout", "network",
    "broken pipe", "remotedisconnected", "eoferror",
)


def extract_media(content: str) -> Tuple[List[Tuple[str, bool]], str]:
    """Extract MEDIA:<path> tags from response text.

    Returns: (list of (path, is_voice), cleaned content)
    """
    media = []
    has_voice = "[[audio_as_voice]]" in content
    cleaned = content.replace("[[audio_as_voice]]", "")

    pattern = re.compile(r'MEDIA:\s*(\S+)')
    for match in pattern.finditer(cleaned):
        path = match.group(1).strip("\"'`")
        if path:
            media.append((path, has_voice))

    if media:
        cleaned = pattern.sub("", cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    return media, cleaned


def extract_images(content: str) -> Tuple[List[Tuple[str, str]], str]:
    """Extract markdown image URLs from response text.

    Returns: (list of (url, alt_text), cleaned content)
    """
    images = []
    cleaned = content

    md_pattern = r'!\[([^\]]*)\]\((https?://[^\s\)]+)\)'
    for match in re.finditer(md_pattern, content):
        alt_text = match.group(1)
        url = match.group(2)
        if any(ext in url.lower() for ext in
               ('.png', '.jpg', '.jpeg', '.gif', '.webp', 'fal.media', 'replicate.delivery')):
            images.append((url, alt_text))

    if images:
        extracted_urls = {url for url, _ in images}
        cleaned = re.sub(
            md_pattern,
            lambda m: '' if m.group(2) in extracted_urls else m.group(0),
            cleaned,
        )
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    return images, cleaned


def is_animation_url(url: str) -> bool:
    """Check if URL points to an animated GIF."""
    return url.lower().split('?')[0].endswith('.gif')


def truncate_message(content: str, max_length: int = 4096) -> List[str]:
    """Split a long message into chunks, preserving code blocks.

    Delegates to message_splitting.split_message for fence-aware splitting.
    Adds (i/N) pagination indicators for multi-chunk results.
    """
    if len(content) <= max_length:
        return [content]
    from caveman.gateway.message_splitting import split_message
    # Reserve space for pagination indicator " (NN/NN)"
    chunks = split_message(content, max_length=max_length - 10)
    if len(chunks) > 1:
        total = len(chunks)
        chunks = [f"{chunk} ({i+1}/{total})" for i, chunk in enumerate(chunks)]
    return chunks

