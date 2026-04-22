"""Vision Tools — image analysis and processing utilities.

Provides vision-related utilities for screenshot analysis,
image resizing, and multi-modal content handling.
Extracted from Hermes tools/vision_tools.py.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict

__all__ = [
    "is_image_size_error",
    "resize_image_for_vision",
    "image_to_data_url",
    "build_vision_message",
    "extract_image_urls",
]


logger = logging.getLogger("caveman.tools.vision")

_RESIZE_TARGET_BYTES = 4 * 1024 * 1024  # 4MB


def is_image_size_error(error: Exception) -> bool:
    """Check if an API error is related to image size."""
    msg = str(error).lower()
    return any(
        phrase in msg
        for phrase in ("image too large", "payload too large", "413", "content_too_large", "max_tokens")
    )


def resize_image_for_vision(
    image_path: Path,
    target_bytes: int = _RESIZE_TARGET_BYTES,
    mime_type: str = "image/png",
) -> str:
    """Resize an image to fit within target bytes, return as data URL."""
    try:
        from PIL import Image
        import io

        img = Image.open(image_path)
        fmt = "PNG" if "png" in mime_type else "JPEG"

        # Try progressively smaller sizes
        for scale in (1.0, 0.8, 0.6, 0.4, 0.25):
            w, h = img.size
            resized = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS,
            )
            buf = io.BytesIO()
            resized.save(buf, format=fmt, optimize=True)
            if buf.tell() <= target_bytes:
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f"data:{mime_type};base64,{b64}"

        # Last resort: very small
        resized = img.resize((640, 480), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format=fmt, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    except ImportError:
        logger.warning("PIL not available for image resize")
        # Return original as data URL
        data = image_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{b64}"


def image_to_data_url(image_path: Path, mime_type: str = "") -> str:
    """Convert an image file to a data URL."""
    if not mime_type:
        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/png")

    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def build_vision_message(
    text: str,
    image_urls: list[str],
) -> Dict[str, Any]:
    """Build a vision-capable message with text and images."""
    content = [{"type": "text", "text": text}]
    for url in image_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    return {"role": "user", "content": content}


def extract_image_urls(text: str) -> list[str]:
    """Extract image URLs from text (markdown images and raw URLs)."""
    urls = []
    # Markdown images: ![alt](url)
    for match in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', text):
        urls.append(match.group(2))
    # Raw image URLs
    for match in re.finditer(r'https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp)', text, re.IGNORECASE):
        url = match.group()
        if url not in urls:
            urls.append(url)
    return urls
