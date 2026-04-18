"""Vision tool — prepare image payloads for LLM vision processing."""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}

_MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB


@tool(
    name="vision_describe",
    description="Describe an image",
    params={
        "image_path": {"type": "string", "description": "Path to the image file"},
        "question": {"type": "string", "description": "Question about the image", "default": "Describe this image in detail."},
    },
    required=["image_path"],
)
async def vision_describe(image_path: str, question: str = "Describe this image in detail.") -> dict:
    """Read an image, base64-encode it, and return a payload for LLM vision."""
    path = Path(image_path)
    if not path.exists():
        return {"error": f"Image not found: {image_path}"}
    suffix = path.suffix.lower()
    mime_type = _MIME_MAP.get(suffix)
    if not mime_type:
        return {"error": f"Unsupported image format: {suffix}"}
    file_size = path.stat().st_size
    if file_size > _MAX_IMAGE_SIZE:
        return {"error": f"Image too large: {file_size} bytes (max {_MAX_IMAGE_SIZE})"}
    try:
        raw = path.read_bytes()
    except OSError as e:
        return {"error": f"Failed to read image: {e}"}
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "ok": True,
        "image_size": len(raw),
        "mime_type": mime_type,
        "question": question,
        "base64_preview": b64[:100],
    }
