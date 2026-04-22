"""Vision tool — prepare image payloads for LLM vision processing."""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from caveman.tools.registry import tool
from caveman.aio import aio_exists, aio_read_bytes, aio_stat

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


async def _download_image(url: str) -> tuple[bytes, str]:
    """Download image from URL. Returns (data, mime_type)."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status}")
            ct = resp.content_type or ""
            data = await resp.read()
            if len(data) > _MAX_IMAGE_SIZE:
                raise ValueError(f"Image too large: {len(data)} bytes")
            # Infer mime from content-type or URL
            if ct.startswith("image/"):
                return data, ct
            for ext, mime in _MIME_MAP.items():
                if url.lower().endswith(ext):
                    return data, mime
            return data, "image/jpeg"  # default


@tool(
    name="vision_describe",
    description="Describe an image from a file path or URL",
    params={
        "image_path": {"type": "string", "description": "Path to image file or URL"},
        "question": {"type": "string", "description": "Question about the image",
                     "default": "Describe this image in detail."},
    },
    required=["image_path"],
)
async def vision_describe(image_path: str, question: str = "Describe this image in detail.") -> dict:
    """Read an image (local or URL), base64-encode it, return payload for LLM vision."""
    # URL handling
    if image_path.startswith(("http://", "https://")):
        try:
            raw, mime_type = await _download_image(image_path)
        except Exception as e:
            return {"error": f"Failed to download image: {e}"}
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "ok": True,
            "image_size": len(raw),
            "mime_type": mime_type,
            "question": question,
            "source": "url",
            "base64_preview": b64[:100],
        }

    # Local file handling
    path = Path(image_path)
    if not await aio_exists(path):
        return {"error": f"Image not found: {image_path}"}
    suffix = path.suffix.lower()
    mime_type = _MIME_MAP.get(suffix)
    if not mime_type:
        return {"error": f"Unsupported image format: {suffix}"}
    file_size = (await aio_stat(path)).st_size
    if file_size > _MAX_IMAGE_SIZE:
        return {"error": f"Image too large: {file_size} bytes (max {_MAX_IMAGE_SIZE})"}
    try:
        raw = await aio_read_bytes(path)
    except OSError as e:
        return {"error": f"Failed to read image: {e}"}
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "ok": True,
        "image_size": len(raw),
        "mime_type": mime_type,
        "question": question,
        "source": "file",
        "base64_preview": b64[:100],
    }
