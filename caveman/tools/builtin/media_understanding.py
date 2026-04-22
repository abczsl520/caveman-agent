"""Media Understanding — image, audio, and document analysis.

Provides media analysis capabilities for understanding attachments
in messages. Core patterns from OpenClaw src/media-understanding/ (10K LOC).
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

__all__ = [
    "MediaAnalysis",
    "analyze_image",
    "analyze_document",
    "detect_media_type",
]


logger = logging.getLogger("caveman.tools.media_understanding")


@dataclass
class MediaAnalysis:
    """Result of media analysis."""
    media_type: str  # image | audio | video | document
    description: str = ""
    text_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    error: str = ""


def analyze_image(
    image_path: str,
    question: str = "Describe this image in detail.",
    model: str = "",
) -> MediaAnalysis:
    """Analyze an image using vision AI."""
    path = Path(image_path)
    if not path.exists():
        return MediaAnalysis(media_type="image", error=f"File not found: {image_path}")

    # Read and encode
    data = path.read_bytes()
    if len(data) > 20 * 1024 * 1024:
        return MediaAnalysis(media_type="image", error="Image too large (>20MB)")

    b64 = base64.b64encode(data).decode("ascii")
    suffix = path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}
    mime = mime_map.get(suffix, "image/png")
    data_url = f"data:{mime};base64,{b64}"

    # Try OpenAI vision
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        return _analyze_openai_vision(data_url, question, model or "gpt-4o", api_key)

    return MediaAnalysis(
        media_type="image",
        error="No vision provider available (set OPENAI_API_KEY)",
    )


def _analyze_openai_vision(
    data_url: str, question: str, model: str, api_key: str,
) -> MediaAnalysis:
    """Analyze image via OpenAI vision."""
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "max_tokens": 2000,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return MediaAnalysis(
            media_type="image",
            description=text,
            provider="openai",
            metadata={"model": model},
        )
    except Exception as e:
        return MediaAnalysis(media_type="image", error=str(e), provider="openai")


def analyze_document(
    doc_path: str,
    question: str = "Summarize this document.",
) -> MediaAnalysis:
    """Analyze a document (PDF, text, etc.)."""
    path = Path(doc_path)
    if not path.exists():
        return MediaAnalysis(media_type="document", error=f"File not found: {doc_path}")

    suffix = path.suffix.lower()

    if suffix in (".txt", ".md", ".csv", ".json", ".yaml", ".yml"):
        try:
            content = path.read_text(encoding="utf-8")
            return MediaAnalysis(
                media_type="document",
                text_content=content[:50000],
                description=f"Text document ({len(content):,} chars)",
                metadata={"format": suffix, "size": path.stat().st_size},
            )
        except Exception as e:
            return MediaAnalysis(media_type="document", error=str(e))

    if suffix == ".pdf":
        return MediaAnalysis(
            media_type="document",
            description="PDF document (extraction requires PyPDF2 or pdfplumber)",
            metadata={"format": "pdf", "size": path.stat().st_size},
        )

    return MediaAnalysis(
        media_type="document",
        error=f"Unsupported document format: {suffix}",
    )


def detect_media_type(path: str) -> str:
    """Detect media type from file extension."""
    suffix = Path(path).suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
    audio_exts = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    doc_exts = {".pdf", ".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".docx"}

    if suffix in image_exts:
        return "image"
    if suffix in audio_exts:
        return "audio"
    if suffix in video_exts:
        return "video"
    if suffix in doc_exts:
        return "document"
    return "unknown"
