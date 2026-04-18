"""Transcription tool — audio/video to text via whisper CLI."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".mp4", ".webm"}
_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB


@tool(
    name="transcribe",
    description="Transcribe audio/video to text",
    params={
        "file_path": {"type": "string", "description": "Path to audio/video file"},
        "language": {"type": "string", "description": "Language code or 'auto'", "default": "auto"},
    },
    required=["file_path"],
)
async def transcribe(file_path: str, language: str = "auto") -> dict:
    """Transcribe audio/video file using whisper CLI."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_FORMATS:
        return {"error": f"Unsupported format: {suffix}. Supported: {', '.join(sorted(_SUPPORTED_FORMATS))}"}
    file_size = path.stat().st_size
    if file_size > _MAX_FILE_SIZE:
        return {"error": f"File too large: {file_size} bytes (max {_MAX_FILE_SIZE})"}

    # Check if whisper is available
    check = await asyncio.create_subprocess_exec(
        "which", "whisper",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await check.communicate()
    if check.returncode != 0:
        return {"error": "whisper CLI not found. Install with: pip install openai-whisper"}

    cmd = ["whisper", str(path), "--output_format", "txt"]
    if language != "auto":
        cmd.extend(["--language", language])

    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_b, stderr_b = await proc.communicate()
    duration = round(time.monotonic() - start, 3)

    if proc.returncode != 0:
        return {"error": f"whisper failed: {stderr_b.decode('utf-8', errors='replace')}"}

    # Whisper outputs a .txt file next to the input
    txt_path = path.with_suffix(".txt")
    if txt_path.exists():
        text = txt_path.read_text().strip()
    else:
        text = stdout_b.decode("utf-8", errors="replace").strip()

    return {"ok": True, "text": text, "language": language, "duration_s": duration}


@tool(
    name="transcribe_url",
    description="Transcribe audio/video from URL",
    params={
        "url": {"type": "string", "description": "URL of audio/video file"},
        "language": {"type": "string", "description": "Language code or 'auto'", "default": "auto"},
    },
    required=["url"],
)
async def transcribe_url(url: str, language: str = "auto") -> dict:
    """Download audio/video from URL and transcribe it."""
    import httpx

    tmp = None
    try:
        # Determine extension from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix.lower() or ".mp3"
        if ext not in _SUPPORTED_FORMATS:
            ext = ".mp3"

        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.close()

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            Path(tmp.name).write_bytes(resp.content)

        return await transcribe(tmp.name, language=language)
    except httpx.HTTPError as e:
        return {"error": f"Download failed: {e}"}
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
