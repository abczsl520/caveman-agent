"""Transcription Tools — audio/video to text conversion.

Provides transcription via Whisper API and local fallbacks.
Extracted from Hermes tools/transcription_tools.py (708 lines).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "TranscriptionConfig",
    "TranscriptionResult",
    "transcribe",
    "check_transcription_available",
]


logger = logging.getLogger("caveman.tools.transcription")

_SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
_MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (Whisper API limit)


@dataclass
class TranscriptionConfig:
    """Configuration for transcription."""
    provider: str = "openai"  # openai | local
    model: str = "whisper-1"
    language: str = ""
    prompt: str = ""
    temperature: float = 0.0


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    success: bool = False
    text: str = ""
    language: str = ""
    duration_seconds: float = 0
    segments: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    error: str = ""


def transcribe(
    audio_path: str,
    config: Optional[TranscriptionConfig] = None,
) -> TranscriptionResult:
    """Transcribe an audio file."""
    config = config or TranscriptionConfig()
    path = Path(audio_path)

    if not path.exists():
        return TranscriptionResult(error=f"File not found: {audio_path}")

    if path.suffix.lower() not in _SUPPORTED_FORMATS:
        return TranscriptionResult(
            error=f"Unsupported format: {path.suffix}. Supported: {_SUPPORTED_FORMATS}",
        )

    if path.stat().st_size > _MAX_FILE_SIZE:
        return TranscriptionResult(
            error=f"File too large: {path.stat().st_size:,} bytes (max: {_MAX_FILE_SIZE:,})",
        )

    if config.provider == "openai":
        return _transcribe_openai(path, config)

    return TranscriptionResult(error=f"Unknown provider: {config.provider}")


def _transcribe_openai(path: Path, config: TranscriptionConfig) -> TranscriptionResult:
    """Transcribe via OpenAI Whisper API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return TranscriptionResult(error="OPENAI_API_KEY not set")

    import urllib.request
    import io

    try:
        # Build multipart form data
        boundary = "----CavemanBoundary"
        body = io.BytesIO()

        # File field
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode())
        body.write(f"Content-Type: application/octet-stream\r\n\r\n".encode())
        body.write(path.read_bytes())
        body.write(b"\r\n")

        # Model field
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="model"\r\n\r\n')
        body.write(f"{config.model}\r\n".encode())

        # Response format
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="response_format"\r\n\r\n')
        body.write(b"verbose_json\r\n")

        if config.language:
            body.write(f"--{boundary}\r\n".encode())
            body.write(b'Content-Disposition: form-data; name="language"\r\n\r\n')
            body.write(f"{config.language}\r\n".encode())

        body.write(f"--{boundary}--\r\n".encode())

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/audio/transcriptions",
            data=body.getvalue(),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        return TranscriptionResult(
            success=True,
            text=data.get("text", ""),
            language=data.get("language", ""),
            duration_seconds=data.get("duration", 0),
            segments=data.get("segments", []),
            provider="openai",
        )

    except Exception as e:
        return TranscriptionResult(error=f"OpenAI transcription failed: {e}", provider="openai")


def check_transcription_available() -> Dict[str, bool]:
    """Check which transcription providers are available."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }
