"""TTS Tool v2 — text-to-speech with multiple providers.

Extracted from Hermes tts_tool.py (1059 lines).
Supports: system TTS, OpenAI TTS, ElevenLabs, edge-tts.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from caveman.tools.registry import tool
from caveman.aio import aio_exists, aio_read_bytes, aio_unlink

__all__ = [
    "AUDIO_CACHE_DIR",
    "MAX_TEXT_LENGTH",
    "MAX_CACHE_SIZE_MB",
    "TTSConfig",
    "register_tts_provider",
    "tts_generate",
    "check_tts_requirements",
]


logger = logging.getLogger("caveman.tools.tts_v2")

AUDIO_CACHE_DIR = Path.home() / ".caveman" / "tts_cache"
MAX_TEXT_LENGTH = 4000
MAX_CACHE_SIZE_MB = 500


@dataclass
class TTSConfig:
    """TTS configuration."""
    provider: str = "system"  # system | openai | elevenlabs | edge
    voice: str = ""
    speed: float = 1.0
    max_chars: int = MAX_TEXT_LENGTH
    cache_enabled: bool = True
    auto_tts: bool = False  # Auto-TTS for all responses
    summary_mode: bool = False  # Summarize long text before TTS


# Provider registry
_providers: Dict[str, Any] = {}


def register_tts_provider(name: str, provider: Any) -> None:
    _providers[name] = provider


# ── Cache ──

def _cache_key(text: str, provider: str, voice: str) -> str:
    h = hashlib.sha256(f"{provider}:{voice}:{text}".encode()).hexdigest()[:16]
    return h


def _get_cached(key: str) -> Optional[str]:
    path = AUDIO_CACHE_DIR / f"{key}.mp3"
    if path.exists():
        return str(path)
    return None


def _save_cache(key: str, audio_data: bytes) -> str:
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_CACHE_DIR / f"{key}.mp3"
    path.write_bytes(audio_data)
    return str(path)


def _cleanup_cache(max_mb: int = MAX_CACHE_SIZE_MB) -> int:
    """Remove oldest cache files if over size limit."""
    if not AUDIO_CACHE_DIR.exists():
        return 0
    files = sorted(AUDIO_CACHE_DIR.iterdir(), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    removed = 0
    while total > max_mb * 1024 * 1024 and files:
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink(missing_ok=True)
        removed += 1
    return removed


# ── Providers ──

async def _tts_system(text: str, voice: str = "") -> Optional[bytes]:
    """System TTS (macOS say, espeak, etc.)."""
    import platform as _platform
    system = _platform.system()

    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        tmp_path = f.name

    try:
        if system == "Darwin":
            cmd = ["say"]
            if voice:
                cmd.extend(["-v", voice])
            cmd.extend(["-o", tmp_path, text[:MAX_TEXT_LENGTH]])
        elif system == "Linux":
            cmd = ["espeak"]
            if voice:
                cmd.extend(["-v", voice])
            cmd.extend(["--stdout", text[:MAX_TEXT_LENGTH]])
            # espeak writes to stdout, redirect
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
            return stdout if stdout else None
        else:
            return None

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)

        if await aio_exists(os.path, tmp_path):
            return await aio_read_bytes(Path(tmp_path))
        return None
    except Exception as e:
        logger.warning("System TTS failed: %s", e)
        return None
    finally:
        try:
            await aio_unlink(tmp_path)
        except Exception:
            pass  # intentional: Exception suppressed


async def _tts_edge(text: str, voice: str = "") -> Optional[bytes]:
    """Edge TTS (free, no API key needed)."""
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        logger.warning("edge-tts not installed")
        return None

    voice = voice or "en-US-AriaNeural"
    try:
        communicate = edge_tts.Communicate(text[:MAX_TEXT_LENGTH], voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        await communicate.save(tmp_path)
        data = await aio_read_bytes(Path(tmp_path))
        await aio_unlink(tmp_path)
        return data
    except Exception as e:
        logger.warning("Edge TTS failed: %s", e)
        return None


async def _tts_openai(text: str, voice: str = "", api_key: str = "") -> Optional[bytes]:
    """OpenAI TTS API."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None

    voice = voice or "alloy"
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "tts-1", "input": text[:MAX_TEXT_LENGTH], "voice": voice},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.content
            logger.warning("OpenAI TTS error: %s", resp.status_code)
            return None
    except Exception as e:
        logger.warning("OpenAI TTS failed: %s", e)
        return None


# ── Main Tool ──

@tool(
    name="tts",
    description="Convert text to speech audio",
    params={
        "text": {"type": "string", "description": "Text to convert to speech"},
        "provider": {"type": "string", "description": "TTS provider (system/edge/openai)"},
        "voice": {"type": "string", "description": "Voice name"},
    },
    required=["text"],
)
async def tts_generate(
    text: str, provider: str = "system", voice: str = "",
) -> Dict[str, Any]:
    """Generate speech from text."""
    if not text.strip():
        return {"ok": False, "error": "Empty text"}

    # Check cache
    key = _cache_key(text, provider, voice)
    cached = _get_cached(key)
    if cached:
        return {"ok": True, "path": cached, "cached": True}

    # Generate
    audio_data = None
    if provider == "system":
        audio_data = await _tts_system(text, voice)
    elif provider == "edge":
        audio_data = await _tts_edge(text, voice)
    elif provider == "openai":
        audio_data = await _tts_openai(text, voice)
    else:
        return {"ok": False, "error": f"Unknown provider: {provider}"}

    if not audio_data:
        return {"ok": False, "error": f"TTS generation failed with {provider}"}

    path = _save_cache(key, audio_data)
    return {"ok": True, "path": path, "size": len(audio_data), "cached": False}


def check_tts_requirements() -> Dict[str, bool]:
    """Check which TTS providers are available."""
    import platform as _platform
    result = {
        "system": _platform.system() in ("Darwin", "Linux"),
        "edge": False,
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }
    try:
        import edge_tts  # noqa: F401
        result["edge"] = True
    except ImportError:
        pass  # intentional: ImportError suppressed
    return result
