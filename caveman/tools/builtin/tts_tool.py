"""Text-to-Speech — multi-provider TTS with format conversion.

Supported providers:
- Edge TTS (default, free): Microsoft Edge neural voices
- OpenAI TTS: High quality, needs OPENAI_API_KEY
- ElevenLabs: Premium voices, needs ELEVENLABS_API_KEY
- Fish Audio: Via proxy, needs FISH_AUDIO_KEY

Output formats:
- MP3 (.mp3) for general use
- OGG/Opus (.ogg) for Telegram voice bubbles (requires ffmpeg)

Configuration via config.yaml under 'tts:' key or environment variables.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from enum import Enum
from pathlib import Path

from caveman.paths import CAVEMAN_HOME
from caveman.tools.registry import tool
from caveman.aio import aio_mkdir, aio_write_bytes

logger = logging.getLogger(__name__)

_TTS_CACHE_DIR = CAVEMAN_HOME / "cache" / "tts"


class TTSProvider(str, Enum):
    """Available text-to-speech provider backends."""
    EDGE = "edge"
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    FISH = "fish"


# Default voices per provider
_DEFAULT_VOICES = {
    TTSProvider.EDGE: "en-US-AriaNeural",
    TTSProvider.OPENAI: "alloy",
    TTSProvider.ELEVENLABS: "Rachel",
    TTSProvider.FISH: "default",
}


async def synthesize(
    text: str,
    provider: TTSProvider = TTSProvider.EDGE,
    voice: str | None = None,
    output_format: str = "mp3",
    output_path: Path | None = None,
) -> Path:
    """Synthesize speech from text.

    Args:
        text: Text to speak.
        provider: TTS provider to use.
        voice: Voice name (provider-specific). Uses default if None.
        output_format: "mp3" or "ogg".
        output_path: Output file path. Auto-generated if None.

    Returns: Path to the generated audio file.
    """
    text = strip_markdown_for_tts(text)
    voice = voice or _DEFAULT_VOICES.get(provider, "default")

    if output_path is None:
        await aio_mkdir(_TTS_CACHE_DIR, parents=True, exist_ok=True)
        output_path = _TTS_CACHE_DIR / f"tts_{hash(text[:50])}_{provider.value}.{output_format}"

    if provider == TTSProvider.EDGE:
        raw_path = await _edge_tts(text, voice)
    elif provider == TTSProvider.OPENAI:
        raw_path = await _openai_tts(text, voice)
    elif provider == TTSProvider.ELEVENLABS:
        raw_path = await _elevenlabs_tts(text, voice)
    elif provider == TTSProvider.FISH:
        raw_path = await _fish_tts(text, voice)
    else:
        raise ValueError(f"Unknown TTS provider: {provider}")

    # Convert format if needed
    if output_format == "ogg" and not str(raw_path).endswith(".ogg"):
        raw_path = _convert_to_ogg(raw_path)

    if raw_path != output_path:
        shutil.move(str(raw_path), str(output_path))

    return output_path


async def _edge_tts(text: str, voice: str) -> Path:
    """Generate speech using Edge TTS (free, no API key)."""
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

    output = Path(tempfile.mktemp(suffix=".mp3"))
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output))
    return output


async def _openai_tts(text: str, voice: str) -> Path:
    """Generate speech using OpenAI TTS API."""
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    output = Path(tempfile.mktemp(suffix=".mp3"))

    async with httpx.AsyncClient(timeout=HTTP_TTS) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "tts-1", "input": text, "voice": voice},
        )
        resp.raise_for_status()
        await aio_write_bytes(output, resp.content)

    return output


async def _elevenlabs_tts(text: str, voice: str) -> Path:
    """Generate speech using ElevenLabs API."""
    import httpx

    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    output = Path(tempfile.mktemp(suffix=".mp3"))

    # First resolve voice name to ID
    async with httpx.AsyncClient(timeout=HTTP_DEFAULT) as client:
        voices_resp = await client.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": api_key},
        )
        voices_resp.raise_for_status()
        voices = voices_resp.json().get("voices", [])
        voice_id = next((v["voice_id"] for v in voices if v["name"].lower() == voice.lower()), None)
        if not voice_id:
            voice_id = voice  # Assume it's already an ID

        resp = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key},
            json={"text": text, "model_id": "eleven_monolingual_v1"},
        )
        resp.raise_for_status()
        await aio_write_bytes(output, resp.content)

    return output


async def _fish_tts(text: str, voice: str) -> Path:
    """Generate speech using Fish Audio API."""
    import httpx

    api_key = os.environ.get("FISH_AUDIO_KEY", "")
    api_base = os.environ.get("FISH_AUDIO_URL", "https://api.fish.audio")
    if not api_key:
        raise RuntimeError("FISH_AUDIO_KEY not set")

    output = Path(tempfile.mktemp(suffix=".mp3"))

    async with httpx.AsyncClient(timeout=HTTP_TTS) as client:
        resp = await client.post(
            f"{api_base}/v1/tts",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"text": text, "reference_id": voice},
        )
        resp.raise_for_status()
        await aio_write_bytes(output, resp.content)

    return output


def _convert_to_ogg(input_path: Path) -> Path:
    """Convert audio to OGG/Opus format using ffmpeg."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found, returning original format")
        return input_path

    output = input_path.with_suffix(".ogg")
    subprocess.run(
        ["ffmpeg", "-i", str(input_path), "-c:a", "libopus", "-b:a", "64k",
         "-y", str(output)],
        capture_output=True, check=True,
    )
    input_path.unlink(missing_ok=True)
    return output


def list_edge_voices(locale: str = "") -> list[dict[str, str]]:
    """List available Edge TTS voices, optionally filtered by locale."""
    try:
        import edge_tts
        voices = asyncio.run(edge_tts.list_voices())
        if locale:
            voices = [v for v in voices if locale.lower() in v.get("Locale", "").lower()]
        return [{"name": v["ShortName"], "locale": v["Locale"], "gender": v.get("Gender", "")}
                for v in voices]
    except ImportError:
        return []


@tool(
    name="text_to_speech",
    description="Convert text to speech audio. Supports Edge TTS (free), OpenAI, ElevenLabs, Fish Audio.",
    params={
        "text": {"type": "string", "description": "Text to convert to speech"},
        "provider": {"type": "string", "description": "TTS provider: edge/openai/elevenlabs/fish"},
        "voice": {"type": "string", "description": "Voice name (provider-specific)"},
        "format": {"type": "string", "description": "Output format: mp3 or ogg"},
    },
    required=["text"],
)
async def text_to_speech_tool(
    text: str,
    provider: str = "edge",
    voice: str = "",
    format: str = "mp3",
) -> str:
    """TTS tool for agent use."""
    try:
        prov = TTSProvider(provider)
    except ValueError:
        return f"Unknown provider: {provider}. Use: edge, openai, elevenlabs, fish"

    try:
        path = await synthesize(text, provider=prov, voice=voice or None, output_format=format)
        return f"Audio saved to: {path}"
    except Exception as e:
        return f"TTS error: {e}"
from caveman.tools.builtin.tts_depth import (  # noqa: F401  # depth wiring
    TTSProviderConfig, PROVIDERS, load_tts_config, strip_markdown_for_tts,
    has_ffmpeg, convert_to_opus, generate_minimax_tts, generate_elevenlabs,
    generate_openai_tts, generate_edge_tts, text_to_speech,
)
from caveman.timeouts import HTTP_DEFAULT, HTTP_TTS

__all__ = [
    "TTSProvider",
    "synthesize",
    "list_edge_voices",
    "text_to_speech_tool",
    # depth re-exports
    "TTSProviderConfig",
    "PROVIDERS",
    "load_tts_config",
    "strip_markdown_for_tts",
    "has_ffmpeg",
    "convert_to_opus",
    "generate_minimax_tts",
    "generate_elevenlabs",
    "generate_openai_tts",
    "generate_edge_tts",
    "text_to_speech",
]

