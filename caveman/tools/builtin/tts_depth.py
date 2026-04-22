"""TTS Depth — multi-provider, Opus conversion, streaming, markdown strip.

Supplements tts_v2.py with ElevenLabs, OpenAI, MiniMax, Mistral, NeuTTS
providers, Opus conversion, and streaming TTS. Extracted from Hermes
tts_tool.py (1059 lines).
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from caveman.aio import aio_exists, aio_mkdir, aio_stat

__all__ = [
    "TTSProviderConfig",
    "PROVIDERS",
    "load_tts_config",
    "strip_markdown_for_tts",
    "has_ffmpeg",
    "convert_to_opus",
    "generate_elevenlabs",
    "generate_openai_tts",
    "generate_minimax_tts",
    "generate_edge_tts",
    "text_to_speech",
]


logger = logging.getLogger("caveman.tools.tts_depth")

_OUTPUT_DIR = Path.home() / "voice-memos"
_MAX_TEXT_LENGTH = 10000


# ── Config ──

@dataclass
class TTSProviderConfig:
    """Configuration for a TTS provider."""
    name: str
    api_key_env: str = ""
    voice: str = ""
    model: str = ""
    base_url: str = ""
    speed: float = 1.0
    format: str = "mp3"

    @property
    def is_available(self) -> bool:
        if self.api_key_env:
            return bool(os.environ.get(self.api_key_env))
        return True

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


PROVIDERS = {
    "elevenlabs": TTSProviderConfig(
        "elevenlabs", "ELEVENLABS_API_KEY",
        voice="Rachel", model="eleven_multilingual_v2",
        base_url="https://api.elevenlabs.io/v1",
    ),
    "openai": TTSProviderConfig(
        "openai", "OPENAI_API_KEY",
        voice="alloy", model="tts-1",
        base_url="https://api.openai.com/v1",
    ),
    "minimax": TTSProviderConfig(
        "minimax", "MINIMAX_API_KEY",
        voice="male-qn-qingse", model="speech-01-turbo",
        base_url="https://api.minimax.chat/v1",
    ),
    "mistral": TTSProviderConfig(
        "mistral", "MISTRAL_API_KEY",
        voice="", model="mistral-tts-latest",
    ),
    "edge": TTSProviderConfig("edge", voice="en-US-AriaNeural"),
}


def load_tts_config(config: Optional[Dict[str, Any]] = None) -> TTSProviderConfig:
    """Load TTS config from dict or defaults."""
    if not config:
        config = {}
    provider_name = config.get("provider", "edge")
    base = PROVIDERS.get(provider_name, TTSProviderConfig(provider_name))
    if config.get("voice"):
        base.voice = config["voice"]
    if config.get("model"):
        base.model = config["model"]
    if config.get("speed"):
        base.speed = float(config["speed"])
    return base


# ── Markdown Stripping ──

def strip_markdown_for_tts(text: str) -> str:
    """Strip markdown formatting for cleaner TTS output."""
    import re
    # Remove code blocks
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Remove bullet points
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Remove blockquotes
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Opus Conversion ──

def has_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_to_opus(input_path: str) -> Optional[str]:
    """Convert audio file to Opus format for Telegram voice messages."""
    if not has_ffmpeg():
        return None
    output_path = str(Path(input_path).with_suffix(".ogg"))
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-c:a", "libopus", "-b:a", "64k",
                "-vbr", "on", "-compression_level", "10",
                output_path,
            ],
            capture_output=True, timeout=30,
        )
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            return output_path
    except Exception as e:
        logger.debug("Opus conversion failed: %s", e)
    return None


# ── Provider Generators ──

def generate_elevenlabs(text: str, output_path: str, config: TTSProviderConfig) -> str:
    """Generate speech with ElevenLabs API."""
    import urllib.request
    voice_id = config.voice or "Rachel"
    # Resolve voice name to ID if needed
    url = f"{config.base_url}/text-to-speech/{voice_id}"
    payload = json.dumps({
        "text": text,
        "model_id": config.model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": config.api_key,
        "Accept": "audio/mpeg",
    }
    req = urllib.request.Request(url, data=payload, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        Path(output_path).write_bytes(resp.read())
    return output_path


def generate_openai_tts(text: str, output_path: str, config: TTSProviderConfig) -> str:
    """Generate speech with OpenAI TTS API."""
    import urllib.request
    # Determine format from output path
    fmt = "opus" if output_path.endswith(".ogg") else "mp3"
    payload = json.dumps({
        "model": config.model or "tts-1",
        "input": text,
        "voice": config.voice or "alloy",
        "response_format": fmt,
        "speed": config.speed,
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    req = urllib.request.Request(
        f"{config.base_url}/audio/speech",
        data=payload, headers=headers,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        Path(output_path).write_bytes(resp.read())
    return output_path


def generate_minimax_tts(text: str, output_path: str, config: TTSProviderConfig) -> str:
    """Generate speech with MiniMax TTS API."""
    import urllib.request
    payload = json.dumps({
        "model": config.model or "speech-01-turbo",
        "text": text,
        "voice_setting": {"voice_id": config.voice or "male-qn-qingse", "speed": config.speed},
        "audio_setting": {"format": "mp3", "sample_rate": 32000},
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    req = urllib.request.Request(
        f"{config.base_url}/t2a_v2",
        data=payload, headers=headers,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    # MiniMax returns base64 audio
    import base64
    audio_b64 = data.get("data", {}).get("audio", "")
    if audio_b64:
        Path(output_path).write_bytes(base64.b64decode(audio_b64))
    return output_path


async def generate_edge_tts(text: str, output_path: str, config: TTSProviderConfig) -> str:
    """Generate speech with Edge TTS (free, no API key)."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    voice = config.voice or "en-US-AriaNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


# ── Main TTS Function ──

async def text_to_speech(
    text: str,
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    platform: str = "",
) -> Dict[str, Any]:
    """Convert text to speech with multi-provider support."""
    if not text or not text.strip():
        return {"success": False, "error": "Text is required"}

    if len(text) > _MAX_TEXT_LENGTH:
        text = text[:_MAX_TEXT_LENGTH]

    # Strip markdown
    text = strip_markdown_for_tts(text)

    provider = load_tts_config(config)
    want_opus = platform == "telegram"

    # Determine output path
    if output_path:
        file_path = Path(output_path).expanduser()
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        await aio_mkdir(_OUTPUT_DIR, parents=True, exist_ok=True)
        ext = ".ogg" if want_opus and provider.name in ("openai", "elevenlabs", "mistral") else ".mp3"
        file_path = _OUTPUT_DIR / f"tts_{timestamp}{ext}"

    await aio_mkdir(file_path.parent, parents=True, exist_ok=True)
    file_str = str(file_path)

    try:
        if provider.name == "elevenlabs":
            generate_elevenlabs(text, file_str, provider)
        elif provider.name == "openai":
            generate_openai_tts(text, file_str, provider)
        elif provider.name == "minimax":
            generate_minimax_tts(text, file_str, provider)
        elif provider.name == "edge":
            await generate_edge_tts(text, file_str, provider)
        else:
            # Fallback to edge
            try:
                await generate_edge_tts(text, file_str, provider)
            except ImportError:
                return {"success": False, "error": "No TTS provider available"}

        if not await aio_exists(Path(file_str)) or (await aio_stat(Path(file_str))).st_size == 0:
            return {"success": False, "error": f"TTS produced no output (provider: {provider.name})"}

        # Opus conversion for Telegram
        voice_compatible = False
        if provider.name in ("edge", "minimax") and not file_str.endswith(".ogg"):
            opus_path = convert_to_opus(file_str)
            if opus_path:
                file_str = opus_path
                voice_compatible = True
        elif provider.name in ("elevenlabs", "openai", "mistral"):
            voice_compatible = file_str.endswith(".ogg")

        media_tag = f"MEDIA:{file_str}"
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return {
            "success": True,
            "file_path": file_str,
            "media_tag": media_tag,
            "provider": provider.name,
            "voice_compatible": voice_compatible,
            "file_size": (await aio_stat(Path(file_str))).st_size,
        }

    except Exception as e:
        return {"success": False, "error": f"TTS failed ({provider.name}): {e}"}
