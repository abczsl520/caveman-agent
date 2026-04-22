"""Voice mode — push-to-talk audio recording and STT/TTS pipeline.

Provides:
- Audio capture via sounddevice (optional dependency)
- WAV encoding via stdlib wave
- STT dispatch via transcription tools
- TTS playback via system audio or sounddevice

Dependencies (optional): pip install sounddevice numpy
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any
from caveman.aio import aio_unlink

__all__ = [
    "VoiceRecorder",
    "AudioPlayer",
    "VoiceMode",
]


logger = logging.getLogger(__name__)


def _audio_available() -> bool:
    """Check if audio libraries are available."""
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def _get_install_hint() -> str:
    """Get installation hint for audio dependencies."""
    return "pip install sounddevice numpy"


class VoiceRecorder:
    """Push-to-talk audio recorder."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames: list = []
        self._stream = None

    def start(self) -> None:
        """Start recording audio."""
        if not _audio_available():
            raise RuntimeError(f"Audio not available. Install: {_get_install_hint()}")

        import sounddevice as sd

        self._frames = []
        self._recording = True

        def callback(indata, frames, time_info, status) -> None:
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop(self) -> Path:
        """Stop recording and save to WAV file.

        Returns path to the WAV file.
        """
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        import numpy as np

        if not self._frames:
            raise RuntimeError("No audio recorded")

        audio_data = np.concatenate(self._frames)
        output = Path(tempfile.mktemp(suffix=".wav"))

        with wave.open(str(output), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

        logger.info("Recording saved: %s (%.1fs)", output, len(audio_data) / self.sample_rate)
        return output

    @property
    def is_recording(self) -> bool:
        return self._recording


class AudioPlayer:
    """Cross-platform audio playback."""

    @staticmethod
    def play(audio_path: Path) -> None:
        """Play an audio file using the best available method."""
        path_str = str(audio_path)

        # Try sounddevice first
        if _audio_available():
            try:
                AudioPlayer._play_sounddevice(audio_path)
                return
            except Exception as e:
                logger.debug("sounddevice playback failed: %s", e)

        # Fallback to system commands
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["afplay", path_str], check=True)
        elif system == "Linux":
            if shutil.which("aplay"):
                subprocess.run(["aplay", path_str], check=True)
            elif shutil.which("paplay"):
                subprocess.run(["paplay", path_str], check=True)
            elif shutil.which("ffplay"):
                subprocess.run(["ffplay", "-nodisp", "-autoexit", path_str],
                             capture_output=True, check=True)
        elif system == "Windows":
            # Use PowerShell
            subprocess.run(
                ["powershell", "-c", f"(New-Object Media.SoundPlayer '{path_str}').PlaySync()"],
                check=True,
            )
        else:
            raise RuntimeError(f"No audio player available on {system}")

    @staticmethod
    def _play_sounddevice(audio_path: Path) -> None:
        """Play audio using sounddevice."""
        import sounddevice as sd
        import numpy as np

        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16)
            if channels > 1:
                data = data.reshape(-1, channels)

        sd.play(data, samplerate=sample_rate)
        sd.wait()


class VoiceMode:
    """Full voice interaction pipeline: Record → STT → Agent → TTS → Play."""

    def __init__(self) -> None:
        self.recorder = VoiceRecorder()
        self.player = AudioPlayer()
        self._active = False

    @property
    def available(self) -> bool:
        return _audio_available()

    def check_requirements(self) -> dict[str, Any]:
        """Check voice mode requirements."""
        return {
            "audio_libs": _audio_available(),
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "install_hint": _get_install_hint() if not _audio_available() else None,
        }

    async def record_and_transcribe(self, duration: float = 10.0) -> str:
        """Record audio for duration seconds and transcribe.

        Returns transcribed text.
        """
        self.recorder.start()
        await _async_sleep(duration)
        wav_path = self.recorder.stop()

        try:
            text = await self._transcribe(wav_path)
            return text
        finally:
            await aio_unlink(wav_path, missing_ok=True)

    async def speak(self, text: str, provider: str = "edge", voice: str = "") -> None:
        """Synthesize and play speech."""
        from caveman.tools.builtin.tts_tool import synthesize, TTSProvider

        try:
            prov = TTSProvider(provider)
        except ValueError:
            prov = TTSProvider.EDGE

        audio_path = await synthesize(text, provider=prov, voice=voice or None)
        self.player.play(audio_path)
        await aio_unlink(audio_path, missing_ok=True)

    async def _transcribe(self, wav_path: Path) -> str:
        """Transcribe audio using available STT."""
        # Try Whisper API
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            return await self._whisper_transcribe(wav_path, api_key)

        # Fallback: local whisper
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(wav_path))
            return result.get("text", "")
        except ImportError:
            raise RuntimeError("No STT available. Set OPENAI_API_KEY or install whisper")

    async def _whisper_transcribe(self, wav_path: Path, api_key: str) -> str:
        """Transcribe using OpenAI Whisper API."""
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            with open(wav_path, "rb") as f:
                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": ("audio.wav", f, "audio/wav")},
                    data={"model": "whisper-1"},
                )
                resp.raise_for_status()
                return resp.json().get("text", "")


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)
