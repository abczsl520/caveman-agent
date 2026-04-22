"""Browser Depth — vision analysis, recording, screenshot cleanup, camofox.

Supplements browser_v2.py with vision AI analysis, session recording,
and anti-fingerprint browser support. Extracted from Hermes
browser_tool.py (2387 lines) + browser_camofox.py (592 lines).
"""
from __future__ import annotations

import base64
import logging
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "VisionConfig",
    "browser_vision",
    "save_screenshot",
    "cleanup_old_screenshots",
    "RecordingSession",
    "cleanup_old_recordings",
    "CamofoxProfile",
    "PRESET_PROFILES",
    "get_camofox_profile",
]


logger = logging.getLogger("caveman.tools.browser_depth")

_SCREENSHOTS_DIR = Path.home() / ".caveman" / "screenshots"
_RECORDINGS_DIR = Path.home() / ".caveman" / "recordings"
_last_cleanup: Dict[str, float] = {}


# ── Vision ──

@dataclass
class VisionConfig:
    """Configuration for browser vision analysis."""
    model: str = ""
    timeout: float = 120.0
    max_image_bytes: int = 20 * 1024 * 1024  # 20MB
    resize_target_bytes: int = 4 * 1024 * 1024  # 4MB


def browser_vision(
    screenshot_path: Path,
    question: str,
    annotate: bool = False,
    config: Optional[VisionConfig] = None,
    call_llm_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Analyze a browser screenshot with vision AI.

    Takes a screenshot path and sends it to a vision model for analysis.
    Useful for CAPTCHAs, visual verification, complex layouts.
    """
    config = config or VisionConfig()

    if not screenshot_path.exists():
        return {"success": False, "error": f"Screenshot not found: {screenshot_path}"}

    screenshot_bytes = screenshot_path.read_bytes()
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{screenshot_b64}"

    # Auto-resize if too large
    if len(screenshot_bytes) > config.max_image_bytes:
        resized = _resize_image(screenshot_path, config.resize_target_bytes)
        if resized:
            data_url = resized

    vision_prompt = (
        f"You are analyzing a screenshot of a web browser.\n\n"
        f"User's question: {question}\n\n"
        f"Provide a detailed and helpful answer based on what you see. "
        f"If there are interactive elements, describe them. "
        f"If there are verification challenges or CAPTCHAs, describe what type "
        f"they are and what action might be needed."
    )

    if not call_llm_fn:
        return {
            "success": True,
            "analysis": "[Vision LLM not configured — screenshot saved]",
            "screenshot_path": str(screenshot_path),
        }

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        response = call_llm_fn(
            messages=messages,
            max_tokens=2000,
            temperature=0.1,
            model=config.model or None,
        )
        analysis = ""
        if hasattr(response, "choices"):
            analysis = (response.choices[0].message.content or "").strip()
        elif isinstance(response, dict):
            analysis = response.get("text", response.get("content", ""))
        elif isinstance(response, str):
            analysis = response

        # Redact potential secrets from vision output
        analysis = _redact_secrets(analysis)

        result: Dict[str, Any] = {
            "success": True,
            "analysis": analysis or "Vision analysis returned no content.",
            "screenshot_path": str(screenshot_path),
        }
        return result

    except Exception as e:
        error_info: Dict[str, Any] = {
            "success": False,
            "error": f"Vision analysis failed: {e}",
        }
        if screenshot_path.exists():
            error_info["screenshot_path"] = str(screenshot_path)
            error_info["note"] = "Screenshot captured but vision failed. Share via MEDIA:<path>."
        return error_info


def _resize_image(path: Path, target_bytes: int) -> Optional[str]:
    """Resize image to fit within target bytes."""
    try:
        from PIL import Image
        import io
        img = Image.open(path)
        quality = 85
        while quality > 10:
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            if buf.tell() <= target_bytes:
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f"data:image/png;base64,{b64}"
            # Reduce size
            w, h = img.size
            img = img.resize((int(w * 0.8), int(h * 0.8)), Image.LANCZOS)
            quality -= 10
    except ImportError:
        logger.debug("PIL not available for image resize")
    except Exception as e:
        logger.debug("Image resize failed: %s", e)
    return None


_SECRET_RE = re.compile(
    r"(sk-[a-zA-Z0-9]{10,}|ghp_[a-zA-Z0-9]{30,}|xoxb-[a-zA-Z0-9-]+|"
    r"AKIA[A-Z0-9]{16}|AIza[a-zA-Z0-9_-]{35})",
)


def _redact_secrets(text: str) -> str:
    """Redact potential secrets from vision analysis output."""
    return _SECRET_RE.sub("[REDACTED]", text)


# ── Screenshots ──

def save_screenshot(
    screenshot_bytes: bytes,
    prefix: str = "browser_screenshot",
) -> Path:
    """Save a screenshot to the persistent screenshots directory."""
    _SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_screenshots()
    path = _SCREENSHOTS_DIR / f"{prefix}_{uuid.uuid4().hex}.png"
    path.write_bytes(screenshot_bytes)
    return path


def cleanup_old_screenshots(max_age_hours: int = 24) -> int:
    """Remove old screenshots. Throttled to once per hour."""
    key = str(_SCREENSHOTS_DIR)
    now = time.time()
    if now - _last_cleanup.get(key, 0) < 3600:
        return 0
    _last_cleanup[key] = now

    if not _SCREENSHOTS_DIR.exists():
        return 0
    cutoff = now - (max_age_hours * 3600)
    removed = 0
    for f in _SCREENSHOTS_DIR.glob("browser_screenshot_*.png"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass  # intentional: Exception suppressed
    return removed


# ── Recording ──

@dataclass
class RecordingSession:
    """A browser session recording."""
    session_id: str
    task_id: str = "default"
    started_at: float = 0
    frames: int = 0
    output_path: str = ""
    is_recording: bool = False

    def start(self) -> None:
        self.started_at = time.time()
        self.is_recording = True
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.output_path = str(
            _RECORDINGS_DIR / f"session_{self.session_id}_{int(self.started_at)}.webm"
        )

    def stop(self) -> Dict[str, Any]:
        self.is_recording = False
        duration = time.time() - self.started_at if self.started_at else 0
        return {
            "session_id": self.session_id,
            "duration_seconds": round(duration, 1),
            "frames": self.frames,
            "output_path": self.output_path,
        }


def cleanup_old_recordings(max_age_hours: int = 72) -> int:
    """Remove old recordings."""
    if not _RECORDINGS_DIR.exists():
        return 0
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in _RECORDINGS_DIR.glob("session_*.webm"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except Exception:
            pass  # intentional: Exception suppressed
    return removed


# ── Camofox (Anti-Fingerprint) ──

@dataclass
class CamofoxProfile:
    """Anti-fingerprint browser profile."""
    profile_id: str
    user_agent: str = ""
    viewport_width: int = 1920
    viewport_height: int = 1080
    timezone: str = "America/New_York"
    locale: str = "en-US"
    webgl_vendor: str = ""
    webgl_renderer: str = ""
    canvas_noise: float = 0.01
    audio_noise: float = 0.001

    def to_launch_args(self) -> List[str]:
        """Convert profile to browser launch arguments."""
        args = [
            f"--user-agent={self.user_agent}" if self.user_agent else "",
            f"--window-size={self.viewport_width},{self.viewport_height}",
        ]
        return [a for a in args if a]


PRESET_PROFILES = {
    "windows-chrome": CamofoxProfile(
        profile_id="windows-chrome",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        webgl_vendor="Google Inc. (NVIDIA)",
        webgl_renderer="ANGLE (NVIDIA, NVIDIA GeForce RTX 3060)",
    ),
    "mac-safari": CamofoxProfile(
        profile_id="mac-safari",
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        webgl_vendor="Apple",
        webgl_renderer="Apple M2",
    ),
    "linux-firefox": CamofoxProfile(
        profile_id="linux-firefox",
        user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
        webgl_vendor="Mesa",
        webgl_renderer="Mesa Intel(R) UHD Graphics 630",
    ),
}


def get_camofox_profile(name: str = "windows-chrome") -> CamofoxProfile:
    """Get a preset anti-fingerprint profile."""
    return PRESET_PROFILES.get(name, PRESET_PROFILES["windows-chrome"])
