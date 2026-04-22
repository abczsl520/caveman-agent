"""Image Generation — text-to-image via multiple providers.

Provides image generation via DALL-E, Stable Diffusion, and other
providers. Extracted from Hermes tools/image_generation_tool.py (703 lines).
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "ImageGenConfig",
    "GeneratedImage",
    "ImageGenResult",
    "generate_image",
    "check_image_gen_available",
]


logger = logging.getLogger("caveman.tools.image_generation")

_OUTPUT_DIR = Path.home() / ".caveman" / "generated_images"


@dataclass
class ImageGenConfig:
    """Configuration for image generation."""
    provider: str = "openai"  # openai | stability
    model: str = "dall-e-3"
    size: str = "1024x1024"
    quality: str = "standard"  # standard | hd
    style: str = "vivid"  # vivid | natural
    n: int = 1


@dataclass
class GeneratedImage:
    """A generated image."""
    url: str = ""
    local_path: str = ""
    revised_prompt: str = ""
    provider: str = ""
    model: str = ""
    size: str = ""


@dataclass
class ImageGenResult:
    """Result of image generation."""
    success: bool = False
    images: List[GeneratedImage] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0


def generate_image(
    prompt: str,
    config: Optional[ImageGenConfig] = None,
) -> ImageGenResult:
    """Generate an image from a text prompt."""
    config = config or ImageGenConfig()

    if not prompt or not prompt.strip():
        return ImageGenResult(error="Prompt is required")

    start = time.monotonic()

    if config.provider == "openai":
        result = _generate_openai(prompt, config)
    else:
        result = ImageGenResult(error=f"Unknown provider: {config.provider}")

    result.duration_ms = (time.monotonic() - start) * 1000
    return result


def _generate_openai(prompt: str, config: ImageGenConfig) -> ImageGenResult:
    """Generate via OpenAI DALL-E API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return ImageGenResult(error="OPENAI_API_KEY not set")

    import urllib.request

    payload = json.dumps({
        "model": config.model,
        "prompt": prompt,
        "n": config.n,
        "size": config.size,
        "quality": config.quality,
        "style": config.style,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/images/generations",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        images = []
        for item in data.get("data", []):
            img = GeneratedImage(
                url=item.get("url", ""),
                revised_prompt=item.get("revised_prompt", ""),
                provider="openai",
                model=config.model,
                size=config.size,
            )

            # Download to local
            if img.url:
                local_path = _download_image(img.url, config.model)
                if local_path:
                    img.local_path = str(local_path)

            images.append(img)

        return ImageGenResult(success=True, images=images)

    except Exception as e:
        return ImageGenResult(error=f"OpenAI image generation failed: {e}")


def _download_image(url: str, model: str) -> Optional[Path]:
    """Download a generated image to local storage."""
    import urllib.request
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{model}_{int(time.time())}.png"
    path = _OUTPUT_DIR / filename
    try:
        urllib.request.urlretrieve(url, str(path))
        return path
    except Exception as e:
        logger.debug("Failed to download image: %s", e)
        return None


def check_image_gen_available() -> Dict[str, bool]:
    """Check which image generation providers are available."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }
