"""Image generation tool — generate/edit images via DALL-E or Stability AI."""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path.home() / ".caveman" / "generated_images"


@tool(
    name="image_generate",
    description="Generate an image from text prompt",
    params={
        "prompt": {"type": "string", "description": "Text prompt for image generation"},
        "model": {"type": "string", "description": "Model: dall-e-3, dall-e-2, stable-*", "default": "dall-e-3"},
        "size": {"type": "string", "description": "Image size", "default": "1024x1024"},
    },
    required=["prompt"],
)
async def image_generate(prompt: str, model: str = "dall-e-3", size: str = "1024x1024") -> dict:
    """Generate an image from a text prompt."""
    import httpx

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if model.startswith("stable-"):
            return await _generate_stability(prompt, model, size)
        return await _generate_dalle(prompt, model, size)
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"API request failed: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def _generate_dalle(prompt: str, model: str, size: str) -> dict:
    import httpx
    import base64

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "OPENAI_API_KEY not set"}

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "prompt": prompt, "size": size, "n": 1, "response_format": "b64_json"},
        )
        resp.raise_for_status()
        data = resp.json()

    img_data = data["data"][0]
    b64 = img_data["b64_json"]
    revised = img_data.get("revised_prompt", prompt)

    ts = int(time.time())
    h = hashlib.md5(prompt.encode()).hexdigest()[:8]
    out_path = _OUTPUT_DIR / f"{ts}_{h}.png"
    out_path.write_bytes(base64.b64decode(b64))

    return {"ok": True, "path": str(out_path), "url": "", "revised_prompt": revised}


async def _generate_stability(prompt: str, model: str, size: str) -> dict:
    import httpx
    import base64

    api_key = os.environ.get("STABILITY_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "STABILITY_API_KEY not set"}

    w, h_val = (int(x) for x in size.split("x"))
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"text_prompts": [{"text": prompt}], "width": w, "height": h_val, "samples": 1},
        )
        resp.raise_for_status()
        data = resp.json()

    b64 = data["artifacts"][0]["base64"]
    ts = int(time.time())
    h = hashlib.md5(prompt.encode()).hexdigest()[:8]
    out_path = _OUTPUT_DIR / f"{ts}_{h}.png"
    out_path.write_bytes(base64.b64decode(b64))

    return {"ok": True, "path": str(out_path), "url": "", "revised_prompt": prompt}


@tool(
    name="image_edit",
    description="Edit an image with instructions",
    params={
        "image_path": {"type": "string", "description": "Path to the image to edit"},
        "prompt": {"type": "string", "description": "Edit instructions"},
        "model": {"type": "string", "description": "Model to use", "default": "dall-e-2"},
    },
    required=["image_path", "prompt"],
)
async def image_edit(image_path: str, prompt: str, model: str = "dall-e-2") -> dict:
    """Edit an image using DALL-E edit API."""
    import httpx
    import base64

    path = Path(image_path)
    if not path.exists():
        return {"ok": False, "error": f"Image not found: {image_path}"}

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "OPENAI_API_KEY not set"}

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        img_bytes = path.read_bytes()
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"image": (path.name, img_bytes, "image/png")},
                data={"prompt": prompt, "model": model, "n": 1, "response_format": "b64_json"},
            )
            resp.raise_for_status()
            data = resp.json()

        b64 = data["data"][0]["b64_json"]
        ts = int(time.time())
        h = hashlib.md5(prompt.encode()).hexdigest()[:8]
        out_path = _OUTPUT_DIR / f"{ts}_{h}_edit.png"
        out_path.write_bytes(base64.b64decode(b64))

        return {"ok": True, "path": str(out_path)}
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"API request failed: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
