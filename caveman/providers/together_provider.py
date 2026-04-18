"""Together AI provider — OpenAI-compatible API at api.together.xyz."""
from __future__ import annotations
import os
from .openai_provider import OpenAIProvider


class TogetherProvider(OpenAIProvider):
    """Together AI inference — inherits OpenAI-compatible interface."""

    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo", api_key: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
            **kwargs,
        )
