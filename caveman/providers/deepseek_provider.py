"""DeepSeek provider — OpenAI-compatible API at api.deepseek.com."""
from __future__ import annotations
import os
from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek inference — inherits OpenAI-compatible interface."""

    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1",
            **kwargs,
        )
