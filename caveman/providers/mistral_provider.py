"""Mistral provider — OpenAI-compatible API at api.mistral.ai."""
from __future__ import annotations
import os
from .openai_provider import OpenAIProvider


class MistralProvider(OpenAIProvider):
    """Mistral AI inference — inherits OpenAI-compatible interface."""

    def __init__(self, model: str = "mistral-large-latest", api_key: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("MISTRAL_API_KEY", ""),
            base_url="https://api.mistral.ai/v1",
            **kwargs,
        )
