"""Groq provider — OpenAI-compatible API at api.groq.com."""
from __future__ import annotations
import os
from .openai_provider import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Groq cloud inference — inherits OpenAI-compatible interface."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str | None = None, **kwargs):
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )
