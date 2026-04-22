"""Mixture-of-Agents — multi-model collaborative reasoning.

Based on "Mixture-of-Agents Enhances Large Language Model Capabilities"
(Wang et al., arXiv:2406.04692v1).

Architecture:
1. Reference models generate diverse responses in parallel
2. Aggregator model synthesizes into a high-quality output

Uses Caveman's provider system — works with any configured provider.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from caveman.tools.registry import tool
from caveman.timeouts import HTTP_SLOW

__all__ = [
    "DEFAULT_REFERENCE_MODELS",
    "DEFAULT_AGGREGATOR_MODEL",
    "AGGREGATOR_SYSTEM_PROMPT",
    "mixture_of_agents",
    "mixture_of_agents_tool",
]


logger = logging.getLogger(__name__)

# Default reference models (override via config)
DEFAULT_REFERENCE_MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o",
    "google/gemini-2.0-flash",
]

DEFAULT_AGGREGATOR_MODEL = "anthropic/claude-sonnet-4-20250514"

AGGREGATOR_SYSTEM_PROMPT = """You have been provided with responses from multiple AI models to the user's query. Your task is to synthesize these into a single, high-quality response.

Critically evaluate each response — some may be biased or incorrect. Don't simply replicate; offer a refined, accurate, and comprehensive answer.

Responses from models:
{responses}

Now provide your synthesized response:"""


async def _call_model(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.6,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Call a single model via Caveman's provider system."""
    try:
        from caveman.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await provider.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = ""
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, dict):
            content = response.get("content", "")

        return {"model": model, "content": str(content), "error": None}

    except Exception as e:
        logger.warning("MoA reference model %s failed: %s", model, e)
        return {"model": model, "content": "", "error": str(e)}


async def _call_model_simple(
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.6,
    max_tokens: int = 4096,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Call a model via direct HTTP (OpenRouter/OpenAI compatible)."""
    try:
        import httpx

        base = api_base or "https://openrouter.ai/api/v1"
        key = api_key

        if not key:
            import os
            key = os.environ.get("OPENROUTER_API_KEY", "")

        if not key:
            return {"model": model, "content": "", "error": "no API key"}

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=HTTP_SLOW) as client:
            resp = await client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {"model": model, "content": content, "error": None}

    except Exception as e:
        logger.warning("MoA model %s failed: %s", model, e)
        return {"model": model, "content": "", "error": str(e)}


async def mixture_of_agents(
    prompt: str,
    reference_models: list[str] | None = None,
    aggregator_model: str | None = None,
    min_successful: int = 1,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run Mixture-of-Agents pipeline.

    Args:
        prompt: The user's query.
        reference_models: Models to generate diverse responses.
        aggregator_model: Model to synthesize the final response.
        min_successful: Minimum successful reference responses needed.
        api_base: API base URL (default: OpenRouter).
        api_key: API key.

    Returns dict with: content, reference_results, aggregator_model, timing.
    """
    refs = reference_models or DEFAULT_REFERENCE_MODELS
    agg = aggregator_model or DEFAULT_AGGREGATOR_MODEL

    start = time.time()

    # Phase 1: Parallel reference model calls
    tasks = [
        _call_model_simple(model, prompt, api_base=api_base, api_key=api_key)
        for model in refs
    ]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r["content"] and not r["error"]]

    if len(successful) < min_successful:
        return {
            "content": f"MoA failed: only {len(successful)}/{len(refs)} models responded",
            "reference_results": results,
            "aggregator_model": agg,
            "timing_seconds": time.time() - start,
            "error": True,
        }

    # Phase 2: Aggregation
    responses_text = "\n\n".join(
        f"[Model {i+1}: {r['model']}]\n{r['content']}"
        for i, r in enumerate(successful)
    )
    agg_system = AGGREGATOR_SYSTEM_PROMPT.format(responses=responses_text)

    agg_result = await _call_model_simple(
        agg, prompt, system=agg_system,
        temperature=0.4, api_base=api_base, api_key=api_key,
    )

    return {
        "content": agg_result["content"] or "Aggregation failed",
        "reference_results": results,
        "aggregator_model": agg,
        "timing_seconds": time.time() - start,
        "error": bool(agg_result["error"]),
    }


@tool(
    name="mixture_of_agents",
    description="Process complex queries using multiple AI models for enhanced reasoning. "
    "Best for extremely difficult problems requiring diverse perspectives.",
    params={
        "user_prompt": {"type": "string", "description": "The complex question or task"},
        "models": {"type": "string", "description": "Comma-separated model list (optional)"},
    },
    required=["user_prompt"],
)
async def mixture_of_agents_tool(
    user_prompt: str,
    models: str = "",
) -> str:
    """Run MoA on a complex query.

    Args:
        user_prompt: The complex question or task.
        models: Comma-separated model list (optional, uses defaults).
    """
    ref_models = [m.strip() for m in models.split(",") if m.strip()] or None

    result = await mixture_of_agents(
        prompt=user_prompt,
        reference_models=ref_models,
    )

    if result.get("error"):
        return f"MoA Error: {result['content']}"

    # Format output
    parts = [result["content"]]
    parts.append(f"\n---\nMoA: {len(result['reference_results'])} models, "
                 f"{result['timing_seconds']:.1f}s, aggregator: {result['aggregator_model']}")
    return "\n".join(parts)
