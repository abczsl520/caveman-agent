"""Embedding providers — generate vector embeddings for memory search."""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Callable, Awaitable

import httpx

from caveman.paths import EMBEDDING_MAX_INPUT

logger = logging.getLogger(__name__)

# Cache for loaded local model
_local_model = None
_local_model_path: str | None = None

# Cache for fastembed model
_fastembed_model = None
_fastembed_model_name: str | None = None


async def fastembed_embedding(text: str, model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> list[float]:
    """Generate embedding via fastembed (ONNX, no PyTorch needed).

    Default: paraphrase-multilingual-MiniLM-L12-v2 (384d, ~50 languages, CJK+Latin).
    Cross-lingual: EN↔CN cosine ~0.70 for same-meaning queries.
    """
    global _fastembed_model, _fastembed_model_name

    try:
        from fastembed import TextEmbedding
    except ImportError:
        raise ImportError("fastembed required. Run: pip install fastembed")

    if _fastembed_model is None or _fastembed_model_name != model:
        logger.info("Loading fastembed model: %s", model)
        _fastembed_model = TextEmbedding(model_name=model)
        _fastembed_model_name = model

    embeddings = list(_fastembed_model.embed([text[:EMBEDDING_MAX_INPUT]]))
    return embeddings[0].tolist()


async def openai_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate embedding via OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"input": text[:EMBEDDING_MAX_INPUT], "model": model},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


async def ollama_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    """Generate embedding via local Ollama."""
    from caveman.paths import DEFAULT_OLLAMA_URL
    ollama_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            ollama_url + "/api/embeddings",
            json={"model": model, "prompt": text[:EMBEDDING_MAX_INPUT]},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def local_embedding(text: str, model_path: str = "") -> list[float]:
    """Generate embedding via locally trained sentence-transformers model.

    This loads the model trained by `caveman train --target embedding`.
    Model is cached after first load.
    """
    global _local_model, _local_model_path

    if not model_path:
        from caveman.paths import TRAINING_DIR
        model_path = str(TRAINING_DIR / "embedding_output")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Local embedding model not found at {model_path}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required for local embedding. Run: pip install sentence-transformers")

    # Load model (cached)
    if _local_model is None or _local_model_path != model_path:
        logger.info("Loading local embedding model from %s", model_path)
        _local_model = SentenceTransformer(model_path)
        _local_model_path = model_path

    embedding = _local_model.encode(text[:EMBEDDING_MAX_INPUT], convert_to_numpy=True)
    return embedding.tolist()


def get_embedding_fn(backend: str = "auto") -> Callable[[str], Awaitable[list[float]]] | None:
    """Get the best available embedding function.

    Priority: local (trained) > openai > fastembed > ollama
    """
    # Local trained model — highest priority if it exists
    if backend == "local":
        return local_embedding
    if backend == "auto":
        from caveman.paths import TRAINING_DIR
        local_path = TRAINING_DIR / "embedding_output"
        if local_path.exists() and any(local_path.iterdir()):
            logger.info("Using locally trained embedding model")
            return local_embedding

    if backend == "openai" or (backend == "auto" and os.environ.get("OPENAI_API_KEY")):
        return openai_embedding

    # fastembed — lightweight ONNX, no PyTorch needed
    if backend == "fastembed" or backend == "auto":
        try:
            import fastembed  # noqa: F401
            logger.info("Using fastembed for embeddings")
            return fastembed_embedding
        except ImportError:
            if backend == "fastembed":
                raise ImportError("fastembed required. Run: pip install fastembed")

    if backend == "ollama" or backend == "auto":
        if backend == "ollama":
            return ollama_embedding
    return None
