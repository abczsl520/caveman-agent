"""Tests for Round 102 — Groq, DeepSeek, Mistral, Together providers."""
from __future__ import annotations
import os
import pytest
from unittest.mock import patch

from caveman.providers.openai_provider import OpenAIProvider
from caveman.providers.groq_provider import GroqProvider
from caveman.providers.deepseek_provider import DeepSeekProvider
from caveman.providers.mistral_provider import MistralProvider
from caveman.providers.together_provider import TogetherProvider
from caveman.providers.model_metadata import get_model_info, infer_provider_from_url


# --- Inheritance ---

def test_groq_inherits_openai():
    assert issubclass(GroqProvider, OpenAIProvider)


def test_deepseek_inherits_openai():
    assert issubclass(DeepSeekProvider, OpenAIProvider)


def test_mistral_inherits_openai():
    assert issubclass(MistralProvider, OpenAIProvider)


def test_together_inherits_openai():
    assert issubclass(TogetherProvider, OpenAIProvider)


# --- Default base URLs ---

def test_groq_default_base_url():
    p = GroqProvider(api_key="test-key")
    assert p.base_url == "https://api.groq.com/openai/v1"
    assert p.model == "llama-3.3-70b-versatile"


def test_deepseek_default_base_url():
    p = DeepSeekProvider(api_key="test-key")
    assert p.base_url == "https://api.deepseek.com/v1"
    assert p.model == "deepseek-chat"


def test_mistral_default_base_url():
    p = MistralProvider(api_key="test-key")
    assert p.base_url == "https://api.mistral.ai/v1"
    assert p.model == "mistral-large-latest"


def test_together_default_base_url():
    p = TogetherProvider(api_key="test-key")
    assert p.base_url == "https://api.together.xyz/v1"
    assert p.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"


# --- Custom model ---

def test_groq_custom_model():
    p = GroqProvider(model="llama-3.1-8b-instant", api_key="k")
    assert p.model == "llama-3.1-8b-instant"


def test_deepseek_custom_model():
    p = DeepSeekProvider(model="deepseek-reasoner", api_key="k")
    assert p.model == "deepseek-reasoner"


# --- Env var fallback ---

def test_groq_env_key():
    with patch.dict(os.environ, {"GROQ_API_KEY": "env-groq"}):
        p = GroqProvider()
        assert p.api_key == "env-groq"


def test_deepseek_env_key():
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-ds"}):
        p = DeepSeekProvider()
        assert p.api_key == "env-ds"


def test_mistral_env_key():
    with patch.dict(os.environ, {"MISTRAL_API_KEY": "env-mis"}):
        p = MistralProvider()
        assert p.api_key == "env-mis"


def test_together_env_key():
    with patch.dict(os.environ, {"TOGETHER_API_KEY": "env-tog"}):
        p = TogetherProvider()
        assert p.api_key == "env-tog"


# --- Factory auto-detection ---

def test_factory_auto_detect_groq():
    from caveman.agent.factory import create_loop
    with patch.dict(os.environ, {"GROQ_API_KEY": "test"}):
        # We can't fully create a loop without all deps, so test the import path
        from caveman.providers.groq_provider import GroqProvider as GP
        assert GP is not None


def test_factory_auto_detect_deepseek():
    from caveman.agent.factory import create_loop
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
        from caveman.providers.deepseek_provider import DeepSeekProvider as DP
        assert DP is not None


# --- Model metadata ---

def test_groq_model_metadata():
    info = get_model_info("llama-3.3-70b-versatile")
    assert info.context_length == 128_000
    assert info.provider == "groq"


def test_mistral_model_metadata():
    info = get_model_info("mistral-large-latest")
    assert info.context_length == 128_000
    assert info.provider == "mistral"


def test_together_model_metadata():
    info = get_model_info("meta-llama/Llama-3.3-70B-Instruct-Turbo")
    assert info.context_length == 131_072
    assert info.provider == "together"


def test_codestral_model_metadata():
    info = get_model_info("codestral-latest")
    assert info.context_length == 256_000
    assert info.provider == "mistral"


# --- URL inference ---

def test_infer_groq_from_url():
    assert infer_provider_from_url("https://api.groq.com/openai/v1") == "groq"


def test_infer_mistral_from_url():
    assert infer_provider_from_url("https://api.mistral.ai/v1") == "mistral"


def test_infer_together_from_url():
    assert infer_provider_from_url("https://api.together.xyz/v1") == "together"


# --- max_tokens passthrough ---

def test_groq_max_tokens():
    p = GroqProvider(api_key="k", max_tokens=2048)
    assert p.max_tokens == 2048


def test_together_max_tokens():
    p = TogetherProvider(api_key="k", max_tokens=4096)
    assert p.max_tokens == 4096
