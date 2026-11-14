"""Config tests: env parsing, defaults and the HF cache redirect."""

from __future__ import annotations

import pytest

from rag_hybrid import config
from rag_hybrid.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_match_the_researched_stack():
    settings = Settings()
    assert settings.embed_model == "qwen/qwen3-embedding-8b"
    assert settings.embed_dim == 4096
    assert settings.embed_batch_size == 32
    assert settings.reranker_model == "mongodb/rerank-3-lite"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.llm_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_model == "deepseek-chat"
    assert settings.chunk_split_by == "word"
    assert settings.retrieval_top_k == 20
    assert settings.rerank_top_k == 5
    assert not hasattr(settings, "device")  # no local ML models remain


def test_env_overrides_take_effect(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example:6333")
    monkeypatch.setenv("RERANK_TOP_K", "3")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    settings = get_settings()
    assert settings.qdrant_url == "http://qdrant.example:6333"
    assert settings.rerank_top_k == 3
    assert settings.deepseek_api_key == "sk-test"


def test_invalid_chunk_split_by_is_rejected(monkeypatch):
    monkeypatch.setenv("CHUNK_SPLIT_BY", "paragraph")
    with pytest.raises(Exception):  # pydantic ValidationError on the Literal
        get_settings()


def test_hf_cache_is_redirected_into_the_project():
    # Isolation contract (S2): models must never land in ~/.cache/huggingface.
    assert config.MODELS_DIR == config.PROJECT_ROOT / ".models"
    assert "HF_HOME" in __import__("os").environ
