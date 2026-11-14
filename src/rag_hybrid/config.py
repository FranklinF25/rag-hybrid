"""Environment-driven configuration for rag-hybrid.

Isolation contract (spec S2): nothing may leak outside the project folder.
Qdrant data lives in the compose bind mount ``./.qdrant_storage``; the
HuggingFace cache redirect below is kept as a safety net for any optional
local-model tooling even though the pipeline itself no longer loads ML
models (embeddings and reranking go through OpenRouter, BM25 is pure Python).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Project-contained model cache (set before HF libs are imported anywhere).
MODELS_DIR = PROJECT_ROOT / ".models"
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Settings(BaseSettings):
    """All runtime knobs, overridable via environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (generation): DeepSeek, OpenAI-compatible -----------------------
    deepseek_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    generation_max_tokens: int = 700

    # --- Qdrant ---------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_hybrid"

    # --- Embeddings / reranking: OpenRouter API -------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embed_model: str = "qwen/qwen3-embedding-8b"
    embed_dim: int = 4096
    embed_batch_size: int = 32
    reranker_model: str = "mongodb/rerank-3-lite"

    # --- Chunking / retrieval -------------------------------------------------
    chunk_split_by: Literal["word", "sentence", "page"] = "word"
    chunk_size: int = 400
    chunk_overlap: int = 50
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    # --- Web UI ---------------------------------------------------------------
    web_host: str = "127.0.0.1"
    web_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (env is read once per process)."""
    return Settings()
