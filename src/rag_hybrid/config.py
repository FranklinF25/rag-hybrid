"""Environment-driven configuration for rag-hybrid.

Isolation contract (spec S2): nothing may leak outside the project folder.
The HuggingFace cache is redirected to ``<repo>/.models`` *before* any
transformers/sentence-transformers import, and Qdrant data lives in the
compose bind mount ``./.qdrant_storage``.
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

    # --- Local models (CPU) ---------------------------------------------------
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"

    # --- Chunking / retrieval -------------------------------------------------
    chunk_split_by: Literal["word", "sentence", "page"] = "word"
    chunk_size: int = 400
    chunk_overlap: int = 50
    retrieval_top_k: int = 20
    rerank_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (env is read once per process)."""
    return Settings()
