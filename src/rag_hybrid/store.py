"""Qdrant document store factory (hybrid: dense + sparse with server-side IDF)."""

from __future__ import annotations

from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from .config import get_settings


def get_document_store(*, recreate_index: bool = False) -> QdrantDocumentStore:
    """Build the project's Qdrant store.

    - ``use_sparse_embeddings=True``: the collection gets a named sparse
      vector next to the dense one, enabling ``QdrantHybridRetriever``.
    - ``sparse_idf=True``: Qdrant applies its IDF modifier at query time,
      turning the stored term frequencies into BM25-style scores.
    """
    settings = get_settings()
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        api_key=Secret.from_token(settings.qdrant_api_key) if settings.qdrant_api_key else None,
        index=settings.qdrant_collection,
        embedding_dim=settings.embed_dim,
        use_sparse_embeddings=True,
        sparse_idf=True,
        similarity="cosine",
        recreate_index=recreate_index,
    )
