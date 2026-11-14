"""Indexing pipeline: documents -> chunks -> dense + BM25 sparse -> Qdrant."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from .config import Settings, get_settings
from .openrouter import OpenRouterDocumentEmbedder
from .sparse import BM25SparseDocumentEmbedder
from .store import get_document_store


def build_indexing_pipeline(settings: Settings | None = None, *, recreate_index: bool = False) -> Pipeline:
    """Wire the indexing pipeline.

    The dense embedder is a stateless OpenRouter API client (no model
    weights, no warm-up), so building the pipeline is cheap and offline;
    the key is only checked when the pipeline actually runs. The BM25
    sparse embedder is pure Python and stays fully local.
    """
    settings = settings or get_settings()
    document_store = get_document_store(recreate_index=recreate_index)

    splitter = DocumentSplitter(
        split_by=settings.chunk_split_by,
        split_length=settings.chunk_size,
        split_overlap=settings.chunk_overlap,
    )
    dense_embedder = OpenRouterDocumentEmbedder(
        model=settings.embed_model,
        batch_size=settings.embed_batch_size,
    )
    sparse_embedder = BM25SparseDocumentEmbedder()
    writer = DocumentWriter(document_store, policy=DuplicatePolicy.SKIP)

    pipeline = Pipeline()
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("dense_embedder", dense_embedder)
    pipeline.add_component("sparse_embedder", sparse_embedder)
    pipeline.add_component("writer", writer)

    pipeline.connect("splitter.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")
    return pipeline
