"""Indexing pipeline: documents -> chunks -> dense + BM25 sparse -> Qdrant."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import ComponentDevice
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
)

from .config import Settings, get_settings
from .sparse import BM25SparseDocumentEmbedder
from .store import get_document_store


def build_indexing_pipeline(settings: Settings | None = None, *, recreate_index: bool = False) -> Pipeline:
    """Wire the indexing pipeline.

    The dense embedder (BGE-M3) loads lazily on ``warm_up()`` / first run,
    so building the pipeline is cheap and offline. The BM25 sparse embedder
    is pure Python and adds no model download.
    """
    settings = settings or get_settings()
    document_store = get_document_store(recreate_index=recreate_index)

    splitter = DocumentSplitter(
        split_by=settings.chunk_split_by,
        split_length=settings.chunk_size,
        split_overlap=settings.chunk_overlap,
    )
    dense_embedder = SentenceTransformersDocumentEmbedder(
        model=settings.embed_model,
        device=ComponentDevice.from_str(settings.device),
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
