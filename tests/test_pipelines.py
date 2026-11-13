"""Pipeline wiring tests.

These build the actual pipelines (no model downloads happen at build time)
and pin the component graph, so a Haystack upgrade that changes connection
semantics fails loudly here instead of at query time.

They need the local Qdrant to be reachable (``docker compose up -d``);
otherwise they are skipped.
"""

from __future__ import annotations

import urllib.request

import pytest

from rag_hybrid.config import get_settings
from rag_hybrid.indexing import build_indexing_pipeline
from rag_hybrid.retrieval import build_query_pipeline, build_retrieval_pipeline


def _qdrant_up() -> bool:
    url = f"{get_settings().qdrant_url.rstrip('/')}/collections"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _qdrant_up(), reason="local Qdrant not running (docker compose up -d)")


def _edges(pipeline) -> set[tuple[str, str]]:
    # Haystack's pipeline graph is a networkx MultiDiGraph: edges are (u, v, key).
    return {(source, target) for source, target, *_ in pipeline.graph.edges}


def _names(pipeline) -> set[str]:
    return set(pipeline.graph.nodes)


def test_indexing_pipeline_wiring():
    pipeline = build_indexing_pipeline()
    assert _names(pipeline) == {"splitter", "dense_embedder", "sparse_embedder", "writer"}
    assert _edges(pipeline) == {
        ("splitter", "dense_embedder"),
        ("dense_embedder", "sparse_embedder"),
        ("sparse_embedder", "writer"),
    }


def test_retrieval_pipeline_wiring():
    pipeline = build_retrieval_pipeline()
    assert _names(pipeline) == {
        "dense_text_embedder",
        "bm25_query_embedder",
        "hybrid_retriever",
        "ranker",
    }
    assert _edges(pipeline) == {
        ("dense_text_embedder", "hybrid_retriever"),
        ("bm25_query_embedder", "hybrid_retriever"),
        ("hybrid_retriever", "ranker"),
    }


def test_query_pipeline_wiring(monkeypatch):
    # Building the pipeline must not require a real key: the generator only
    # validates its shape at init time and is never called here.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-wiring-only")
    from rag_hybrid.config import get_settings

    get_settings.cache_clear()
    pipeline = build_query_pipeline()
    assert _edges(pipeline) == {
        ("dense_text_embedder", "hybrid_retriever"),
        ("bm25_query_embedder", "hybrid_retriever"),
        ("hybrid_retriever", "ranker"),
        ("ranker", "prompt_builder"),
        ("prompt_builder", "llm"),
        ("llm", "answer_builder"),
    }


def test_settings_flow_into_components():
    settings = get_settings()
    indexing = build_indexing_pipeline()
    splitter = indexing.get_component("splitter").to_dict()["init_parameters"]
    assert splitter["split_by"] == settings.chunk_split_by
    assert splitter["split_length"] == settings.chunk_size
    assert splitter["split_overlap"] == settings.chunk_overlap

    retrieval = build_retrieval_pipeline()
    hybrid = retrieval.get_component("hybrid_retriever").to_dict()["init_parameters"]
    ranker = retrieval.get_component("ranker").to_dict()["init_parameters"]
    assert hybrid["top_k"] == settings.retrieval_top_k
    assert ranker["model"] == settings.reranker_model
    assert ranker["top_k"] == settings.rerank_top_k


def test_store_is_hybrid_with_server_side_idf():
    from rag_hybrid.store import get_document_store

    store = get_document_store()
    assert store.use_sparse_embeddings is True
    assert store.sparse_idf is True
    assert store.embedding_dim == get_settings().embed_dim
