"""Unit tests for the BM25-style sparse embedder (pure Python, no models)."""

from __future__ import annotations

import math

from rag_hybrid.sparse import bm25_sparse_embedding, tokenize


def test_tokenize_is_lowercase_and_unicode_aware():
    assert tokenize("Híbrido RAG, con BGE-M3 ¡ya!") == ["híbrido", "rag", "con", "bge", "m3", "ya"]


def test_tokenize_empty_or_none_is_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_embedding_is_deterministic():
    first = bm25_sparse_embedding("hybrid retrieval with reciprocal rank fusion")
    second = bm25_sparse_embedding("hybrid retrieval with reciprocal rank fusion")
    assert first.indices == second.indices
    assert first.values == second.values


def test_word_order_does_not_matter():
    assert bm25_sparse_embedding("dense and sparse") == bm25_sparse_embedding("sparse and dense")


def test_sublinear_term_frequency():
    embedding = bm25_sparse_embedding("qdrant qdrant qdrant once")
    # tf=3 -> 1 + ln(3); tf=1 -> 1 + ln(1) = 1.0
    assert sorted(embedding.values) == [1.0, 1.0 + math.log(3)]


def test_indices_are_sorted_and_unique():
    text = "the quick brown fox jumps over the lazy dog"
    embedding = bm25_sparse_embedding(text)
    assert embedding.indices == sorted(embedding.indices)
    assert len(set(embedding.indices)) == len(embedding.indices)
    assert len(embedding.indices) == len(set(text.lower().split()))  # 8 unique words


def test_empty_text_yields_empty_embedding():
    embedding = bm25_sparse_embedding("")
    assert embedding.indices == []
    assert embedding.values == []


def test_shared_tokens_map_to_shared_indices():
    # The same word must hash to the same index in two different texts:
    # this is what makes query/document sparse matching work.
    query = bm25_sparse_embedding("how does fusion work")
    document = bm25_sparse_embedding("rank fusion merges two result lists")
    assert set(query.indices) & set(document.indices)  # "fusion" overlaps
