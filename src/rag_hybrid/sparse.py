"""Minimal BM25-style sparse embedding components.

This follows Qdrant's recommended BM25 recipe: hash tokens into a sparse
vector client-side and let the server apply IDF scoring through the
collection's ``modifier=IDF`` (enabled by
``QdrantDocumentStore(sparse_idf=True)``).

Client side we only need stable token hashing and a sublinear term
frequency weight. No corpus statistics are required, which keeps query and
document encoding symmetric without a shared vocabulary file.

Limitations (accepted for v1): no stemming and no stop-word removal, so
morphological variants do not match. Add them here if recall on keyword
queries becomes a problem.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import md5
from typing import List

from haystack import component
from haystack.dataclasses import Document, SparseEmbedding

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_HASH_SPACE = 2**31


def tokenize(text: str) -> list[str]:
    """Lowercase unicode word tokens (letters, digits, underscore, accents)."""
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def bm25_sparse_embedding(text: str) -> SparseEmbedding:
    """Hashed bag-of-words with sublinear term frequency ``1 + ln(tf)``.

    IDF weighting and document-length normalization are applied server-side
    by Qdrant's IDF modifier, so the values here are pure term frequencies.
    Indices are sorted because the Qdrant client serializes sparse vectors
    as parallel index/value arrays.
    """
    counts = Counter(tokenize(text))
    pairs: list[tuple[int, float]] = []
    for token, tf in counts.items():
        index = int(md5(token.encode("utf-8")).hexdigest()[:8], 16) % _HASH_SPACE
        pairs.append((index, 1.0 + math.log(tf)))
    pairs.sort()
    return SparseEmbedding(indices=[pair[0] for pair in pairs], values=[pair[1] for pair in pairs])


@component
class BM25SparseTextEmbedder:
    """Encodes a query into a BM25-style sparse embedding."""

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: str):
        return {"sparse_embedding": bm25_sparse_embedding(text)}


@component
class BM25SparseDocumentEmbedder:
    """Attaches BM25-style sparse embeddings to documents, in place."""

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        for document in documents:
            document.sparse_embedding = bm25_sparse_embedding(document.content or "")
        return {"documents": documents}
