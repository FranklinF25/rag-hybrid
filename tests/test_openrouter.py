"""Unit tests for the OpenRouter components (fully mocked API client).

No network access happens here: the shared client factory is monkeypatched
with a fake that records requests and returns canned API responses. The
fakes deliberately return embeddings/results out of order so the components'
defensive ``index`` sorting is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from haystack import Document

from rag_hybrid import openrouter
from rag_hybrid.config import Settings
from rag_hybrid.openrouter import (
    OpenRouterDocumentEmbedder,
    OpenRouterReranker,
    OpenRouterTextEmbedder,
    get_openrouter_client,
)

# Canned rerank scores keyed by document content.
RERANK_SCORES = {"alpha": 0.1, "beta": 0.9, "gamma": 0.5}


class FakeOpenRouterClient:
    """Stands in for the OpenAI SDK client pointed at OpenRouter."""

    def __init__(self) -> None:
        self.embedding_calls: list[dict] = []
        self.posts: list[tuple[str, dict, dict]] = []

        client = self

        class _Embeddings:
            def create(self, **kwargs):
                client.embedding_calls.append(kwargs)
                # Embeddings are derived from the text so order can be asserted
                # across batches; ``index`` stays the per-request input position.
                base = {"alpha": 0.0, "beta": 1.0, "gamma": 2.0}
                data = [
                    SimpleNamespace(
                        embedding=[base.get(text, 0.0), base.get(text, 0.0) + 0.5], index=i
                    )
                    for i, text in enumerate(kwargs["input"])
                ]
                data.reverse()  # out of order: components must sort by "index"
                return SimpleNamespace(data=data)

        self.embeddings = _Embeddings()

    def post(self, path, *, body=None, cast_to=None, options=None):
        self.posts.append((path, body, options))
        results = [
            {"index": i, "relevance_score": RERANK_SCORES[text]}
            for i, text in enumerate(body["documents"])
        ]
        results.reverse()  # unordered response: components must sort by score
        return {"results": results}


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeOpenRouterClient()
    monkeypatch.setattr(openrouter, "get_openrouter_client", lambda: client)
    return client


# --- text embedder -----------------------------------------------------------


def test_text_embedder_returns_float_list(fake_client):
    embedder = OpenRouterTextEmbedder(model="qwen/qwen3-embedding-8b")
    result = embedder.run(text="hello")
    assert result == {"embedding": [0.0, 0.5]}
    assert all(isinstance(value, float) for value in result["embedding"])

    (call,) = fake_client.embedding_calls
    assert call["model"] == "qwen/qwen3-embedding-8b"
    assert call["input"] == ["hello"]
    assert call["encoding_format"] == "float"
    assert "dimensions" not in call  # only sent when explicitly configured


def test_text_embedder_forwards_dimensions_when_set(fake_client):
    embedder = OpenRouterTextEmbedder(model="m", dimensions=512)
    embedder.run(text="hello")
    assert fake_client.embedding_calls[0]["dimensions"] == 512


# --- document embedder ---------------------------------------------------------


def test_document_embedder_batches_and_preserves_order(fake_client):
    documents = [Document(content=text) for text in ["alpha", "beta", "gamma"]]
    embedder = OpenRouterDocumentEmbedder(model="m", batch_size=2)
    result = embedder.run(documents=documents)

    # 3 texts with batch_size=2 -> two API calls of sizes 2 and 1.
    assert [len(call["input"]) for call in fake_client.embedding_calls] == [2, 1]
    assert fake_client.embedding_calls[0]["input"] == ["alpha", "beta"]
    assert fake_client.embedding_calls[1]["input"] == ["gamma"]

    # Order is preserved and embeddings land on the right documents.
    embedded = result["documents"]
    assert [document.content for document in embedded] == ["alpha", "beta", "gamma"]
    assert [document.embedding for document in embedded] == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]

    # New Document objects via dataclasses.replace: inputs are not mutated.
    assert all(original.embedding is None for original in documents)


# --- reranker ------------------------------------------------------------------


def test_reranker_sorts_truncates_and_sets_scores(fake_client):
    documents = [Document(content=text) for text in ["alpha", "beta", "gamma"]]
    reranker = OpenRouterReranker(model="mongodb/rerank-3-lite", top_k=2)
    result = reranker.run(query="q", documents=documents)

    assert [document.content for document in result["documents"]] == ["beta", "gamma"]
    assert [document.score for document in result["documents"]] == [0.9, 0.5]

    path, body, options = fake_client.posts[0]
    assert path == "rerank"
    assert body["model"] == "mongodb/rerank-3-lite"
    assert body["query"] == "q"
    assert body["documents"] == ["alpha", "beta", "gamma"]
    assert body["top_n"] == 2  # init top_k flows into the request

    # Input documents are not mutated: scores live on the copies.
    assert all(original.score is None for original in documents)


def test_reranker_top_k_argument_overrides_init(fake_client):
    documents = [Document(content=text) for text in ["alpha", "beta", "gamma"]]
    reranker = OpenRouterReranker(model="m", top_k=3)
    result = reranker.run(query="q", documents=documents, top_k=1)
    assert [document.content for document in result["documents"]] == ["beta"]
    assert fake_client.posts[0][1]["top_n"] == 1


def test_reranker_with_empty_documents_returns_empty_list(fake_client):
    reranker = OpenRouterReranker(model="m", top_k=2)
    assert reranker.run(query="q", documents=[]) == {"documents": []}
    assert fake_client.posts == []  # short-circuits before any API call


# --- missing key -----------------------------------------------------------------


@pytest.fixture
def no_openrouter_key(monkeypatch):
    monkeypatch.setattr(
        openrouter,
        "get_settings",
        lambda: Settings(openrouter_api_key="", openrouter_base_url="https://openrouter.ai/api/v1"),
    )
    openrouter._client_cache.clear()
    yield
    openrouter._client_cache.clear()


def test_missing_key_raises_actionable_error(no_openrouter_key):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_openrouter_client()

    # The check happens at run time, so every component surfaces it on run.
    for component, kwargs in [
        (OpenRouterTextEmbedder(), {"text": "hello"}),
        (OpenRouterDocumentEmbedder(), {"documents": [Document(content="hello")]}),
        (OpenRouterReranker(), {"query": "q", "documents": [Document(content="hello")]}),
    ]:
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            component.run(**kwargs)


# --- serialization ----------------------------------------------------------------


def test_serialization_round_trip():
    components = [
        OpenRouterTextEmbedder(model="m1", dimensions=8, batch_size=4, timeout=12.0),
        OpenRouterDocumentEmbedder(model="m2", batch_size=7),
        OpenRouterReranker(model="r1", top_k=3, timeout=5.0),
    ]
    for component in components:
        data = component.to_dict()
        assert data["type"].endswith(type(component).__name__)
        restored = type(component).from_dict(data)
        assert restored.to_dict() == data
    assert OpenRouterReranker().to_dict()["init_parameters"]["top_k"] == 5
