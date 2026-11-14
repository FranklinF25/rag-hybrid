"""OpenRouter API components: dense embeddings + cross-encoder reranking.

Embeddings go through OpenRouter's OpenAI-compatible ``/embeddings``
endpoint (via the ``openai`` SDK); reranking posts to ``/rerank`` with the
SDK's low-level request method because the SDK has no typed rerank API.

All components are stateless API clients: no model weights, no ``warm_up``,
no device. Together with ``sparse.py`` staying local this keeps the whole
project free of in-memory ML models; only Qdrant (server-side) and the
remote APIs do heavy lifting.

The API key is checked at run time (not at init) so that pipelines can be
built and serialized without a key, mirroring how the DeepSeek generator
treats its key.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from openai import APIConnectionError, InternalServerError, OpenAI

from .config import get_settings

# One shared HTTP client per (api_key, base_url): connection pooling across
# every component in the process, rebuilt automatically if settings change.
_client_cache: dict[tuple[str, str], OpenAI] = {}


def get_openrouter_client() -> OpenAI:
    """Return a cached OpenAI-compatible client pointed at OpenRouter.

    Reads the current settings on every call, so a missing key fails fast
    with an actionable message at query time.
    """
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy env.example to .env, add your "
            "OpenRouter key (https://openrouter.ai/settings/keys) and try again."
        )
    cache_key = (settings.openrouter_api_key, settings.openrouter_base_url)
    client = _client_cache.get(cache_key)
    if client is None:
        client = OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)
        _client_cache[cache_key] = client
    return client


def _embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    dimensions: int | None,
    timeout: float,
) -> list[list[float]]:
    """Call ``/embeddings`` once and return embeddings in input order.

    The API is OpenAI-compatible and echoes inputs in order, but sorting by
    the response's ``index`` field makes that guarantee explicit.
    ``dimensions`` is only forwarded when explicitly configured.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
        "timeout": timeout,
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def _post_rerank(client: OpenAI, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST to ``/rerank`` with one local retry on transient HTTP errors."""
    for _attempt in range(2):  # initial try + one retry
        try:
            return client.post("rerank", body=body, cast_to=dict, options={"timeout": timeout})
        except (APIConnectionError, InternalServerError):
            continue  # transient: retry once, then let the error surface
    raise RuntimeError("unreachable: rerank retry loop exhausted")  # pragma: no cover


@component
class OpenRouterTextEmbedder:
    """Embeds a query string via OpenRouter's embeddings API (stateless)."""

    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-8b",
        dimensions: int | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> None:
        """
        :param model: OpenRouter embedding model id.
        :param dimensions: Optional Matryoshka truncation; only sent when set.
        :param batch_size: Kept for symmetry with the document embedder.
        :param timeout: Per-request timeout in seconds.
        """
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model=self.model,
            dimensions=self.dimensions,
            batch_size=self.batch_size,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenRouterTextEmbedder":
        return default_from_dict(cls, data)

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        client = get_openrouter_client()
        embedding = _embed_texts(client, self.model, [text], self.dimensions, self.timeout)[0]
        return {"embedding": embedding}


@component
class OpenRouterDocumentEmbedder:
    """Attaches dense embeddings to documents via OpenRouter (stateless).

    Texts are sent in batches of ``batch_size`` (the API accepts arrays)
    and embeddings are re-attached with ``dataclasses.replace`` so no other
    pipeline stage sees in-place mutations.
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-8b",
        dimensions: int | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> None:
        """
        :param model: OpenRouter embedding model id.
        :param dimensions: Optional Matryoshka truncation; only sent when set.
        :param batch_size: Number of texts per API call.
        :param timeout: Per-request timeout in seconds.
        """
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            model=self.model,
            dimensions=self.dimensions,
            batch_size=self.batch_size,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenRouterDocumentEmbedder":
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        client = get_openrouter_client()
        texts = [document.content or "" for document in documents]
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(_embed_texts(client, self.model, batch, self.dimensions, self.timeout))
        return {
            "documents": [
                dataclasses.replace(document, embedding=embedding)
                for document, embedding in zip(documents, embeddings)
            ]
        }


@component
class OpenRouterReranker:
    """Cross-encoder reranking via OpenRouter's ``/rerank`` (stateless)."""

    def __init__(
        self,
        model: str = "mongodb/rerank-3-lite",
        top_k: int = 5,
        timeout: float = 60.0,
    ) -> None:
        """
        :param model: OpenRouter rerank model id.
        :param top_k: Default number of documents to keep; a ``run`` argument
            of the same name overrides it per call.
        :param timeout: Per-request timeout in seconds.
        """
        self.model = model
        self.top_k = top_k
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(self, model=self.model, top_k=self.top_k, timeout=self.timeout)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenRouterReranker":
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> dict[str, list[Document]]:
        if not documents:
            return {"documents": []}
        effective_top_k = top_k if top_k is not None else self.top_k
        client = get_openrouter_client()
        body: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [document.content or "" for document in documents],
            "top_n": effective_top_k,
        }
        response = _post_rerank(client, body, self.timeout)

        # ``index`` points into the documents list we sent; ``relevance_score``
        # becomes the document score. Unknown indices are skipped defensively.
        reranked: list[Document] = []
        for result in response.get("results", []):
            index = result.get("index")
            score = result.get("relevance_score")
            if index is None or score is None or not 0 <= index < len(documents):
                continue
            reranked.append(dataclasses.replace(documents[index], score=float(score)))

        # The API already truncates via ``top_n``; sort and slice again so the
        # contract holds even if a server returns extra or unordered results.
        reranked.sort(key=lambda document: document.score or 0.0, reverse=True)
        return {"documents": reranked[:effective_top_k]}
