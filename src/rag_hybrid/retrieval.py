"""Query pipelines: hybrid retrieval -> RRF -> rerank -> DeepSeek -> answer.

Two pipelines are provided:

- ``build_retrieval_pipeline``: up to the reranker. Used for debugging and
  for evaluation (retrieval_context) and works without an LLM API key.
- ``build_query_pipeline``: full RAG with the DeepSeek generator.

RRF itself runs inside Qdrant (server-side ``FusionQuery(RRF)`` over the
dense and sparse prefetches), so no client-side ``DocumentJoiner`` is
needed.
"""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.dataclasses import Answer, Document
from haystack.utils import ComponentDevice, Secret
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.rankers.sentence_transformers import (
    SentenceTransformersSimilarityRanker,
)
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever

from .config import Settings, get_settings
from .sparse import BM25SparseTextEmbedder
from .store import get_document_store

PROMPT_TEMPLATE = (
    "Answer the question using only the documents below.\n"
    "If the documents do not contain the answer, say you don't know.\n"
    "Cite the documents you use like [1], [2].\n\n"
    "{% for doc in documents %}"
    "[{{ loop.index }}] {{ doc.content }}\n\n"
    "{% endfor %}"
    "Question: {{ question }}\n"
    "Answer:"
)


def _add_retrieval_components(pipeline: Pipeline, settings: Settings) -> None:
    """Shared head: query embedders -> hybrid retriever (RRF) -> reranker."""
    device = ComponentDevice.from_str(settings.device)
    pipeline.add_component(
        "dense_text_embedder",
        SentenceTransformersTextEmbedder(model=settings.embed_model, device=device),
    )
    pipeline.add_component("bm25_query_embedder", BM25SparseTextEmbedder())
    pipeline.add_component(
        "hybrid_retriever",
        QdrantHybridRetriever(
            document_store=get_document_store(),
            top_k=settings.retrieval_top_k,
        ),
    )
    pipeline.add_component(
        "ranker",
        SentenceTransformersSimilarityRanker(
            model=settings.reranker_model,
            device=device,
            top_k=settings.rerank_top_k,
        ),
    )
    pipeline.connect("dense_text_embedder.embedding", "hybrid_retriever.query_embedding")
    pipeline.connect("bm25_query_embedder.sparse_embedding", "hybrid_retriever.query_sparse_embedding")
    pipeline.connect("hybrid_retriever.documents", "ranker.documents")


def build_retrieval_pipeline(settings: Settings | None = None) -> Pipeline:
    """Retrieval-only pipeline (no LLM): hybrid search + rerank."""
    settings = settings or get_settings()
    pipeline = Pipeline()
    _add_retrieval_components(pipeline, settings)
    return pipeline


def build_query_pipeline(settings: Settings | None = None) -> Pipeline:
    """Full RAG pipeline: retrieval + rerank + prompt + DeepSeek + answer."""
    settings = settings or get_settings()
    pipeline = Pipeline()
    _add_retrieval_components(pipeline, settings)

    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(template=PROMPT_TEMPLATE, required_variables=["documents", "question"]),
    )
    pipeline.add_component(
        "llm",
        OpenAIGenerator(
            api_key=Secret.from_token(settings.deepseek_api_key),
            model=settings.llm_model,
            api_base_url=settings.llm_base_url,
            generation_kwargs={"max_tokens": settings.generation_max_tokens},
        ),
    )
    pipeline.add_component("answer_builder", AnswerBuilder())

    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.prompt")
    pipeline.connect("llm.replies", "answer_builder.replies")
    return pipeline


def _run_retrieval(pipeline: Pipeline, question: str) -> dict[str, list[Document]]:
    return pipeline.run(
        {
            "dense_text_embedder": {"text": question},
            "bm25_query_embedder": {"text": question},
            "ranker": {"query": question},
        }
    )


def retrieve(pipeline: Pipeline, question: str) -> list[Document]:
    """Return the reranked documents for a question."""
    return _run_retrieval(pipeline, question)["ranker"]["documents"]


def ask(pipeline: Pipeline, question: str) -> tuple[Answer, list[Document]]:
    """Run the full RAG pipeline; returns (answer, source documents)."""
    result = pipeline.run(
        {
            "dense_text_embedder": {"text": question},
            "bm25_query_embedder": {"text": question},
            "ranker": {"query": question},
            "prompt_builder": {"question": question},
            "answer_builder": {"query": question},
        }
    )
    answer = result["answer_builder"]["answers"][0]
    documents = result["ranker"]["documents"]
    return answer, documents
