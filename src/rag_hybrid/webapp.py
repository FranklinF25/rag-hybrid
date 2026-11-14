"""Lightweight web UI: single-page chat + document upload over the RAG stack.

The retrieval components are stateless API clients (OpenRouter embeddings
+ rerank, local BM25 sparse, DeepSeek generation), so there are no warm
models to share: pipelines are built once on first use and cached. The
only lock guards document-store construction.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from pydantic import BaseModel

from .config import get_settings
from .openrouter import OpenRouterDocumentEmbedder
from .retrieval import ask, build_query_pipeline, build_retrieval_pipeline, retrieve
from .sparse import BM25SparseDocumentEmbedder
from .store import get_document_store

STATIC_DIR = Path(__file__).parent / "static"
SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
HEALTH_TIMEOUT_SECONDS = 2


class NoKeyError(RuntimeError):
    """Raised when chat is requested but DEEPSEEK_API_KEY is not configured."""


class Engine:
    """Lazily built, thread-safe holders for the pipelines and the store.

    Pipeline components are stateless API clients (no ``warm_up``, no
    weights), so building a pipeline is cheap; each is constructed on first
    use and cached. A concurrent double-build is harmless (same result),
    so only store construction keeps a lock.
    """

    def __init__(self) -> None:
        self._store_lock = threading.Lock()
        self._query_pipeline = None
        self._retrieval_pipeline = None
        self._store = None

    # --- pipelines --------------------------------------------------------------

    def query_pipeline(self):
        """Full RAG pipeline (retrieval + rerank + DeepSeek), built once, cached."""
        if self._query_pipeline is None:
            self._query_pipeline = build_query_pipeline()
        return self._query_pipeline

    def search(self, question: str) -> list[Document]:
        """Reranked documents for a question (no LLM involved)."""
        if self._retrieval_pipeline is None:
            self._retrieval_pipeline = build_retrieval_pipeline()
        return retrieve(self._retrieval_pipeline, question)

    def chat(self, message: str):
        """Answer + source documents via the full RAG pipeline."""
        if not get_settings().deepseek_api_key:
            raise NoKeyError("DEEPSEEK_API_KEY is not set")
        return ask(self.query_pipeline(), message)

    # --- ingestion ----------------------------------------------------------------

    def store(self):
        """Cached Qdrant document store used for uploads."""
        with self._store_lock:
            if self._store is None:
                self._store = get_document_store()
            return self._store

    def upload(self, files: list[tuple[str, bytes]]) -> dict:
        """Split, embed (dense + BM25 sparse) and index uploaded files.

        Mirrors the ``indexing`` pipeline with the stateless components: one
        batched OpenRouter call for all dense embeddings of a file, local
        BM25 sparse embeddings, then one write per file with
        ``DuplicatePolicy.SKIP``.
        """
        settings = get_settings()
        store = self.store()
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

        indexed_files = 0
        total_chunks = 0
        skipped: list[str] = []
        for filename, payload in files:
            if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped.append(f"{filename}: unsupported type (allowed: .md, .txt, .rst)")
                continue
            if len(payload) > MAX_UPLOAD_BYTES:
                skipped.append(f"{filename}: larger than 10 MB")
                continue
            try:
                content = payload.decode("utf-8")
            except UnicodeDecodeError:
                skipped.append(f"{filename}: not valid UTF-8 text")
                continue
            if not content.strip():
                skipped.append(f"{filename}: empty file")
                continue

            document = Document(content=content, meta={"source": filename})
            chunks = splitter.run(documents=[document])["documents"]
            chunks = dense_embedder.run(documents=chunks)["documents"]
            chunks = sparse_embedder.run(documents=chunks)["documents"]
            store.write_documents(chunks, policy=DuplicatePolicy.SKIP)
            indexed_files += 1
            total_chunks += len(chunks)
        return {"files": indexed_files, "chunks": total_chunks, "skipped": skipped}

    # --- observability -----------------------------------------------------------

    def health(self) -> dict:
        """Reachability snapshot for the UI badge; never raises."""
        settings = get_settings()
        base = settings.qdrant_url.rstrip("/")
        qdrant = False
        points = None
        try:
            with urllib.request.urlopen(f"{base}/collections", timeout=HEALTH_TIMEOUT_SECONDS) as response:
                qdrant = response.status == 200
            # HTTPError (e.g. 404 for a missing collection) keeps points = None.
            with urllib.request.urlopen(
                f"{base}/collections/{settings.qdrant_collection}", timeout=HEALTH_TIMEOUT_SECONDS
            ) as response:
                points = json.load(response).get("result", {}).get("points_count")
        except OSError:
            pass
        return {
            "qdrant": qdrant,
            "points": points,
            "key": bool(settings.deepseek_api_key),
            "openrouter": bool(settings.openrouter_api_key),
            "pipelines_built": self._query_pipeline is not None,
        }


class ChatRequest(BaseModel):
    message: str


def _result(document: Document) -> dict:
    """Serialize a retrieved document for the UI (source, score, preview)."""
    preview = " ".join((document.content or "").split())[:140]
    return {
        "source": document.meta.get("source", "?"),
        "score": document.score,
        "preview": preview,
    }


def create_app(engine: Engine | None = None) -> FastAPI:
    """FastAPI application factory: single-page UI + JSON API."""
    app = FastAPI(title="rag-hybrid", docs_url=None, redoc_url=None)
    app.state.engine = engine or Engine()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return app.state.engine.health()

    @app.get("/api/search")
    def search(q: str) -> dict:
        try:
            documents = app.state.engine.search(q)
        except Exception as exc:  # Qdrant unreachable / store errors
            raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {exc}") from exc
        return {"results": [_result(document) for document in documents]}

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> dict:
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="message must not be empty")
        try:
            answer, documents = app.state.engine.chat(message)
        except NoKeyError:
            raise HTTPException(
                status_code=503,
                detail="DEEPSEEK_API_KEY is not set. Copy env.example to .env and add your key.",
            ) from None
        except Exception as exc:  # Qdrant unreachable / store errors
            raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {exc}") from exc
        return {"answer": answer.data or "", "sources": [_result(d) for d in documents]}

    @app.post("/api/documents")
    def upload(files: list[UploadFile] = File(...)) -> dict:
        payloads = [(file.filename or "unnamed", file.file.read()) for file in files]
        try:
            result = app.state.engine.upload(payloads)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Indexing failed: {exc}") from exc
        if result["files"] == 0:
            reasons = "; ".join(result["skipped"]) or "no files received"
            raise HTTPException(status_code=400, detail=f"No files indexed: {reasons}")
        return result

    return app
