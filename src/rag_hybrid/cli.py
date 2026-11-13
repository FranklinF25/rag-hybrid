"""Command line interface: index / retrieve / ask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from haystack.dataclasses import Document

from .config import get_settings
from .indexing import build_indexing_pipeline
from .retrieval import ask, build_query_pipeline, build_retrieval_pipeline, retrieve

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}


def load_documents(path: Path) -> list[Document]:
    """Read every supported file under ``path`` (recursive) into Documents."""
    if path.is_file():
        files = [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    else:
        files = sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
        )
    if not files:
        raise SystemExit(f"No supported files ({', '.join(sorted(SUPPORTED_SUFFIXES))}) found in: {path}")
    documents = []
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        if content.strip():
            documents.append(Document(content=content, meta={"source": str(file_path)}))
    if not documents:
        raise SystemExit(f"Files found but all empty in: {path}")
    return documents


def cmd_index(args: argparse.Namespace) -> int:
    pipeline = build_indexing_pipeline(recreate_index=args.recreate)
    documents = load_documents(Path(args.path))
    result = pipeline.run({"splitter": {"documents": documents}})
    written = result["writer"]["documents_written"]
    print(f"Indexed {len(documents)} file(s) -> {written} chunk(s) in collection "
          f"'{get_settings().qdrant_collection}'")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    pipeline = build_retrieval_pipeline()
    documents = retrieve(pipeline, args.query)
    if not documents:
        print("No documents found. Did you index anything? (rag-hybrid index <path>)")
        return 1
    for document in documents:
        score = document.score if document.score is not None else float("nan")
        source = document.meta.get("source", "?")
        preview = " ".join((document.content or "").split())[:140]
        print(f"[{score:.3f}] {source} :: {preview}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.deepseek_api_key:
        print("DEEPSEEK_API_KEY is not set. Copy env.example to .env and add your key.", file=sys.stderr)
        return 2
    pipeline = build_query_pipeline()
    answer, documents = ask(pipeline, args.query)
    print(answer.data or "(empty answer)")
    if documents:
        print("\nSources:")
        for document in documents:
            source = document.meta.get("source", "?")
            preview = " ".join((document.content or "").split())[:100]
            print(f"  - {source} :: {preview}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-hybrid",
        description="Semi-local hybrid RAG: Qdrant + BGE-M3 + BM25 + RRF + BGE-reranker + DeepSeek",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index .md/.txt/.rst files into Qdrant")
    index_parser.add_argument("path", help="File or directory to index (recursive)")
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before indexing",
    )
    index_parser.set_defaults(func=cmd_index)

    retrieve_parser = subparsers.add_parser("retrieve", help="Hybrid retrieval + rerank (no LLM)")
    retrieve_parser.add_argument("query", help="Query text")
    retrieve_parser.set_defaults(func=cmd_retrieve)

    ask_parser = subparsers.add_parser("ask", help="Full RAG answer via DeepSeek")
    ask_parser.add_argument("query", help="Question to answer")
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
