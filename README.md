# rag-hybrid

Semi-local hybrid RAG in **Go** (stdlib-only): **Qdrant** with dense
**Qwen3-Embedding-8B** (OpenRouter) + sparse **BM25** (local) fused with
server-side **RRF**, reranked by **Voyage rerank-3-lite** (OpenRouter),
answers by the **DeepSeek API**. Single 11 MB static binary; scratch Docker
image; no ML models on the machine.

```
query → Qwen3 embed (API) ┐
       → BM25 sparse ──────┴→ Qdrant hybrid + RRF → rerank-3-lite → DeepSeek → answer + sources
```

## Layout

```
cmd/rag-hybrid/       CLI entrypoint (serve / index / retrieve / ask)
internal/bm25/        unicode tokenizer + hashed sparse encoder
internal/openrouter/  embeddings + rerank HTTP clients
internal/qdrant/      REST client: ensure/upsert/hybrid-query/drop
internal/engine/      document loading, chunking, index & query flows
internal/server/      net/http API + embedded web UI
evals/                DeepEval harness + golden set (Python, optional)
sample/               demo documents
```

## Quickstart (Docker)

```bash
cp env.example .env   # set OPENROUTER_API_KEY and DEEPSEEK_API_KEY
docker compose up -d --build
open http://127.0.0.1:8000
```

Two containers: `webapp` (11 MB scratch image) + `qdrant` (v1.14.1, data in
`./.qdrant_storage`, owned by your user). `docker compose down -v` plus
deleting the repo leaves the machine exactly as it was.

## Quickstart (binary)

```bash
go build -o rag-hybrid ./cmd/rag-hybrid   # or: go run ./cmd/rag-hybrid …
docker compose up -d qdrant
./rag-hybrid index sample/ --recreate
./rag-hybrid retrieve "damping constant RRF"
./rag-hybrid ask "How does RRF fuse the two ranked lists?"
./rag-hybrid serve                       # web UI on WEB_HOST:WEB_PORT
```

## Configuration

Same variables as `env.example` (loaded from the environment or `.env`):
OpenRouter (embed + rerank), DeepSeek, Qdrant URL/collection, chunking
(400 words / 50 overlap), retrieval top-k (20 → rerank → 5), web host/port.

## Evaluation

```bash
uv sync                    # only pytest/deepeval/openai (Python used for evals only)
pytest -m eval evals/      # needs the API running + DEEPSEEK_API_KEY for the judge
```

Thresholds (research baseline): faithfulness ≥ 0.85, contextual precision
≥ 0.75; recall and relevancy at 0.70.

## Notes

- Qdrant server is pinned to `v1.14.1` in compose; the hybrid query uses
  the REST API with two prefetch legs and `{"fusion":"rrf"}`.
- The Go BM25 needs no Python-parity: the collection was recreated from
  scratch after the rewrite (old data was disposable).
- Python (Haystack) implementation lives in git history on branch
  `feat/v1-scaffold`; `evals/` is the only Python that remains.
