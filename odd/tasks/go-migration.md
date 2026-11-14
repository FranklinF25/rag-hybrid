# rag-hybrid Go migration — full rewrite of the engine in Go

Rewrite the Python/FastAPI engine in Go (stdlib-only), keeping the same architecture: Qwen3-Embedding-8B + rerank-3-lite via OpenRouter, local BM25 sparse, Qdrant server-side RRF, DeepSeek official API. The web UI HTML stays as-is. Fresh Qdrant collection (old data was disposable test data). Basis: verified Python implementation on branch feat/v1-scaffold (commits d24eb52..6093aa9).

## Specs

- S1 (user, verbatim): "ahora te pregunto podemos migrar a go, igual el BM25?" → full Go rewrite including the BM25 (native Go, not Rust — decision after analysis: BM25 is <0.1% of latency, network-bound system, Rust adds FFI/service cost for nothing).
- S2 (user, verbatim): "actualmente yo no puse ningun dato importante son puros datos basura o de prueba que tu pusiste asi que no importa comenzandar con la db vacia" → start with an empty collection; NO Python-parity constraint for the tokenizer/hashes. The Go BM25 only needs self-consistency.
- S3: Unicode tokenizer still required for Spanish quality (not for parity): `[\p{L}\p{N}_]+` + unicode-aware lowercase, so "búsqueda"/"híbrida" survive as tokens.
- S4: Same stack contracts as the Python version: OpenRouter /api/v1/embeddings (qwen/qwen3-embedding-8b, 4096d, input array, encoding_format float) and /api/v1/rerank (mongodb/rerank-3-lite, top_n; results[].{index,relevance_score}) with the user's exact curls as spec; DeepSeek official API for generation; Qdrant REST (no gRPC dep) with named vectors text-dense (4096/cosine) + text-sparse (modifier idf), hybrid query with prefetch + fusion rrf.
- S5: Same external behavior: web UI (same single-file HTML, embedded via Go embed.FS), endpoints GET /, /api/health, /api/search, POST /api/chat, POST /api/documents (multi-file .md/.txt/.rst, 10MB cap, skip reasons); CLI: serve/index/retrieve/ask; config via env/.env (same variable names as env.example).
- S6: Stdlib-only where possible (net/http, encoding/json, embed, flag): zero third-party deps keeps the binary scratch-deployable (~15MB) and the toolchain clean.
- S7: DeepEval evals stay as optional Python tooling (evals/, unchanged); golden set preserved.

## Tasks

- G1 — Module scaffold: go.mod (go 1.27), internal/config (env/.env parsing, same vars), cmd/rag-hybrid/main.go skeleton — **parent, inline**
- G2 — internal/bm25: unicode tokenizer + md5→31bit hash + sublinear TF sparse vectors, unit tests — **worker A**
- G3 — internal/openrouter + internal/qdrant: embed/rerank HTTP clients (httptest mocks) + Qdrant REST client (ensure collection, upsert dense+sparse, hybrid query w/ RRF; integration test against local Qdrant, skip-if-down) — **worker B**
- G4 — internal/engine: index flow (read files → chunk 400/50 → embed + sparse → upsert) and query flow (embed → hybrid → rerank → prompt → DeepSeek → answer+sources); DeepSeek client — **parent or worker C**
- G5 — internal/server + CLI: net/http server with embedded HTML (copy static/index.html), routes per S5; flag-based subcommands — **parent or worker C**
- G6 — Dockerfile (golang:1.25-alpine build → scratch) + compose switch to Go build; E2E: fresh collection, index sample/, search, chat via HTTP — **parent**
- G7 — Cleanup: remove Python runtime from the branch after E2E parity (keep evals/ + odd/ + docs); README update; close-out — **parent**

## Log

- L1 (2026-11-14, user): "si dale arranca" (after feasibility analysis, Go-vs-Rust BM25 explanation, and the empty-DB clarification S2).
- L2 (2026-11-14, context): local Go 1.27 available; branch go-rewrite from 6093aa9; Qdrant container running (empty after recreate).
