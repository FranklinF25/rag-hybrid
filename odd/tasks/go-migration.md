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

- G1 — Module scaffold: go.mod (go 1.27), internal/config (env/.env parsing, same vars), cmd/rag-hybrid/main.go skeleton — **done (parent)**
- G2 — internal/bm25: unicode tokenizer + md5→31bit hash + sublinear TF, unit tests — **done (worker A; parent re-verified + Spanish smoke)**
- G3 — internal/openrouter + internal/qdrant: clients + tests (17 unit + live integration) — **done (worker B; live fix: bare {indices,values} sparse prefetch)**
- G4 — internal/engine: index/query flows + DeepSeek client — **done (parent; real-API verified: index fresh, retrieve 0.813, ask cited)**
- G5 — internal/server + CLI — **done (parent; HTTP E2E 4/4 on host binary)**
- G6 — Docker scratch (11.2MB) + compose + E2E — **done (container E2E green)**
- G7 — Cleanup — **done**: Python runtime removed (commit 867d43d), evals ported to HTTP (language-agnostic, ?full=1 added), pyproject eval-only (.venv 134MB), README rewritten (Go)

## Log

- L1 (2026-11-14, user): "si dale arranca" (after feasibility analysis, Go-vs-Rust BM25 explanation, and the empty-DB clarification S2).
- L2 (2026-11-14, context): local Go 1.27 available; branch go-rewrite from 6093aa9; Qdrant container running (empty after recreate).
- L3 (2026-11-14, results): commits 40dc9ac (Go engine, stdlib-only, 11MB scratch image) + 867d43d (Python removal, evals→HTTP). Numbers: image 360MB→11.2MB; webapp RAM ~200MB→~40MB; .venv (eval-only) 134MB; 20 Go tests green; CLI E2E with real APIs (fresh collection 4096d+idf, retrieve 0.813/0.875, DeepSeek answer with citation); container E2E green (health/search 0.813/chat "typically 60 [1]"). Incidents fixed on the way: Go flag-after-positional (splitFlags), flags.Arg(0) leftover (empty path), Python container still bound to :8000 (docker compose stop webapp). Qdrant live-server adaptation: sparse prefetch must be the bare {indices,values} object (v1.14.1 rejects the nested sparse_vector wrapper).

Risk: none for the Go code itself (all new, tested, E2E-verified); one accepted trade-off recorded in S1/S7: evals remain Python (HTTP-coupled, optional) and the Python implementation survives only in git history on feat/v1-scaffold.
