# rag-hybrid v1 — semi-local hybrid RAG scaffold

Haystack 2.x + Qdrant hybrid pipeline (BGE-M3 dense + BM25 sparse, RRF, BGE-reranker-v2-m3) with DeepSeek as generation LLM. Full isolation: Docker for Qdrant, all caches/venv/models inside the project folder. Basis: research sessions of 2026-09-07 / 09-11 (see Engram `research/rag-framework-landscape-2026`).

## Specs

- S1 — Stack (user, verbatim): "Qdrant + BGE-M3 + BM25 + RRF + BGE-reranker-v2-m3 + tu LLM." Hybrid retrieval: dense BGE-M3 embeddings + BM25 sparse, RRF fusion, cross-encoder rerank, LLM answer generation.
- S2 — Semi-local budget + isolation (user): "seria semi-local." and "ah oye vas a usar docker para no ensuciar mi maquina cierto?" → embeddings + reranker on CPU, LLM via API, ~8GB RAM machine (7.3Gi observed). Qdrant in Docker with project-local bind mount; venv, HF model cache and Qdrant data all inside the repo; nothing installed system-wide.
- S3 — Framework: Haystack 2.x (deep research conclusion after user's "haz un deep research sobre frameworks de rag"; strongest fit for Qdrant hybrid + RRF + sentence-transformers ranker on 8GB).
- S4 — Evaluation from day one: DeepEval harness with golden set; thresholds from research: faithfulness >= 0.85, context precision >= 0.75 (golden set 20-50 questions at maturity; v1 ships a stub + runner).
- S5 — LLM provider (user, verbatim): "vamos a usar deepseek" → DeepSeek API, OpenAI-compatible (base URL `https://api.deepseek.com/v1`, default model `deepseek-chat`), all LLM settings env-driven.

## Tasks

- T1 — Scaffold: branch `feat/v1-scaffold`, uv project (src layout), torch CPU index, `.env.example`, `.gitignore`, `docker-compose.yml` (Qdrant, bind mount `./.qdrant_storage`) — S2, S3
- T2 — `config.py` (pydantic-settings, env-driven; `HF_HOME=./.models`) + `store.py` (QdrantDocumentStore factory with sparse support) — S1, S2
- T3 — `indexing.py`: ingest → chunk → BGE-M3 dense embed + BM25 sparse embed → Qdrant writer — S1
- T4 — `retrieval.py`: hybrid retrieve → RRF → BGE-reranker-v2-m3 → prompt → DeepSeek (OpenAI-compatible) → answer — S1, S5
- T5 — `cli.py`: `rag-hybrid index <path>` / `rag-hybrid retrieve <q>` / `rag-hybrid ask <q>` — S1
- T6 — tests (config + pipeline wiring with fake components) + `evals/` DeepEval harness + golden-set stub — S4
- T7 — README quickstart + end-to-end smoke (Docker Qdrant up, index `sample/`, retrieve without LLM key; `ask` requires `DEEPSEEK_API_KEY`) — all

## Log

- L1 (2026-09-07, user): "oye quiero hacer algo sobre rag, entiendo que hay muchos investiga sobre ello y particularmente sobre hybrid rag y lightrag. ademas sobre qdran"
- L2 (2026-09-07, user): "seria semi-local. alguien me recomendo esto dime que opinas en comparacion a lo que estamos viendo: Qdrant + BGE-M3 + BM25 + RRF + BGE-reranker-v2-m3 + tu LLM."
- L3 (2026-09-11, user): "haz un deep research sobre frameworks de rag" → decision: Haystack 2.x (4-layer taxonomy, benchmark overhead negligible, eval stack DeepEval CI + RAGAS nightly + TruLens).
- L4 (2026-11-13): proposal to scaffold v1 (Haystack 2.x + Qdrant hybrid pipeline + DeepEval day one) → user: "si dale"
- L5 (2026-11-13, user): "ah oye vas a usar docker para no ensuciar mi maquina cierto?" → isolation plan confirmed: Docker Qdrant + project-contained venv/models/data; cleanup = `docker compose down -v` + delete folder.
- L6 (2026-11-13, user): "vamos a usar deepseek" → S5.
