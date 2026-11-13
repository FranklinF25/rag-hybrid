# rag-hybrid v1 — semi-local hybrid RAG scaffold

Haystack 2.x + Qdrant hybrid pipeline (BGE-M3 dense + BM25 sparse, RRF, BGE-reranker-v2-m3) with DeepSeek as generation LLM. Full isolation: Docker for Qdrant, all caches/venv/models inside the project folder. Basis: research sessions of 2026-09-07 / 09-11 (see Engram `research/rag-framework-landscape-2026`).

## Specs

- S1 — Stack (user, verbatim): "Qdrant + BGE-M3 + BM25 + RRF + BGE-reranker-v2-m3 + tu LLM." Hybrid retrieval: dense BGE-M3 embeddings + BM25 sparse, RRF fusion, cross-encoder rerank, LLM answer generation.
- S2 — Semi-local budget + isolation (user): "seria semi-local." and "ah oye vas a usar docker para no ensuciar mi maquina cierto?" → embeddings + reranker on CPU, LLM via API, ~8GB RAM machine (7.3Gi observed). Qdrant in Docker with project-local bind mount; venv, HF model cache and Qdrant data all inside the repo; nothing installed system-wide.
- S3 — Framework: Haystack 2.x (deep research conclusion after user's "haz un deep research sobre frameworks de rag"; strongest fit for Qdrant hybrid + RRF + sentence-transformers ranker on 8GB).
- S4 — Evaluation from day one: DeepEval harness with golden set; thresholds from research: faithfulness >= 0.85, context precision >= 0.75 (golden set 20-50 questions at maturity; v1 ships a stub + runner).
- S5 — LLM provider (user, verbatim): "vamos a usar deepseek" → DeepSeek API, OpenAI-compatible (base URL `https://api.deepseek.com/v1`, default model `deepseek-chat`), all LLM settings env-driven.

## Tasks

- T1 — Scaffold: branch `feat/v1-scaffold`, uv project (src layout), torch CPU index, `.env.example`, `.gitignore`, `docker-compose.yml` (Qdrant, bind mount `./.qdrant_storage`) — S2, S3 — **done, commit d24eb52**
- T2 — `config.py` (pydantic-settings, env-driven; `HF_HOME=./.models`) + `store.py` (QdrantDocumentStore factory with sparse support) + `sparse.py` (BM25 hashed sparse embedder) — S1, S2 — **done, commit d24eb52**
- T3 — `indexing.py`: ingest → chunk → BGE-M3 dense embed + BM25 sparse embed → Qdrant writer — S1 — **done, commit fcd6346**
- T4 — `retrieval.py`: hybrid retrieve → RRF (server-side) → BGE-reranker-v2-m3 → prompt → DeepSeek (OpenAI-compatible) → answer — S1, S5 — **done, commit fcd6346**
- T5 — `cli.py`: `rag-hybrid index <path>` / `rag-hybrid retrieve <q>` / `rag-hybrid ask <q>` — S1 — **done, commit fcd6346**
- T6 — tests (config + pipeline wiring) + `evals/` DeepEval harness + golden-set stub — S4 — **done (17/17 unit tests green), commit cfb8297**
- T7 — README quickstart + end-to-end smoke (Docker Qdrant up, index `sample/`, retrieve without LLM key; `ask` requires `DEEPSEEK_API_KEY`) — all — **done, commit 3e05dd9** (SMOKE-OK: 2 files -> 2 chunks indexed, collection `text-dense` 1024/cosine + `text-sparse` modifier `idf`, semantic retrieve 0.972/0.983, keyword retrieve 0.955 on the right doc; `ask` leg built and unit-verified, full run pending the user's DEEPSEEK_API_KEY)

## Log

- L1 (2026-09-07, user): "oye quiero hacer algo sobre rag, entiendo que hay muchos investiga sobre ello y particularmente sobre hybrid rag y lightrag. ademas sobre qdran"
- L2 (2026-09-07, user): "seria semi-local. alguien me recomendo esto dime que opinas en comparacion a lo que estamos viendo: Qdrant + BGE-M3 + BM25 + RRF + BGE-reranker-v2-m3 + tu LLM."
- L3 (2026-09-11, user): "haz un deep research sobre frameworks de rag" → decision: Haystack 2.x (4-layer taxonomy, benchmark overhead negligible, eval stack DeepEval CI + RAGAS nightly + TruLens).
- L4 (2026-11-13): proposal to scaffold v1 (Haystack 2.x + Qdrant hybrid pipeline + DeepEval day one) → user: "si dale"
- L5 (2026-11-13, user): "ah oye vas a usar docker para no ensuciar mi maquina cierto?" → isolation plan confirmed: Docker Qdrant + project-contained venv/models/data; cleanup = `docker compose down -v` + delete folder.
- L6 (2026-11-13, user): "vamos a usar deepseek" → S5.
- L7 (2026-11-13, verification): resolved against installed versions — haystack-ai 2.31.1, qdrant-haystack 7.0.0 (namespace `haystack_integrations`), sentence-transformers 6.3, deepeval 4.3.4. RRF is server-side inside `QdrantHybridRetriever` (`FusionQuery(RRF)` over dense+sparse prefetches); `sparse_idf=True` only sets Qdrant's `Modifier.IDF`, so the client supplies both embeddings → custom BM25 hashed sparse components (no fastembed dependency).
- L8 (2026-11-13, fixes from the wiring-test cycle): added `accelerate` + `sentencepiece` deps (ranker optional deps); ST embedders imported from `sentence-transformers-haystack` (core ones deprecated for 3.0); ranker swapped to `SentenceTransformersSimilarityRanker` (TransformersSimilarityRanker is legacy); `PromptBuilder` got `required_variables`; pipeline graph is a networkx MultiDiGraph (3-tuple edges).
- L9 (2026-11-13, E2E incident): index failed with `Unknown arguments: ['init_from']` — qdrant-client 1.19/1.20 removed the `init_from` param and assert-reject unknown kwargs locally; qdrant-haystack 7.0 still passes it. Fix (commit 3e05dd9): pin `qdrant-client>=1.12,<1.15` (1.14.3 resolved) + server image `v1.14.1` (same minor line; newer servers also warn on gap). Second incident in the same fix: bind-mount files were root-owned (breaks the rm -rf isolation promise) -> container now runs as 1000:1000 with `QDRANT__STORAGE__SNAPSHOTS_PATH` redirected into the mounted storage (image's ./snapshots is root-owned and panics otherwise).
- L10 (2026-11-13, post-smoke polish, commit 3e05dd9): BM25 document embedder switched to `dataclasses.replace` (in-place mutation warned); ranker import moved to `haystack_integrations.components.rankers.sentence_transformers`. Final checks: 17/17 pytest green, retrieve re-verified (0.983).

Risk: none (greenfield project: all code is new, no pre-existing behavior modified; verified by 17 unit tests + full E2E smoke).
