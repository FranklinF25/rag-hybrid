# rag-hybrid conversations — persistent chat with context-aware retrieval

ChatGPT-style conversation experience on the Go engine: conversations persist in embedded SQLite, follow-up questions get rewritten into standalone queries (context-aware retrieval), answers stream token-by-token (SSE), and retention is a configurable policy. Basis: conversation design discussion of 2026-11-14 (user pushed from ephemeral RAM store to durable store — the right product call).

## Specs

- S1 (user, verbatim): "al ser usuario de chatgpt el cual siempre tiene las conversaciones y vulves y le preguntas y sigue con el hilo de cualquier conversacion" → durable conversation store, reloadable history, follow-ups keep context. NOT ephemeral RAM.
- S2 (user, verbatim): "si dale con el diseño completo" approving the full design: SQLite store (embedded, pure-Go driver, no cgo — binary stays static/scratch) + query rewriting + streaming SSE + configurable retention.
- S3: Two-layer memory (as designed in conversation): (a) STORE — conversations + turns in SQLite; (b) CONTEXT ASSEMBLY — last HISTORY_TURNS (default 4) loaded per question to (1) rewrite the question into a standalone query BEFORE retrieval (fast-path: first question of a conversation skips rewriting entirely), and (2) give the generator recent turns for coherent replies.
- S4: Sources cited must come from the REWRITTEN query's retrieval. Rewrite model = same DeepSeek client, strict no-answer prompt.
- S5: Streaming: POST /api/chat/stream returns SSE (events: conversation, sources, delta, done); non-stream POST /api/chat keeps working (back-compat) and now accepts/returns conversation_id.
- S6: Conversation management endpoints: GET /api/conversations, DELETE /api/conversations?id=, GET /api/conversations/{id}/turns. UI gets a conversations panel (list, select to reload, new chat).
- S7: Retention as policy: RETENTION_DAYS (default 90, 0 = never) enforced by an hourly background purge; deleting is a feature (compliance), consistent with the existing forget semantics.
- S8: New deps allowed: modernc.org/sqlite (pure Go). No cgo. Image stays scratch.

## Tasks

- C1 — go.mod dep + feature doc + split — **parent**
- C2 — internal/store: SQLite store (conversations, turns, CRUD, purge) + tests — **worker C**
- C3 — internal/engine: DeepSeek streaming (SSE) + RewriteQuery (fast-path) + Turn type + tests — **worker D**
- C4 — engine integration: Ask with history/rewrite; server endpoints (chat, chat/stream SSE, conversations CRUD); retention loop; config knobs — **parent**
- C5 — UI: conversations panel, history reload, streaming render — **parent**
- C6 — E2E: follow-up question that only works with rewrite; streaming via curl; retention purge; README + env.example + commit — **parent**

## Log

- L1 (2026-11-14, user): pushed back on ephemeral design ("las conversaciones se pierden no? asi es como funcionan las implementaciones de RAG a empresas?") — parent corrected the plan: durable store from day one, RAM design dropped.
- L2 (2026-11-14, user): "si dale con el diseño completo".
