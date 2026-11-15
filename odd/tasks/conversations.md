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

- C1 — go.mod dep + feature doc + split — **done** (note: dep was swept by go mod tidy before the importing code existed; re-added at C2 verification)
- C2 — internal/store: SQLite store + 9 tests — **done (worker C, parent-verified 9/9)**
- C3 — DeepSeek streaming (SSE) + RewriteQuery + 6 tests — **done (worker D, parent-verified)**
- C4 — engine staged flow + server endpoints + retention loop + Flush fix — **done**
- C5 — UI conversations panel + streaming render + searched-as line — **done**
- C6 — E2E + docs — **done**: dependent follow-up rewritten ('and what value does that constant use?' → 'What value does the constant in Reciprocal Rank Fusion use?'), 21 streamed deltas, 'value 60 [1]'; conversations CRUD verified; sqlite persisted in ./.data. Commit c85cf6a.

## Log

- L1 (2026-11-14, user): pushed back on ephemeral design ("las conversaciones se pierden no? asi es como funcionan las implementaciones de RAG a empresas?") — parent corrected the plan: durable store from day one, RAM design dropped.
- L2 (2026-11-14, user): "si dale con el diseño completo".
- L3 (2026-11-15, results): commit c85cf6a. Incidents: (1) go mod tidy swept modernc.org/sqlite added before internal/store existed — re-added at verification; (2) logging middleware's statusRecorder hid http.Flusher → SSE endpoint returned 500 'streaming unsupported' — fixed with Flush() propagation; (3) scratch container had no writable DATA_DIR → ENV DATA_DIR=/data + compose volume ./.data:/data.

Risk: none — all new code, 5/5 packages tested, in-container E2E including the rewrite chain.
