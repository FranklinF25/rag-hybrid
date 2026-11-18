# rag-hybrid overview

rag-hybrid is a semi-local Retrieval-Augmented Generation (RAG) service.
The retrieval stack pairs a hand-written Go BM25 sparse encoder with dense
embeddings from the Qwen3-Embedding-8B model (called through the
OpenRouter API), fuses both result lists with Reciprocal Rank Fusion (RRF)
executed inside Qdrant, and reranks the fused candidates with the
rerank-3-lite cross-encoder, also through OpenRouter. The final generation
step calls the DeepSeek API (model deepseek-chat), which is
OpenAI-compatible.

The service ships as one static Go binary (11 MB scratch image) with the
web UI embedded. Qdrant runs in Docker with its storage bind-mounted
inside the repository folder so the host machine stays clean; embedding,
reranking and generation are API calls, so no ML runtime runs locally.

There is no orchestration framework: the pipeline is hand-written Go on
the standard library only, organized as ports and adapters. Indexing
flows through a chunker, the dense embedding API and the local BM25
encoder into Qdrant's hybrid (dense + sparse) collection. Query time runs
hybrid retrieval, fuses the dense and sparse rankings with server-side
RRF, reranks with rerank-3-lite, renders a prompt with the top documents
and asks the LLM for a grounded answer.
