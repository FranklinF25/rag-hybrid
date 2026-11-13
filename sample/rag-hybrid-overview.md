# rag-hybrid overview

rag-hybrid is a semi-local Retrieval-Augmented Generation (RAG) service.
The retrieval stack runs entirely on the local CPU: Qdrant as the vector
database, BGE-M3 for dense embeddings, BM25 for sparse keyword retrieval,
Reciprocal Rank Fusion (RRF) to combine both result lists, and the
BGE-reranker-v2-m3 cross-encoder to rerank the fused candidates. The final
generation step calls the DeepSeek API (model deepseek-chat), which is
OpenAI-compatible.

The project targets an 8 GB RAM machine. Embedding and reranking models run
in CPU mode, and Qdrant runs in Docker with its storage bind-mounted inside
the repository folder so the host machine stays clean.

The orchestration framework is Haystack 2.x. Indexing flows through a
document splitter, a dense document embedder and a sparse document embedder
into a Qdrant document writer. Query time runs hybrid retrieval, fuses the
dense and sparse rankings with RRF, reranks with the cross-encoder, renders
a prompt with the top documents and asks the LLM for a grounded answer.
