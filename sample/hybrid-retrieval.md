# How hybrid retrieval works

Hybrid retrieval combines two families of search that fail in opposite
ways. Dense vector search (Qwen3-Embedding-8B embeddings via the OpenRouter
API) captures semantic similarity:
it finds paraphrases and synonyms, but it can miss exact identifiers,
acronyms and rare tokens. Sparse keyword search (BM25) matches exact terms
and scores them by inverse document frequency, but it cannot bridge
vocabulary gaps between the question and the document.

Reciprocal Rank Fusion (RRF) merges the two ranked lists without needing
comparable scores. Each document earns a contribution of 1 / (k + rank)
from every list it appears in, where k is a damping constant, typically 60.
A document ranked high in both lists therefore accumulates the most score.
RRF needs no score normalization because it only uses ranks.

After fusion, a cross-encoder reranker (rerank-3-lite via the OpenRouter
API) rescores the
top candidates jointly: it reads the query and the candidate text together
and outputs a relevance score. Cross-encoders are more accurate than
bi-encoders because attention flows across the query-document pair, at the
cost of scoring one pair at a time. This is why reranking happens last, on
a shortlist of roughly twenty candidates, and only the best five documents
survive into the final prompt.

The practical recipe for query time is therefore: retrieve twenty
candidates with hybrid search, fuse them with RRF, rerank with the
cross-encoder, keep the top five, and pass them as context to the
generation LLM.
