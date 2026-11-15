package engine

import (
	"context"
	"fmt"
	"strings"

	"rag-hybrid/internal/bm25"
	"rag-hybrid/internal/config"
)

// Searcher runs the query-side flows: hybrid retrieval, reranking and
// answer generation.
type Searcher struct {
	cfg    *config.Config
	embed  Embedder
	rerank Reranker
	store  VectorStore
	llm    LLM
}

// NewSearcher wires a Searcher from its driven ports. The composition
// root builds the concrete adapters; this package imports none of them.
func NewSearcher(cfg *config.Config, store VectorStore, embed Embedder, rerank Reranker, llm LLM) *Searcher {
	return &Searcher{cfg: cfg, embed: embed, rerank: rerank, store: store, llm: llm}
}

// Retrieve answers nothing: it embeds the question (dense + BM25 sparse),
// runs Qdrant's server-side RRF hybrid query and reranks the candidates
// with rerank-3-lite. Returned hits carry the reranker score, best first.
func (s *Searcher) Retrieve(ctx context.Context, question string) ([]Hit, error) {
	embeddings, err := s.embed.EmbedTexts(ctx, s.cfg.EmbedModel, []string{question})
	if err != nil {
		return nil, fmt.Errorf("embedding question: %w", err)
	}
	sparse := bm25.Sparse(question)

	hits, err := s.store.HybridQuery(ctx, embeddings[0], sparse.Indices, sparse.Values, s.cfg.RetrievalTopK)
	if err != nil {
		return nil, fmt.Errorf("hybrid query: %w", err)
	}
	if len(hits) == 0 {
		return nil, nil
	}

	documents := make([]string, len(hits))
	for i, hit := range hits {
		documents[i] = hit.Content
	}
	scored, err := s.rerank.Rerank(ctx, s.cfg.RerankerModel, question, documents, s.cfg.RerankTopK)
	if err != nil {
		return nil, fmt.Errorf("reranking: %w", err)
	}

	results := make([]Hit, 0, len(scored))
	for _, sd := range scored {
		if sd.Index < 0 || sd.Index >= len(hits) {
			continue
		}
		hit := hits[sd.Index]
		hit.Score = sd.Score
		results = append(results, hit)
	}
	return results, nil
}

// Answer is the full RAG result: generated text plus the cited sources.
type Answer struct {
	Text    string
	Sources []Hit
}

// Ask runs Retrieve and then generates a grounded answer with DeepSeek.
// When the best rerank score stays below cfg.RerankScoreThreshold (and the
// threshold is enabled), it returns a deterministic "not in corpus"
// answer WITHOUT calling the LLM — cheaper, faster, no hallucination path.
func (s *Searcher) Ask(ctx context.Context, question string) (Answer, error) {
	hits, err := s.Retrieve(ctx, question)
	if err != nil {
		return Answer{}, err
	}
	if len(hits) == 0 {
		return Answer{Text: "No indexed documents matched the question.", Sources: nil}, nil
	}
	if s.cfg.RerankScoreThreshold > 0 && hits[0].Score < s.cfg.RerankScoreThreshold {
		return Answer{Text: fmt.Sprintf("I don't have relevant indexed documents for that question (best relevance %.2f).", hits[0].Score), Sources: nil}, nil
	}
	prompt := buildPrompt(question, hits)
	text, err := s.llm.Complete(ctx, s.cfg.LLMModel, prompt, s.cfg.GenerationMaxTokens)
	if err != nil {
		return Answer{Sources: hits}, err
	}
	return Answer{Text: text, Sources: hits}, nil
}

// buildPrompt is the same grounded template as the Python version.
func buildPrompt(question string, hits []Hit) string {
	var b strings.Builder
	b.WriteString("Answer the question using only the documents below.\n")
	b.WriteString("If the documents do not contain the answer, say you don't know.\n")
	b.WriteString("Cite the documents you use like [1], [2].\n\n")
	for i, hit := range hits {
		fmt.Fprintf(&b, "[%d] %s\n\n", i+1, hit.Content)
	}
	fmt.Fprintf(&b, "Question: %s\nAnswer:", question)
	return b.String()
}
