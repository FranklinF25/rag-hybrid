package engine

import (
	"context"
	"fmt"
	"strings"

	"rag-hybrid/internal/bm25"
	"rag-hybrid/internal/config"
	"rag-hybrid/internal/openrouter"
	"rag-hybrid/internal/qdrant"
)

// Searcher runs the query-side flows: hybrid retrieval, reranking and
// answer generation.
type Searcher struct {
	cfg  *config.Config
	open *openrouter.Client
	qd   *qdrant.Client
	llm  *DeepSeek
}

// NewSearcher wires a Searcher from config.
func NewSearcher(cfg *config.Config) *Searcher {
	return &Searcher{
		cfg:  cfg,
		open: openrouter.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL),
		qd:   qdrant.NewClient(cfg.QdrantURL, cfg.QdrantAPIKey, cfg.QdrantCollection),
		llm:  NewDeepSeek(cfg.DeepSeekAPIKey, cfg.LLMBaseURL),
	}
}

// Retrieve answers nothing: it embeds the question (dense + BM25 sparse),
// runs Qdrant's server-side RRF hybrid query and reranks the candidates
// with rerank-3-lite. Returned hits carry the reranker score, best first.
func (s *Searcher) Retrieve(ctx context.Context, question string) ([]qdrant.Hit, error) {
	embeddings, err := s.open.EmbedTexts(ctx, s.cfg.EmbedModel, []string{question})
	if err != nil {
		return nil, fmt.Errorf("embedding question: %w", err)
	}
	sparse := bm25.Sparse(question)

	hits, err := s.qd.HybridQuery(ctx, embeddings[0], sparse.Indices, sparse.Values, s.cfg.RetrievalTopK)
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
	scored, err := s.open.Rerank(ctx, s.cfg.RerankerModel, question, documents, s.cfg.RerankTopK)
	if err != nil {
		return nil, fmt.Errorf("reranking: %w", err)
	}

	results := make([]qdrant.Hit, 0, len(scored))
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
	Sources []qdrant.Hit
}

// Ask runs Retrieve and then generates a grounded answer with DeepSeek.
func (s *Searcher) Ask(ctx context.Context, question string) (Answer, error) {
	hits, err := s.Retrieve(ctx, question)
	if err != nil {
		return Answer{}, err
	}
	if len(hits) == 0 {
		return Answer{Text: "No indexed documents matched the question.", Sources: nil}, nil
	}
	prompt := buildPrompt(question, hits)
	text, err := s.llm.Complete(ctx, s.cfg.LLMModel, prompt, s.cfg.GenerationMaxTokens)
	if err != nil {
		return Answer{Sources: hits}, err
	}
	return Answer{Text: text, Sources: hits}, nil
}

// buildPrompt is the same grounded template as the Python version.
func buildPrompt(question string, hits []qdrant.Hit) string {
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
