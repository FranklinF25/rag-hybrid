package engine

import (
	"context"
	"fmt"
	"strings"

	"rag-hybrid/internal/qdrant"
)

// PrepareQuery is stage one of a conversational answer: rewrite the
// question into a standalone query against the recent history (fast-path:
// no rewrite and no extra LLM call when there is no history), retrieve with
// the REWRITTEN query (so cited sources match what was actually searched)
// and apply the relevance threshold.
//
// Returns the rewritten query, the surviving hits (possibly empty), and a
// non-empty localAnswer when the system can answer deterministically
// without generation (nothing retrieved or below threshold). Callers either
// show localAnswer directly or continue with Generate/GenerateStream.
func (s *Searcher) PrepareQuery(ctx context.Context, question string, history []Turn) (string, []qdrant.Hit, string, error) {
	rewritten, err := RewriteQuery(ctx, s.llm, s.cfg.LLMModel, history, question)
	if err != nil {
		return "", nil, "", fmt.Errorf("rewriting: %w", err)
	}
	if strings.TrimSpace(rewritten) == "" {
		rewritten = question
	}

	hits, err := s.Retrieve(ctx, rewritten)
	if err != nil {
		return rewritten, nil, "", err
	}
	if len(hits) == 0 {
		return rewritten, nil, "No indexed documents matched the question.", nil
	}
	if s.cfg.RerankScoreThreshold > 0 && hits[0].Score < s.cfg.RerankScoreThreshold {
		local := fmt.Sprintf("I don't have relevant indexed documents for that question (best relevance %.2f).", hits[0].Score)
		return rewritten, nil, local, nil
	}
	return rewritten, hits, "", nil
}

// Generate is stage two (non-streaming): produce the grounded answer for
// the ORIGINAL question with the recent history for coherence.
func (s *Searcher) Generate(ctx context.Context, question string, history []Turn, hits []qdrant.Hit) (string, error) {
	return s.llm.Complete(ctx, s.cfg.LLMModel, buildContextualPrompt(question, history, hits), s.cfg.GenerationMaxTokens)
}

// GenerateStream is stage two (streaming): same prompt, deltas via onDelta.
func (s *Searcher) GenerateStream(ctx context.Context, question string, history []Turn, hits []qdrant.Hit, onDelta func(string) error) (string, error) {
	return s.llm.CompleteStream(ctx, s.cfg.LLMModel, buildContextualPrompt(question, history, hits), s.cfg.GenerationMaxTokens, onDelta)
}

// AskWithContext is the one-shot conversational answer (non-streaming):
// PrepareQuery + Generate. Returns the answer, its sources and the
// rewritten query (useful for debugging/transparency).
func (s *Searcher) AskWithContext(ctx context.Context, question string, history []Turn) (Answer, string, error) {
	rewritten, hits, localAnswer, err := s.PrepareQuery(ctx, question, history)
	if err != nil {
		return Answer{}, rewritten, err
	}
	if localAnswer != "" {
		return Answer{Text: localAnswer}, rewritten, nil
	}
	text, err := s.Generate(ctx, question, history, hits)
	if err != nil {
		return Answer{Sources: hits}, rewritten, err
	}
	return Answer{Text: text, Sources: hits}, rewritten, nil
}

// buildContextualPrompt renders the grounded prompt with the recent
// conversation as a preamble, then the numbered documents, then the
// question. History turns are truncated to keep the prompt lean.
func buildContextualPrompt(question string, history []Turn, hits []qdrant.Hit) string {
	var b strings.Builder
	if len(history) > 0 {
		b.WriteString("Recent conversation (for context only):\n")
		for _, turn := range history {
			fmt.Fprintf(&b, "%s: %s\n", turn.Role, truncate(turn.Content, 300))
		}
		b.WriteString("\n")
	}
	b.WriteString("Answer the question using only the documents below.\n")
	b.WriteString("If the documents do not contain the answer, say you don't know.\n")
	b.WriteString("Cite the documents you use like [1], [2].\n\n")
	for i, hit := range hits {
		fmt.Fprintf(&b, "[%d] %s\n\n", i+1, hit.Content)
	}
	fmt.Fprintf(&b, "Question: %s\nAnswer:", question)
	return b.String()
}

func truncate(text string, max int) string {
	if len(text) <= max {
		return text
	}
	return text[:max] + "…"
}
