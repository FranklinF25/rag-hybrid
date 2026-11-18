package engine

import (
	"strings"
	"testing"
)

// The conversational prompt must keep everything cache-stable at the
// front: static instructions first, then the append-only history, and
// only then the per-turn material (retrieved documents, question).
// DeepSeek's automatic prefix cache matches in 64-token blocks, so a
// static-first ordering is what lets consecutive turns of one
// conversation share a growing cacheable prefix.
func TestContextualPromptStaticPrefixFirst(t *testing.T) {
	history := []Turn{
		{Role: "user", Content: "What does RRF stand for?"},
		{Role: "assistant", Content: "Reciprocal Rank Fusion."},
	}
	hits := []Hit{{Source: "s", Content: "some chunk text", Score: 0.9}}

	prompt := buildContextualPrompt("and the second one?", history, hits)

	instructions := strings.Index(prompt, "Answer the question using only the documents below")
	historyBlock := strings.Index(prompt, "Recent conversation")
	documents := strings.Index(prompt, "[1] some chunk text")
	question := strings.Index(prompt, "Question: and the second one?")
	if instructions < 0 || historyBlock < 0 || documents < 0 || question < 0 {
		t.Fatalf("prompt missing a section:\n%s", prompt)
	}
	if !(instructions < historyBlock && historyBlock < documents && documents < question) {
		t.Fatalf("prompt sections out of cache-friendly order: instructions@%d history@%d documents@%d question@%d\n%s",
			instructions, historyBlock, documents, question, prompt)
	}
}
