package engine

import (
	"context"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Turn is one conversation turn, in chronological order, used as context
// for rewriting a follow-up question. Role is "user" or "assistant".
type Turn struct {
	Role    string
	Content string
}

// rewriteTurnMaxChars caps each history turn in the rewrite prompt so a
// long conversation cannot blow the context window.
const rewriteTurnMaxChars = 400

// RewriteQuery rewrites a follow-up question into ONE standalone question
// that retrieval can match without the surrounding conversation: pronouns
// and ellipsis in the question are resolved against the recent history.
// The first question of a conversation (empty history) is returned
// unchanged without calling the LLM (fast-path). The model is instructed
// to rewrite only — never to answer and never to invent facts.
func RewriteQuery(ctx context.Context, llm LLM, model string, history []Turn, question string) (string, error) {
	if len(history) == 0 {
		return question, nil
	}

	var prompt strings.Builder
	prompt.WriteString(
		"You rewrite follow-up questions for a retrieval system. " +
			"This is a rewriting task, NOT an answering task: you must not answer the question.\n\n" +
			"You will receive the recent conversation turns followed by the new question. " +
			"Output ONE standalone question that:\n" +
			"- preserves the intent of the new question\n" +
			"- resolves pronouns and ellipsis against the conversation history\n" +
			"- adds no facts that are not already present in the history or the new question\n\n" +
			"Output ONLY the rewritten question text: no quotes, no prefix, no explanation.\n\n" +
			"Conversation:\n")
	for _, turn := range history {
		fmt.Fprintf(&prompt, "%s: %s\n", turn.Role, truncateTurn(turn.Content, rewriteTurnMaxChars))
	}
	fmt.Fprintf(&prompt, "\nNew question: %s", question)

	rewritten, err := llm.Complete(ctx, model, prompt.String(), 200)
	if err != nil {
		return "", err
	}
	rewritten = strings.TrimSpace(strings.Trim(strings.TrimSpace(rewritten), `"'`))
	if rewritten == "" {
		// A silent model must never lose the user's question.
		return question, nil
	}
	return rewritten, nil
}

// truncateTurn cuts content to max bytes, backing off to a rune boundary
// so a multi-byte character is never split in half.
func truncateTurn(content string, max int) string {
	if len(content) <= max {
		return content
	}
	cut := content[:max]
	for len(cut) > 0 && !utf8.ValidString(cut) {
		cut = cut[:len(cut)-1]
	}
	return cut
}
