package engine

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"rag-hybrid/internal/deepseek"
)

// fakeRewriteServer answers a non-streaming chat completion with the given
// content and records the raw request body for assertions.
func fakeRewriteServer(t *testing.T, content string) (*httptest.Server, *string) {
	t.Helper()
	var requestBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload, _ := io.ReadAll(r.Body)
		requestBody = string(payload)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": content}},
			},
		})
	}))
	t.Cleanup(server.Close)
	return server, &requestBody
}

func TestRewriteQueryNoHistorySkipsLLM(t *testing.T) {
	// First question of a conversation: rewriting must not even wake the
	// server up.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("rewrite must not call the LLM for empty history, got %s %s", r.Method, r.URL.Path)
	}))
	t.Cleanup(server.Close)
	llm := deepseek.New("test-key", server.URL)

	question := "What is Kappa frequency?"
	got, err := RewriteQuery(context.Background(), llm, "deepseek-chat", nil, question)
	if err != nil {
		t.Fatalf("RewriteQuery: %v", err)
	}
	if got != question {
		t.Errorf("got %q, want the original question %q", got, question)
	}
}

func TestRewriteQueryWithHistory(t *testing.T) {
	server, requestBody := fakeRewriteServer(t, `"  What is the top speed of the Swift?  "`)
	llm := deepseek.New("test-key", server.URL)

	history := []Turn{
		{Role: "user", Content: "What is the fastest production car?"},
		{Role: "assistant", Content: "The Bugatti Chiron Super Sport, at 490 km/h."},
	}
	got, err := RewriteQuery(context.Background(), llm, "deepseek-chat", history, "and the second one?")
	if err != nil {
		t.Fatalf("RewriteQuery: %v", err)
	}
	if want := "What is the top speed of the Swift?"; got != want {
		t.Errorf("got %q, want trimmed rewrite %q", got, want)
	}

	for _, want := range []string{
		"rewriting task, NOT an answering task", // strict no-answer instruction
		"user: What is the fastest production car?",
		"assistant: The Bugatti Chiron Super Sport, at 490 km/h.",
		"New question: and the second one?",
	} {
		if !strings.Contains(*requestBody, want) {
			t.Errorf("request body missing %q:\n%s", want, *requestBody)
		}
	}

	var sent struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
		Stream    bool   `json:"stream"`
	}
	if err := json.Unmarshal([]byte(*requestBody), &sent); err != nil {
		t.Fatalf("decoding request body: %v", err)
	}
	if sent.Model != "deepseek-chat" || sent.MaxTokens != 200 || sent.Stream {
		t.Errorf("rewrite call must be plain Complete-style: %+v", sent)
	}
}

func TestRewriteQueryEmptyResponseFallsBack(t *testing.T) {
	server, _ := fakeRewriteServer(t, "")
	llm := deepseek.New("test-key", server.URL)

	question := "How does it compare to the Vue?"
	history := []Turn{{Role: "user", Content: "Tell me about the Swift."}}
	got, err := RewriteQuery(context.Background(), llm, "deepseek-chat", history, question)
	if err != nil {
		t.Fatalf("RewriteQuery: %v", err)
	}
	if got != question {
		t.Errorf("got %q, want fallback to the original question %q", got, question)
	}
}
