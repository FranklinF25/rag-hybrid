package openrouter

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// recorded captures the last request a test server received.
type recorded struct {
	req  *http.Request
	body []byte
}

// newTestServer starts an httptest server that records each request and
// replies with status+reply. A non-nil handler overrides the canned reply.
func newTestServer(t *testing.T, status int, reply string, handler http.HandlerFunc) (*httptest.Server, *recorded) {
	t.Helper()
	rec := &recorded{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.req = r
		rec.body, _ = io.ReadAll(r.Body)
		if handler != nil {
			handler(w, r)
			return
		}
		w.WriteHeader(status)
		fmt.Fprint(w, reply)
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

func TestEmbedTextsRequestAndOrdering(t *testing.T) {
	// Data shuffled relative to "index": the client must reorder by index.
	srv, rec := newTestServer(t, http.StatusOK, `{
		"data": [
			{"embedding": [0.3, 0.4], "index": 1},
			{"embedding": [0.1, 0.2], "index": 0}
		],
		"model": "qwen/qwen3-embedding-8b",
		"usage": {"prompt_tokens": 8, "total_tokens": 8}
	}`, nil)
	c := NewClient("test-key", srv.URL)

	got, err := c.EmbedTexts(context.Background(), "qwen/qwen3-embedding-8b", []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedTexts: %v", err)
	}
	want := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	if len(got) != len(want) {
		t.Fatalf("got %d embeddings, want %d", len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("embedding %d: got %v, want %v", i, got[i], want[i])
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("embedding %d: got %v, want %v", i, got[i], want[i])
			}
		}
	}

	if rec.req.URL.Path != "/embeddings" {
		t.Errorf("path = %q, want /embeddings", rec.req.URL.Path)
	}
	if rec.req.Method != http.MethodPost {
		t.Errorf("method = %q, want POST", rec.req.Method)
	}
	if auth := rec.req.Header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", auth, "Bearer test-key")
	}
	var sent map[string]any
	if err := json.Unmarshal(rec.body, &sent); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	if sent["model"] != "qwen/qwen3-embedding-8b" {
		t.Errorf("model = %v, want qwen/qwen3-embedding-8b", sent["model"])
	}
	if sent["encoding_format"] != "float" {
		t.Errorf("encoding_format = %v, want float", sent["encoding_format"])
	}
	input, ok := sent["input"].([]any)
	if !ok || len(input) != 2 || input[0] != "hello" || input[1] != "world" {
		t.Errorf("input = %v, want [hello world]", sent["input"])
	}
}

func TestEmbedTextsMissingKey(t *testing.T) {
	srv, _ := newTestServer(t, http.StatusOK, `{}`, nil)
	c := NewClient("", srv.URL)
	_, err := c.EmbedTexts(context.Background(), "m", []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "OPENROUTER_API_KEY") {
		t.Fatalf("error = %v, want one mentioning OPENROUTER_API_KEY", err)
	}
}

func TestEmbedTextsCountMismatch(t *testing.T) {
	srv, _ := newTestServer(t, http.StatusOK,
		`{"data":[{"embedding":[1],"index":0}]}`, nil)
	c := NewClient("k", srv.URL)
	_, err := c.EmbedTexts(context.Background(), "m", []string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "1 embeddings for 2") {
		t.Fatalf("error = %v, want count mismatch", err)
	}
}

func TestEmbedTextsIndexOutOfRange(t *testing.T) {
	srv, _ := newTestServer(t, http.StatusOK,
		`{"data":[{"embedding":[1],"index":0},{"embedding":[2],"index":5}]}`, nil)
	c := NewClient("k", srv.URL)
	_, err := c.EmbedTexts(context.Background(), "m", []string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("error = %v, want out-of-range index", err)
	}
}

func TestRerankRequestSortAndTruncate(t *testing.T) {
	// Unordered, containing an out-of-range index (7) and a negative one;
	// the client must skip those, sort desc and truncate to topN.
	srv, rec := newTestServer(t, http.StatusOK, `{
		"results": [
			{"index": 1, "relevance_score": 0.55},
			{"index": 7, "relevance_score": 0.99},
			{"index": 0, "relevance_score": 0.80},
			{"index": -1, "relevance_score": 0.77}
		]
	}`, nil)
	c := NewClient("test-key", srv.URL)

	docs := []string{"alpha", "beta", "gamma"}
	got, err := c.Rerank(context.Background(), "mongodb/rerank-3-lite", "q", docs, 2)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	want := []ScoredDoc{{Index: 0, Score: 0.80}, {Index: 1, Score: 0.55}}
	if len(got) != len(want) {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %+v, want %+v", got, want)
		}
	}

	if rec.req.URL.Path != "/rerank" {
		t.Errorf("path = %q, want /rerank", rec.req.URL.Path)
	}
	if auth := rec.req.Header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", auth, "Bearer test-key")
	}
	var sent map[string]any
	if err := json.Unmarshal(rec.body, &sent); err != nil {
		t.Fatalf("request body is not JSON: %v", err)
	}
	if sent["model"] != "mongodb/rerank-3-lite" {
		t.Errorf("model = %v, want mongodb/rerank-3-lite", sent["model"])
	}
	if sent["query"] != "q" {
		t.Errorf("query = %v, want q", sent["query"])
	}
	if sent["top_n"] != float64(2) {
		t.Errorf("top_n = %v, want 2", sent["top_n"])
	}
	if docs2, ok := sent["documents"].([]any); !ok || len(docs2) != 3 || docs2[2] != "gamma" {
		t.Errorf("documents = %v, want [alpha beta gamma]", sent["documents"])
	}
}

func TestRerankEmptyDocuments(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer srv.Close()
	c := NewClient("k", srv.URL)
	got, err := c.Rerank(context.Background(), "m", "q", nil, 5)
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("got %+v, want empty", got)
	}
	if called {
		t.Error("Rerank made a network call for zero documents")
	}
}

func TestNon2xxIncludesStatusAndSnippet(t *testing.T) {
	long := `{"error":{"message":"` + strings.Repeat("x", 400) + `"}}`
	srv, _ := newTestServer(t, http.StatusTooManyRequests, long, nil)
	c := NewClient("k", srv.URL)

	_, err := c.EmbedTexts(context.Background(), "m", []string{"x"})
	if err == nil {
		t.Fatal("EmbedTexts: want error for 429")
	}
	if !strings.Contains(err.Error(), "status 429") {
		t.Errorf("error %q lacks status", err)
	}
	if len(err.Error()) > 300 { // prefix + "status 429: " + at most 200 body chars
		t.Errorf("error message too long (%d chars); snippet not trimmed", len(err.Error()))
	}
	if !strings.Contains(err.Error(), `{"error":{"message":"`) {
		t.Errorf("error %q lacks body excerpt", err)
	}
	if !strings.Contains(err.Error(), strings.Repeat("x", 100)) {
		t.Errorf("error %q body excerpt is not a meaningful chunk", err)
	}

	_, err = c.Rerank(context.Background(), "m", "q", []string{"d"}, 5)
	if err == nil || !strings.Contains(err.Error(), "status 429") {
		t.Fatalf("Rerank error = %v, want status 429", err)
	}
}
