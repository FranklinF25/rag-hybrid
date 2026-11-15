// Package openrouter provides a minimal, stdlib-only client for the
// OpenRouter API: the OpenAI-compatible /embeddings endpoint and the
// /rerank endpoint. The client is stateless apart from one shared
// http.Client, mirroring the Python reference in src/rag_hybrid/openrouter.py.
// It implements the engine.Embedder and engine.Reranker ports.
package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"rag-hybrid/internal/engine"
)

// DefaultBaseURL is the OpenRouter API root used when NewClient gets "".
const DefaultBaseURL = "https://openrouter.ai/api/v1"

// defaultTimeout bounds each HTTP request, matching the Python clients.
const defaultTimeout = 60 * time.Second

// errorSnippetLen caps the response-body excerpt included in error messages.
const errorSnippetLen = 200

// Client talks to OpenRouter. It is safe for concurrent use and satisfies
// the engine.Embedder and engine.Reranker ports.
type Client struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

// NewClient builds a Client. An empty baseURL falls back to DefaultBaseURL.
// All requests share one http.Client, so connections are pooled.
func NewClient(apiKey, baseURL string) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: defaultTimeout},
	}
}

// Compile-time proof that the client satisfies both ports.
var (
	_ engine.Embedder = (*Client)(nil)
	_ engine.Reranker = (*Client)(nil)
)

// ScoredDoc is one rerank result. Alias of the domain type: Index points
// into the documents slice that was passed to Rerank; Score is the
// cross-encoder relevance score.
type ScoredDoc = engine.ScoredDoc

type embeddingsRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
}

type embeddingsResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n"`
}

type rerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

// EmbedTexts sends all texts in a single /embeddings request (the caller is
// responsible for batching) and returns the embeddings ordered by the
// response's "index" field, making the input-order guarantee explicit.
func (c *Client) EmbedTexts(ctx context.Context, model string, texts []string) ([][]float64, error) {
	if err := c.ensureKey(); err != nil {
		return nil, err
	}
	var resp embeddingsResponse
	err := c.post(ctx, "/embeddings", &embeddingsRequest{
		Model:          model,
		Input:          texts,
		EncodingFormat: "float",
	}, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("openrouter /embeddings: got %d embeddings for %d input texts", len(resp.Data), len(texts))
	}
	ordered := make([][]float64, len(resp.Data))
	for _, item := range resp.Data {
		if item.Index < 0 || item.Index >= len(ordered) {
			return nil, fmt.Errorf("openrouter /embeddings: index %d out of range for %d texts", item.Index, len(texts))
		}
		ordered[item.Index] = item.Embedding
	}
	return ordered, nil
}

// Rerank posts the documents to /rerank and returns up to topN ScoredDocs
// sorted by score descending. The server already truncates via top_n, but
// the contract (sort + slice) is enforced locally so it holds even if a
// server returns extra or unordered results. Out-of-range indices from a
// misbehaving server are skipped defensively.
func (c *Client) Rerank(ctx context.Context, model, query string, documents []string, topN int) ([]ScoredDoc, error) {
	if err := c.ensureKey(); err != nil {
		return nil, err
	}
	if len(documents) == 0 {
		return []ScoredDoc{}, nil
	}
	var resp rerankResponse
	err := c.post(ctx, "/rerank", &rerankRequest{
		Model:     model,
		Query:     query,
		Documents: documents,
		TopN:      topN,
	}, &resp)
	if err != nil {
		return nil, err
	}
	scored := make([]ScoredDoc, 0, len(resp.Results))
	for _, r := range resp.Results {
		if r.Index < 0 || r.Index >= len(documents) {
			continue
		}
		scored = append(scored, ScoredDoc{Index: r.Index, Score: r.RelevanceScore})
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if topN >= 0 && len(scored) > topN {
		scored = scored[:topN]
	}
	return scored, nil
}

// ensureKey fails fast with an actionable message, mirroring the Python
// client which checks the key at run time rather than at construction.
func (c *Client) ensureKey() error {
	if c.apiKey == "" {
		return fmt.Errorf("openrouter: OPENROUTER_API_KEY is not set; get a key at https://openrouter.ai/settings/keys")
	}
	return nil
}

// post sends body as JSON to path, checks the status code and decodes a 2xx
// response into out. Non-2xx responses become an error carrying the status
// and a trimmed body snippet.
func (c *Client) post(ctx context.Context, path string, body, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("openrouter %s: encoding request: %w", path, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("openrouter %s: %w", path, err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("openrouter %s: %w", path, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("openrouter %s: reading response: %w", path, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("openrouter %s: status %d: %s", path, resp.StatusCode, snippet(data))
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("openrouter %s: decoding response: %w", path, err)
	}
	return nil
}

// snippet returns a whitespace-trimmed body excerpt for error messages.
func snippet(body []byte) string {
	s := strings.TrimSpace(string(body))
	if len(s) > errorSnippetLen {
		s = s[:errorSnippetLen]
	}
	return s
}
