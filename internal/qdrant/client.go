// Package qdrant provides a minimal, stdlib-only client for the Qdrant REST
// API (v1.14): collection management, hybrid upserts (dense + sparse named
// vectors) and server-side RRF queries. It mirrors the collection layout of
// the Python store in src/rag_hybrid/store.py: a "text-dense" cosine vector
// space plus a "text-sparse" space with the IDF modifier applied at query
// time by the server. The client implements the engine.VectorStore port;
// engine defines the types it exchanges.
package qdrant

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

// defaultBaseURL is the Qdrant REST root used when NewClient gets "".
const defaultBaseURL = "http://localhost:6333"

// defaultTimeout bounds each HTTP request.
const defaultTimeout = 30 * time.Second

// errorSnippetLen caps the response-body excerpt included in error messages.
const errorSnippetLen = 200

// Named vector spaces inside the collection. The sparse space's IDF modifier
// turns stored term frequencies into BM25-style scores at query time.
const (
	denseVectorName  = "text-dense"
	sparseVectorName = "text-sparse"
)

// Client talks to a single Qdrant collection. It is safe for concurrent use
// and satisfies the engine.VectorStore port.
type Client struct {
	baseURL    string
	apiKey     string
	collection string
	http       *http.Client
}

// NewClient builds a Client for one collection. An empty baseURL falls back
// to the local default; an empty apiKey means no auth header is sent (all
// requests share one http.Client for connection pooling).
func NewClient(baseURL, apiKey, collection string) *Client {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		collection: collection,
		http:       &http.Client{Timeout: defaultTimeout},
	}
}

// Compile-time proof that the client satisfies the VectorStore port.
var _ engine.VectorStore = (*Client)(nil)

// Point is one document chunk ready for a hybrid upsert. Alias of the
// domain type so callers of this package rarely need to import engine.
type Point = engine.Point

// Hit is one row of a hybrid query result.
type Hit = engine.Hit

// vectorParams is the config of one named dense vector space.
type vectorParams struct {
	Size     int    `json:"size"`
	Distance string `json:"distance"`
}

// sparseParams configures a named sparse vector space. Modifier "idf" makes
// Qdrant weigh sparse terms by inverse document frequency at query time.
type sparseParams struct {
	Modifier string `json:"modifier"`
}

// createCollectionRequest is the exact body verified against the running
// Qdrant 1.14.1 server (see the Python reference collection).
type createCollectionRequest struct {
	Vectors       map[string]vectorParams `json:"vectors"`
	SparseVectors map[string]sparseParams `json:"sparse_vectors"`
}

// sparseVectorBody is Qdrant's sparse vector representation.
type sparseVectorBody struct {
	Indices []uint32  `json:"indices"`
	Values  []float64 `json:"values"`
}

type pointVectorBody struct {
	Dense  []float64        `json:"text-dense"`
	Sparse sparseVectorBody `json:"text-sparse"`
}

type pointPayloadBody struct {
	Content string `json:"content"`
	Source  string `json:"source"`
}

type upsertPointBody struct {
	ID      string           `json:"id"`
	Vector  pointVectorBody  `json:"vector"`
	Payload pointPayloadBody `json:"payload"`
}

type upsertRequest struct {
	Points []upsertPointBody `json:"points"`
}

// sparsePrefetch is one prefetch leg over the sparse space. Verified against
// the live server: the query value is the bare {"indices","values"} object;
// Qdrant 1.14.1 rejects the older {"sparse_vector":{...}} wrapper here.
type sparsePrefetch struct {
	Query sparseVectorBody `json:"query"`
	Using string           `json:"using"`
	Limit int              `json:"limit"`
}

// densePrefetch is one prefetch leg over the dense space; its query value is
// a plain float array.
type densePrefetch struct {
	Query []float64 `json:"query"`
	Using string    `json:"using"`
	Limit int       `json:"limit"`
}

// fusionQuery asks the server to fuse the prefetch legs with RRF.
type fusionQuery struct {
	Fusion string `json:"fusion"`
}

type hybridQueryRequest struct {
	Prefetch    []any       `json:"prefetch"`
	Query       fusionQuery `json:"query"`
	Limit       int         `json:"limit"`
	WithPayload bool        `json:"with_payload"`
}

type queryResponse struct {
	Result struct {
		Points []struct {
			ID      json.RawMessage `json:"id"` // Qdrant returns UUID strings or unsigned ints
			Score   float64         `json:"score"`
			Payload struct {
				Content string `json:"content"`
				Source  string `json:"source"`
			} `json:"payload"`
		} `json:"points"`
	} `json:"result"`
}

// EnsureCollection makes sure the collection exists. It GETs the collection
// first and only PUTs the create body (dense cosine space + sparse IDF space)
// on a 404, so an existing collection is left untouched and the call is
// idempotent.
func (c *Client) EnsureCollection(ctx context.Context, dim int) error {
	path := "/collections/" + c.collection
	status, body, err := c.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	if status == http.StatusOK {
		return nil
	}
	if status != http.StatusNotFound {
		return c.statusError(http.MethodGet, path, status, body)
	}
	create := createCollectionRequest{
		Vectors:       map[string]vectorParams{denseVectorName: {Size: dim, Distance: "Cosine"}},
		SparseVectors: map[string]sparseParams{sparseVectorName: {Modifier: "idf"}},
	}
	status, body, err = c.do(ctx, http.MethodPut, path, create)
	if err != nil {
		return err
	}
	return c.statusError(http.MethodPut, path, status, body)
}

// DropCollection deletes the collection. A 404 is treated as success so the
// call is idempotent (the collection is already gone).
func (c *Client) DropCollection(ctx context.Context) error {
	path := "/collections/" + c.collection
	status, body, err := c.do(ctx, http.MethodDelete, path, nil)
	if err != nil {
		return err
	}
	if status == http.StatusNotFound {
		return nil
	}
	return c.statusError(http.MethodDelete, path, status, body)
}

// Upsert writes points with wait=true so the call returns once Qdrant has
// applied them (no eventual-consistency surprise for the caller).
func (c *Client) Upsert(ctx context.Context, points []Point) error {
	path := "/collections/" + c.collection + "/points?wait=true"
	req := upsertRequest{Points: make([]upsertPointBody, 0, len(points))}
	for _, p := range points {
		req.Points = append(req.Points, upsertPointBody{
			ID: p.ID,
			Vector: pointVectorBody{
				Dense:  p.Dense,
				Sparse: sparseVectorBody{Indices: p.SparseIndices, Values: p.SparseValues},
			},
			Payload: pointPayloadBody{Content: p.Content, Source: p.Source},
		})
	}
	status, body, err := c.do(ctx, http.MethodPut, path, req)
	if err != nil {
		return err
	}
	return c.statusError(http.MethodPut, path, status, body)
}

// HybridQuery runs a two-leg prefetch (sparse, dense) fused server-side with
// reciprocal rank fusion and returns the payload-bearing hits.
func (c *Client) HybridQuery(ctx context.Context, dense []float64, sparseIndices []uint32, sparseValues []float64, limit int) ([]Hit, error) {
	path := "/collections/" + c.collection + "/points/query"
	req := hybridQueryRequest{
		Prefetch: []any{
			sparsePrefetch{
				Query: sparseVectorBody{Indices: sparseIndices, Values: sparseValues},
				Using: sparseVectorName,
				Limit: limit,
			},
			densePrefetch{Query: dense, Using: denseVectorName, Limit: limit},
		},
		Query:       fusionQuery{Fusion: "rrf"},
		Limit:       limit,
		WithPayload: true,
	}
	status, body, err := c.do(ctx, http.MethodPost, path, req)
	if err != nil {
		return nil, err
	}
	if err := c.statusError(http.MethodPost, path, status, body); err != nil {
		return nil, err
	}
	var resp queryResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("qdrant POST %s: decoding response: %w", path, err)
	}
	hits := make([]Hit, 0, len(resp.Result.Points))
	for _, p := range resp.Result.Points {
		hits = append(hits, Hit{
			ID:      parsePointID(p.ID),
			Score:   p.Score,
			Content: p.Payload.Content,
			Source:  p.Payload.Source,
		})
	}
	return hits, nil
}

// do executes one JSON request and returns the status code and raw body.
// A non-nil error means the request never completed (transport failure);
// HTTP error statuses are for the caller to interpret.
func (c *Client) do(ctx context.Context, method, path string, body any) (int, []byte, error) {
	var reader io.Reader
	if body != nil {
		payload, err := json.Marshal(body)
		if err != nil {
			return 0, nil, fmt.Errorf("qdrant %s %s: encoding request: %w", method, path, err)
		}
		reader = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return 0, nil, fmt.Errorf("qdrant %s %s: %w", method, path, err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("api-key", c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("qdrant %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, fmt.Errorf("qdrant %s %s: reading response: %w", method, path, err)
	}
	return resp.StatusCode, data, nil
}

// statusError turns a non-2xx status into an error carrying the status code
// and a trimmed body snippet; 2xx returns nil.
func (c *Client) statusError(method, path string, status int, body []byte) error {
	if status >= 200 && status < 300 {
		return nil
	}
	return fmt.Errorf("qdrant %s %s: status %d: %s", method, path, status, snippet(body))
}

// snippet returns a whitespace-trimmed body excerpt for error messages.
func snippet(body []byte) string {
	s := strings.TrimSpace(string(body))
	if len(s) > errorSnippetLen {
		s = s[:errorSnippetLen]
	}
	return s
}

// parsePointID renders a Qdrant point id, which is either a UUID string or
// an unsigned integer, as a Go string.
func parsePointID(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var n json.Number
	if err := json.Unmarshal(raw, &n); err == nil {
		return n.String()
	}
	return string(raw)
}

func (c *Client) PointsCount(ctx context.Context) (int, error) {
	path := "/collections/" + c.collection
	status, body, err := c.do(ctx, http.MethodGet, path, nil)
	if err != nil {
		return 0, err
	}
	if err := c.statusError(http.MethodGet, path, status, body); err != nil {
		return 0, err
	}
	var parsed struct {
		Result struct {
			PointsCount int `json:"points_count"`
		} `json:"result"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return 0, fmt.Errorf("qdrant GET %s: decoding response: %w", path, err)
	}
	return parsed.Result.PointsCount, nil
}

// DeleteBySource removes every point whose payload "source" matches the
// given value (used for document sync: delete-then-insert keeps uploads
// idempotent). A no-op when nothing matches.
func (c *Client) DeleteBySource(ctx context.Context, source string) error {
	path := "/collections/" + c.collection + "/points/delete?wait=true"
	payload := map[string]any{
		"filter": map[string]any{
			"must": []map[string]any{
				{"key": "source", "match": map[string]any{"value": source}},
			},
		},
	}
	status, body, err := c.do(ctx, http.MethodPost, path, payload)
	if err != nil {
		return err
	}
	if status == http.StatusNotFound {
		return nil // collection gone: nothing to delete
	}
	return c.statusError(http.MethodPost, path, status, body)
}

// SourceStat aggregates the chunk count of one indexed source file.
type SourceStat = engine.SourceStat

// ListSources returns every distinct payload source with its chunk count,
// ordered by chunks descending (scrolls sources only, no vectors).
func (c *Client) ListSources(ctx context.Context) ([]SourceStat, error) {
	path := "/collections/" + c.collection + "/points/scroll"
	counts := make(map[string]int)
	var cursor *string
	for {
		payload := map[string]any{
			"limit":        256,
			"with_payload": []string{"source"},
		}
		if cursor != nil {
			payload["offset"] = *cursor
		}
		status, body, err := c.do(ctx, http.MethodPost, path, payload)
		if err != nil {
			return nil, err
		}
		if status < 200 || status >= 300 {
			return nil, c.statusError(http.MethodPost, path, status, body)
		}
		var parsed struct {
			Result struct {
				Points []struct {
					Payload struct {
						Source string `json:"source"`
					} `json:"payload"`
				} `json:"points"`
				NextPageOffset *string `json:"next_page_offset"`
			} `json:"result"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			return nil, fmt.Errorf("decoding scroll response: %w", err)
		}
		for _, point := range parsed.Result.Points {
			counts[point.Payload.Source]++
		}
		if parsed.Result.NextPageOffset == nil {
			break
		}
		cursor = parsed.Result.NextPageOffset
	}
	stats := make([]SourceStat, 0, len(counts))
	for source, chunks := range counts {
		stats = append(stats, SourceStat{Source: source, Chunks: chunks})
	}
	sort.Slice(stats, func(i, j int) bool {
		if stats[i].Chunks != stats[j].Chunks {
			return stats[i].Chunks > stats[j].Chunks
		}
		return stats[i].Source < stats[j].Source
	})
	return stats, nil
}
