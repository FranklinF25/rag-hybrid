package qdrant

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

func TestEnsureCollectionCreatesOn404(t *testing.T) {
	rec := &recorded{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.req = r
		rec.body, _ = io.ReadAll(r.Body)
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/collections/rag_test":
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, `{"status":{"error":"Not found: Collection `+"`rag_test`"+` doesn't exist!"},"time":0.0}`)
		case r.Method == http.MethodPut && r.URL.Path == "/collections/rag_test":
			fmt.Fprint(w, `{"result":true,"status":"ok","time":0.1}`)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "rag_test")

	if err := c.EnsureCollection(context.Background(), 4); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	if rec.req.Method != http.MethodPut {
		t.Fatalf("last method = %s, want PUT after 404", rec.req.Method)
	}
	var sent map[string]any
	if err := json.Unmarshal(rec.body, &sent); err != nil {
		t.Fatalf("create body is not JSON: %v", err)
	}
	want := map[string]any{
		"vectors": map[string]any{
			"text-dense": map[string]any{"size": float64(4), "distance": "Cosine"},
		},
		"sparse_vectors": map[string]any{
			"text-sparse": map[string]any{"modifier": "idf"},
		},
	}
	if !mapsEqual(sent, want) {
		t.Errorf("create body = %v, want %v", sent, want)
	}
}

func TestEnsureCollectionExistingIsUntouched(t *testing.T) {
	putSeen := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut {
			putSeen = true
		}
		fmt.Fprint(w, `{"result":{"status":"green"},"status":"ok","time":0.0}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "rag_test")

	if err := c.EnsureCollection(context.Background(), 4096); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	if putSeen {
		t.Error("EnsureCollection issued a PUT for an existing collection")
	}
}

func TestEnsureCollectionServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "rag_test")

	err := c.EnsureCollection(context.Background(), 4)
	if err == nil || !strings.Contains(err.Error(), "status 500") || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("error = %v, want status 500 with body snippet", err)
	}
}

func TestUpsertRequestShape(t *testing.T) {
	rec := &recorded{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.req = r
		rec.body, _ = io.ReadAll(r.Body)
		fmt.Fprint(w, `{"result":{"operation_id":0,"status":"completed"},"status":"ok"}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "secret", "rag_test")

	points := []Point{{
		ID:            "11111111-1111-1111-1111-111111111111",
		Dense:         []float64{0.1, 0.2, 0.3, 0.4},
		SparseIndices: []uint32{1, 5},
		SparseValues:  []float64{0.5, 0.7},
		Content:       "alpha doc",
		Source:        "a.txt",
	}}
	if err := c.Upsert(context.Background(), points); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	if rec.req.Method != http.MethodPut {
		t.Errorf("method = %s, want PUT", rec.req.Method)
	}
	if rec.req.URL.Path != "/collections/rag_test/points" {
		t.Errorf("path = %q, want /collections/rag_test/points", rec.req.URL.Path)
	}
	if wait := rec.req.URL.Query().Get("wait"); wait != "true" {
		t.Errorf("wait = %q, want true", wait)
	}
	if key := rec.req.Header.Get("api-key"); key != "secret" {
		t.Errorf("api-key header = %q, want secret", key)
	}

	var sent map[string]any
	if err := json.Unmarshal(rec.body, &sent); err != nil {
		t.Fatalf("upsert body is not JSON: %v", err)
	}
	pts, _ := sent["points"].([]any)
	if len(pts) != 1 {
		t.Fatalf("points = %v, want 1 entry", sent["points"])
	}
	pt := pts[0].(map[string]any)
	if pt["id"] != "11111111-1111-1111-1111-111111111111" {
		t.Errorf("id = %v", pt["id"])
	}
	vec := pt["vector"].(map[string]any)
	dense := vec["text-dense"].([]any)
	if len(dense) != 4 || dense[0] != 0.1 || dense[3] != 0.4 {
		t.Errorf("text-dense = %v", vec["text-dense"])
	}
	sparse := vec["text-sparse"].(map[string]any)
	if idx := sparse["indices"].([]any); len(idx) != 2 || idx[0] != float64(1) || idx[1] != float64(5) {
		t.Errorf("indices = %v", sparse["indices"])
	}
	if vals := sparse["values"].([]any); len(vals) != 2 || vals[1] != 0.7 {
		t.Errorf("values = %v", sparse["values"])
	}
	payload := pt["payload"].(map[string]any)
	if payload["content"] != "alpha doc" || payload["source"] != "a.txt" {
		t.Errorf("payload = %v", payload)
	}
}

func TestUpsertError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad point", http.StatusBadRequest)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "rag_test")

	err := c.Upsert(context.Background(), []Point{{ID: "x"}})
	if err == nil || !strings.Contains(err.Error(), "status 400") || !strings.Contains(err.Error(), "bad point") {
		t.Fatalf("error = %v, want status 400 with body snippet", err)
	}
}

func TestHybridQueryRequestShapeAndParsing(t *testing.T) {
	rec := &recorded{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.req = r
		rec.body, _ = io.ReadAll(r.Body)
		fmt.Fprint(w, `{"result":{"points":[
			{"id":"11111111-1111-1111-1111-111111111111","score":1.0,"payload":{"content":"alpha doc","source":"a.txt"}},
			{"id":42,"score":0.5,"payload":{"content":"beta doc","source":"b.txt"}}
		]},"status":"ok"}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "rag_test")

	hits, err := c.HybridQuery(context.Background(),
		[]float64{0.1, 0.2, 0.3, 0.4}, []uint32{1, 5}, []float64{0.5, 0.7}, 10)
	if err != nil {
		t.Fatalf("HybridQuery: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2", len(hits))
	}
	if hits[0] != (Hit{ID: "11111111-1111-1111-1111-111111111111", Score: 1.0, Content: "alpha doc", Source: "a.txt"}) {
		t.Errorf("hits[0] = %+v", hits[0])
	}
	if hits[1] != (Hit{ID: "42", Score: 0.5, Content: "beta doc", Source: "b.txt"}) {
		t.Errorf("hits[1] = %+v (numeric id not rendered as string)", hits[1])
	}

	if rec.req.Method != http.MethodPost {
		t.Errorf("method = %s, want POST", rec.req.Method)
	}
	if rec.req.URL.Path != "/collections/rag_test/points/query" {
		t.Errorf("path = %q, want /collections/rag_test/points/query", rec.req.URL.Path)
	}
	var sent map[string]any
	if err := json.Unmarshal(rec.body, &sent); err != nil {
		t.Fatalf("query body is not JSON: %v", err)
	}
	if sent["with_payload"] != true {
		t.Errorf("with_payload = %v, want true", sent["with_payload"])
	}
	if sent["limit"] != float64(10) {
		t.Errorf("limit = %v, want 10", sent["limit"])
	}
	if q := sent["query"].(map[string]any); q["fusion"] != "rrf" {
		t.Errorf("query = %v, want {fusion: rrf}", sent["query"])
	}
	prefetch, _ := sent["prefetch"].([]any)
	if len(prefetch) != 2 {
		t.Fatalf("prefetch = %v, want 2 legs", sent["prefetch"])
	}
	// Leg 0: sparse. Verified against live Qdrant 1.14.1: the query value is
	// the bare {indices, values} object, NOT wrapped in "sparse_vector".
	sp := prefetch[0].(map[string]any)
	if sp["using"] != "text-sparse" || sp["limit"] != float64(10) {
		t.Errorf("sparse prefetch meta = %v", sp)
	}
	spq, ok := sp["query"].(map[string]any)
	if !ok {
		t.Fatalf("sparse prefetch query = %v, want an object", sp["query"])
	}
	if idx := spq["indices"].([]any); len(idx) != 2 || idx[0] != float64(1) || idx[1] != float64(5) {
		t.Errorf("sparse indices = %v", spq["indices"])
	}
	if vals := spq["values"].([]any); len(vals) != 2 || vals[1] != 0.7 {
		t.Errorf("sparse values = %v", spq["values"])
	}
	// Leg 1: dense, a plain float array.
	dp := prefetch[1].(map[string]any)
	if dp["using"] != "text-dense" || dp["limit"] != float64(10) {
		t.Errorf("dense prefetch meta = %v", dp)
	}
	dq, ok := dp["query"].([]any)
	if !ok || len(dq) != 4 || dq[0] != 0.1 || dq[3] != 0.4 {
		t.Errorf("dense query = %v, want [0.1 0.2 0.3 0.4]", dp["query"])
	}
}

func TestHybridQueryError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "collection missing", http.StatusNotFound)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "rag_test")

	_, err := c.HybridQuery(context.Background(), []float64{1}, []uint32{1}, []float64{1}, 5)
	if err == nil || !strings.Contains(err.Error(), "status 404") {
		t.Fatalf("error = %v, want status 404", err)
	}
}

func TestDropCollection(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		wantErr bool
	}{
		{"deleted", http.StatusOK, false},
		{"already gone", http.StatusNotFound, false},
		{"server error", http.StatusInternalServerError, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := &recorded{}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				rec.req = r
				w.WriteHeader(tc.status)
				fmt.Fprint(w, `{"result":true,"status":"ok"}`)
			}))
			defer srv.Close()
			c := NewClient(srv.URL, "", "rag_test")

			err := c.DropCollection(context.Background())
			if tc.wantErr && (err == nil || !strings.Contains(err.Error(), fmt.Sprintf("status %d", tc.status))) {
				t.Fatalf("error = %v, want status %d", err, tc.status)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("error = %v, want nil for status %d", err, tc.status)
			}
			if rec.req.Method != http.MethodDelete || rec.req.URL.Path != "/collections/rag_test" {
				t.Errorf("request = %s %s, want DELETE /collections/rag_test", rec.req.Method, rec.req.URL.Path)
			}
		})
	}
}

func TestNoAuthHeaderWithoutKey(t *testing.T) {
	rec := &recorded{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.req = r
		fmt.Fprint(w, `{"result":{"status":"green"},"status":"ok"}`)
	}))
	defer srv.Close()
	c := NewClient(srv.URL, "", "rag_test")

	if err := c.EnsureCollection(context.Background(), 4); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	if key := rec.req.Header.Get("api-key"); key != "" {
		t.Errorf("api-key header = %q, want absent for empty key", key)
	}
}

func mapsEqual(a, b map[string]any) bool {
	ab, _ := json.Marshal(a)
	bb, _ := json.Marshal(b)
	return string(ab) == string(bb)
}
