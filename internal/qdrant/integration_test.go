package qdrant

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"
)

const liveQdrantURL = "http://localhost:6333"

// TestQdrantIntegration exercises EnsureCollection, Upsert and HybridQuery
// against a live local Qdrant (v1.14.1, no auth) using a throwaway
// collection. It skips when no server is running on localhost:6333.
func TestQdrantIntegration(t *testing.T) {
	probe := &http.Client{Timeout: 2 * time.Second}
	resp, err := probe.Get(liveQdrantURL)
	if err != nil {
		t.Skipf("qdrant not reachable at %s: %v", liveQdrantURL, err)
	}
	resp.Body.Close()

	collection := fmt.Sprintf("go_test_%d", os.Getpid())
	c := NewClient(liveQdrantURL, "", collection)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	defer func() {
		if err := c.DropCollection(ctx); err != nil {
			t.Errorf("cleanup DropCollection: %v", err)
		}
	}()

	if err := c.EnsureCollection(ctx, 4); err != nil {
		t.Fatalf("EnsureCollection: %v", err)
	}
	if err := c.EnsureCollection(ctx, 4); err != nil {
		t.Fatalf("EnsureCollection (idempotent second call): %v", err)
	}

	points := []Point{
		{
			ID:            "11111111-1111-1111-1111-111111111111",
			Dense:         []float64{0.1, 0.2, 0.3, 0.4},
			SparseIndices: []uint32{1, 5},
			SparseValues:  []float64{0.5, 0.7},
			Content:       "alpha doc",
			Source:        "a.txt",
		},
		{
			ID:            "22222222-2222-2222-2222-222222222222",
			Dense:         []float64{0.4, 0.3, 0.2, 0.1},
			SparseIndices: []uint32{2, 5},
			SparseValues:  []float64{0.6, 0.9},
			Content:       "beta doc",
			Source:        "b.txt",
		},
	}
	if err := c.Upsert(ctx, points); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	hits, err := c.HybridQuery(ctx,
		[]float64{0.1, 0.2, 0.3, 0.4}, []uint32{1, 5}, []float64{0.5, 0.7}, 10)
	if err != nil {
		t.Fatalf("HybridQuery: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2: %+v", len(hits), hits)
	}
	byID := map[string]Hit{}
	for _, h := range hits {
		byID[h.ID] = h
	}
	alpha, ok := byID["11111111-1111-1111-1111-111111111111"]
	if !ok {
		t.Fatalf("alpha point missing from hits: %+v", hits)
	}
	if alpha.Score <= 0 {
		t.Errorf("alpha score = %v, want > 0", alpha.Score)
	}
	if alpha.Content != "alpha doc" || alpha.Source != "a.txt" {
		t.Errorf("alpha payload = %+v, want content/source", alpha)
	}
	if beta, ok := byID["22222222-2222-2222-2222-222222222222"]; !ok || beta.Content != "beta doc" || beta.Source != "b.txt" {
		t.Errorf("beta hit = %+v, want content and source payload", byID["22222222-2222-2222-2222-222222222222"])
	}
}
