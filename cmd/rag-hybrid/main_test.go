package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"rag-hybrid/internal/config"
)

// fakeQdrant answers the collection GET after the given delay, letting a
// test hold an /api/health request in flight across a shutdown.
func fakeQdrant(t *testing.T, delay time.Duration) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(delay)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"result": map[string]any{"points_count": 1},
			"status": "ok",
		})
	}))
	t.Cleanup(server.Close)
	return server
}

// testConfig is the minimal config server.New needs: an isolated SQLite
// dir plus the fake qdrant URL. No real service is contacted.
func testConfig(t *testing.T, qdrantURL string) *config.Config {
	t.Helper()
	return &config.Config{
		DataDir:          t.TempDir(),
		RetentionDays:    7,
		QdrantURL:        qdrantURL,
		QdrantCollection: "test-collection",
	}
}

// startServer runs runServer on an ephemeral listener and returns its
// base URL, the shutdown trigger and the channel delivering its return
// value.
func startServer(t *testing.T, cfg *config.Config) (string, context.CancelFunc, chan error) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- runServer(ctx, cfg, ln) }()
	return fmt.Sprintf("http://%s", ln.Addr()), cancel, done
}

// requestResult carries one GET outcome across a channel.
type requestResult struct {
	resp *http.Response
	err  error
}

// getHealth fires GET /api/health and reports its outcome on the returned
// channel.
func getHealth(baseURL string) chan requestResult {
	ch := make(chan requestResult, 1)
	go func() {
		resp, err := http.Get(baseURL + "/api/health")
		ch <- requestResult{resp, err}
	}()
	return ch
}

// An in-flight request when shutdown starts must complete with its full
// response: the server drains instead of cutting it.
func TestRunServerDrainsInFlightRequests(t *testing.T) {
	cfg := testConfig(t, fakeQdrant(t, 800*time.Millisecond).URL)
	baseURL, cancel, done := startServer(t, cfg)

	ch := getHealth(baseURL)
	time.Sleep(200 * time.Millisecond) // the request is now inside the fake qdrant delay
	select {
	case <-ch:
		t.Fatal("request finished too early: it should be stuck in the fake qdrant delay")
	default:
	}
	cancel()

	select {
	case res := <-ch:
		if res.err != nil {
			t.Fatalf("in-flight request was cut by the shutdown: %v", res.err)
		}
		defer res.resp.Body.Close()
		if res.resp.StatusCode != http.StatusOK {
			t.Fatalf("in-flight request status = %d, want 200", res.resp.StatusCode)
		}
		body, _ := io.ReadAll(res.resp.Body)
		var health struct {
			Qdrant bool `json:"qdrant"`
			Points int  `json:"points"`
		}
		if err := json.Unmarshal(body, &health); err != nil {
			t.Fatalf("decoding health body %q: %v", body, err)
		}
		if !health.Qdrant || health.Points != 1 {
			t.Fatalf("health body = %q, want drained response with qdrant=true, points=1", body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("in-flight request never completed: shutdown did not drain")
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("runServer returned error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runServer did not return after shutdown")
	}
}

// When the drain deadline is smaller than the in-flight request, shutdown
// force-closes the connection: the request fails, but the server still
// returns nil (it shut down, deterministically).
func TestRunServerForcesCloseWhenDrainExceedsDeadline(t *testing.T) {
	restore := drainTimeout
	drainTimeout = 150 * time.Millisecond
	t.Cleanup(func() { drainTimeout = restore })

	cfg := testConfig(t, fakeQdrant(t, 2*time.Second).URL) // longer than the drain window
	baseURL, cancel, done := startServer(t, cfg)

	ch := getHealth(baseURL)
	time.Sleep(200 * time.Millisecond)
	cancel()

	select {
	case res := <-ch:
		if res.err == nil {
			res.resp.Body.Close()
			t.Fatal("forced close must cut the over-deadline request, got a response")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the over-deadline request hung: force close did not happen")
	}

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("runServer returned error after forced close: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("runServer did not return after forced close")
	}
}
