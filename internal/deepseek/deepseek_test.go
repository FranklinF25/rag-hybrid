package deepseek

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"rag-hybrid/internal/engine"
)

// Compile-time proof that the client satisfies the LLM port.
var _ engine.LLM = (*Client)(nil)

// newStreamTestServer spins up a fake DeepSeek API the client can talk to.
func newStreamTestServer(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return New("test-key", server.URL)
}

func TestCompleteStreamDeltas(t *testing.T) {
	llm := newStreamTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		var body chatRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
			return
		}
		if body.Model != "deepseek-chat" || body.MaxTokens != 64 || !body.Stream {
			t.Errorf("unexpected request fields: %+v", body)
		}
		if len(body.Messages) != 1 || body.Messages[0].Role != "user" || body.Messages[0].Content != "say hi" {
			t.Errorf("unexpected messages: %+v", body.Messages)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		frames := []string{
			`: keep-alive comment`,
			`data: {"choices":[{"delta":{"role":"assistant"}}]}`,
			`data: {"choices":[{"delta":{"content":"Hello"}}]}`,
			`data: {"choices":[{"delta":{"content":" wo"}}]}`,
			`data: {"choices":[{"delta":{"content":"rld"}}]}`,
			`data: [DONE]`,
		}
		for _, frame := range frames {
			fmt.Fprintf(w, "%s\n\n", frame)
		}
	})

	var got []string
	text, err := llm.CompleteStream(context.Background(), "deepseek-chat", "say hi", 64, func(delta string) error {
		got = append(got, delta)
		return nil
	})
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if want := []string{"Hello", " wo", "rld"}; !reflect.DeepEqual(got, want) {
		t.Errorf("onDelta deltas = %q, want %q", got, want)
	}
	if text != "Hello world" {
		t.Errorf("returned text = %q, want %q", text, "Hello world")
	}
}

func TestCompleteStreamAbortOnCallbackError(t *testing.T) {
	llm := newStreamTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"two\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"three\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	})

	boom := errors.New("client hung up")
	var got []string
	text, err := llm.CompleteStream(context.Background(), "deepseek-chat", "count", 8, func(delta string) error {
		got = append(got, delta)
		if delta == "two" {
			return boom
		}
		return nil
	})
	if !errors.Is(err, boom) {
		t.Fatalf("error = %v, want boom", err)
	}
	if want := []string{"one", "two"}; !reflect.DeepEqual(got, want) {
		t.Errorf("onDelta deltas = %q, want %q (stream must abort)", got, want)
	}
	if text != "one" {
		t.Errorf("accumulated text = %q, want %q", text, "one")
	}
}

func TestCompleteStreamErrorStatus(t *testing.T) {
	llm := newStreamTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"bad key"}`, http.StatusUnauthorized)
	})

	text, err := llm.CompleteStream(context.Background(), "deepseek-chat", "say hi", 64, func(string) error {
		return nil
	})
	if err == nil {
		t.Fatalf("expected an error for a non-2xx response, got text %q", text)
	}
	if !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "bad key") {
		t.Errorf("error should carry status and body, got: %v", err)
	}
	if text != "" {
		t.Errorf("text = %q, want empty", text)
	}
}

func TestCompleteLogsCacheUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"hi there"}}],"usage":{"prompt_cache_hit_tokens":120,"prompt_cache_miss_tokens":340,"total_tokens":460}}`)
	}))
	t.Cleanup(server.Close)
	llm := New("test-key", server.URL)

	var logs []string
	llm.Log = func(format string, args ...any) {
		logs = append(logs, fmt.Sprintf(format, args...))
	}

	text, err := llm.Complete(context.Background(), "deepseek-chat", "say hi", 64)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if text != "hi there" {
		t.Errorf("text = %q, want %q", text, "hi there")
	}
	if len(logs) != 1 {
		t.Fatalf("log lines = %d (%q), want exactly one line", len(logs), logs)
	}
	for _, want := range []string{"deepseek-chat", "cache_hit=120", "cache_miss=340"} {
		if !strings.Contains(logs[0], want) {
			t.Errorf("log line %q missing %q", logs[0], want)
		}
	}
}

func TestCompleteStreamRequestsUsageAndLogsIt(t *testing.T) {
	llm := newStreamTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
			return
		}
		if !strings.Contains(string(raw), `"stream_options":{"include_usage":true}`) {
			t.Errorf("request body %s missing stream_options include_usage", raw)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		frames := []string{
			`data: {"choices":[{"delta":{"content":"Hel"}}]}`,
			`data: {"choices":[{"delta":{"content":"lo"}}]}`,
			// DeepSeek sends usage only in the final chunk, with an empty choices array.
			`data: {"choices":[],"usage":{"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":500,"total_tokens":500}}`,
			`data: [DONE]`,
		}
		for _, frame := range frames {
			fmt.Fprintf(w, "%s\n\n", frame)
		}
	})

	var logs []string
	llm.Log = func(format string, args ...any) {
		logs = append(logs, fmt.Sprintf(format, args...))
	}

	text, err := llm.CompleteStream(context.Background(), "deepseek-chat", "say hi", 64, func(string) error {
		return nil
	})
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if text != "Hello" {
		t.Errorf("text = %q, want %q", text, "Hello")
	}
	if len(logs) != 1 {
		t.Fatalf("log lines = %d (%q), want exactly one line", len(logs), logs)
	}
	for _, want := range []string{"deepseek-chat", "cache_hit=0", "cache_miss=500"} {
		if !strings.Contains(logs[0], want) {
			t.Errorf("log line %q missing %q", logs[0], want)
		}
	}
}
