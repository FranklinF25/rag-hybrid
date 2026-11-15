package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// newStreamTestServer spins up a fake DeepSeek API the client can talk to.
func newStreamTestServer(t *testing.T, handler http.HandlerFunc) *DeepSeek {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return NewDeepSeek("test-key", server.URL)
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
