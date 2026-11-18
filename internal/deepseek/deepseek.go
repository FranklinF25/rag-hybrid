// Package deepseek provides a minimal, stdlib-only client for the DeepSeek
// chat-completions API (OpenAI-compatible dialect). It implements the
// engine.LLM port; the engine itself never imports this package.
package deepseek

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client generates the final answer. It speaks the OpenAI-compatible
// chat-completions dialect against the official DeepSeek API and
// satisfies the engine.LLM port.
type Client struct {
	APIKey  string
	BaseURL string
	Client  *http.Client

	// Log is the sink for the one usage line emitted after each completed
	// request ("deepseek usage: model=… cache_hit=… cache_miss=…"). When nil,
	// fmt.Printf is used, so a running server logs usage automatically. It
	// is read at call time, so swapping it is safe for in-flight calls.
	Log func(format string, args ...any)
}

// New builds a client with sane defaults (base URL and 120s
// timeout; generation can take a while).
func New(apiKey, baseURL string) *Client {
	if baseURL == "" {
		baseURL = "https://api.deepseek.com/v1"
	}
	return &Client{APIKey: apiKey, BaseURL: baseURL, Client: &http.Client{Timeout: 120 * time.Second}}
}

type chatRequest struct {
	Model         string         `json:"model"`
	Messages      []chatMessage  `json:"messages"`
	MaxTokens     int            `json:"max_tokens"`
	Stream        bool           `json:"stream,omitempty"`         // set only by CompleteStream
	StreamOptions *streamOptions `json:"stream_options,omitempty"` // set only by CompleteStream
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// streamOptions asks DeepSeek to include the usage object in the final
// SSE chunk of a streaming completion (it is omitted otherwise).
type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// usage mirrors the usage object DeepSeek returns with completions: how
// many prompt tokens were served from DeepSeek's prompt cache versus
// billed as a cache miss.
type usage struct {
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
	TotalTokens           int `json:"total_tokens"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage *usage `json:"usage"`
}

// logUsage emits the one-line usage report for a completed request so
// cache-hit rates can be measured across conversation turns. The sink is
// read at call time; nil falls back to fmt.Printf.
func (d *Client) logUsage(model string, u *usage) {
	if u == nil {
		return // server sent no usage object; nothing meaningful to report
	}
	logf := d.Log
	if logf == nil {
		logf = func(format string, args ...any) { fmt.Printf(format, args...) }
	}
	logf("deepseek usage: model=%s cache_hit=%d cache_miss=%d", model, u.PromptCacheHitTokens, u.PromptCacheMissTokens)
}

// Complete sends one user prompt and returns the generated text.
func (d *Client) Complete(ctx context.Context, model, prompt string, maxTokens int) (string, error) {
	if d.APIKey == "" {
		return "", fmt.Errorf("DEEPSEEK_API_KEY is not set: copy env.example to .env and add your key")
	}
	body, err := json.Marshal(chatRequest{
		Model:     model,
		Messages:  []chatMessage{{Role: "user", Content: prompt}},
		MaxTokens: maxTokens,
	})
	if err != nil {
		return "", err
	}
	url := strings.TrimRight(d.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+d.APIKey)

	resp, err := d.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("deepseek request: %w", err)
	}
	defer resp.Body.Close()

	payload, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("deepseek %s: %s", resp.Status, snippet(payload))
	}
	var parsed chatResponse
	if err := json.Unmarshal(payload, &parsed); err != nil {
		return "", fmt.Errorf("decoding deepseek response: %w", err)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("deepseek returned no choices")
	}
	d.logUsage(model, parsed.Usage)
	return parsed.Choices[0].Message.Content, nil
}

func snippet(data []byte) string {
	text := string(data)
	if len(text) > 200 {
		text = text[:200] + "…"
	}
	return text
}

// chatStreamResponse is one SSE chunk of a streaming completion: the new
// piece of text arrives in choices[0].delta.content.
type chatStreamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *usage `json:"usage"` // present only in the final chunk (empty choices), thanks to stream_options
}

// CompleteStream sends one user prompt and streams the generated text as
// SSE. Every non-empty delta is passed to onDelta as soon as it arrives;
// the concatenation of all deltas is also returned when the stream ends
// ("data: [DONE]"). If onDelta returns an error the stream is aborted and
// that error is returned along with the text accumulated so far.
func (d *Client) CompleteStream(ctx context.Context, model, prompt string, maxTokens int, onDelta func(string) error) (string, error) {
	if d.APIKey == "" {
		return "", fmt.Errorf("DEEPSEEK_API_KEY is not set: copy env.example to .env and add your key")
	}
	body, err := json.Marshal(chatRequest{
		Model:         model,
		Messages:      []chatMessage{{Role: "user", Content: prompt}},
		MaxTokens:     maxTokens,
		Stream:        true,
		StreamOptions: &streamOptions{IncludeUsage: true},
	})
	if err != nil {
		return "", err
	}
	url := strings.TrimRight(d.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Authorization", "Bearer "+d.APIKey)

	resp, err := d.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("deepseek request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return "", fmt.Errorf("deepseek %s: %s", resp.Status, snippet(payload))
	}

	var full strings.Builder
	var lastUsage *usage
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20) // SSE lines can exceed the 64KB default
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue // comments (": ping") and other event fields are ignored
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var chunk chatStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return full.String(), fmt.Errorf("decoding deepseek stream chunk: %w", err)
		}
		if chunk.Usage != nil {
			lastUsage = chunk.Usage // final chunk carries usage alongside an empty choices array
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta.Content
		if delta == "" {
			continue
		}
		if err := onDelta(delta); err != nil {
			return full.String(), err
		}
		full.WriteString(delta)
	}
	if err := scanner.Err(); err != nil {
		return full.String(), fmt.Errorf("reading deepseek stream: %w", err)
	}
	d.logUsage(model, lastUsage)
	return full.String(), nil
}
