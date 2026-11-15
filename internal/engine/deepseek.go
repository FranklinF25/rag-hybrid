package engine

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

// DeepSeek generates the final answer. It speaks the OpenAI-compatible
// chat-completions dialect against the official DeepSeek API.
type DeepSeek struct {
	APIKey  string
	BaseURL string
	Client  *http.Client
}

// NewDeepSeek builds a client with sane defaults (base URL and 120s
// timeout; generation can take a while).
func NewDeepSeek(apiKey, baseURL string) *DeepSeek {
	if baseURL == "" {
		baseURL = "https://api.deepseek.com/v1"
	}
	return &DeepSeek{APIKey: apiKey, BaseURL: baseURL, Client: &http.Client{Timeout: 120 * time.Second}}
}

type chatRequest struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens"`
	Stream    bool          `json:"stream,omitempty"` // set only by CompleteStream
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// Complete sends one user prompt and returns the generated text.
func (d *DeepSeek) Complete(ctx context.Context, model, prompt string, maxTokens int) (string, error) {
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
}

// CompleteStream sends one user prompt and streams the generated text as
// SSE. Every non-empty delta is passed to onDelta as soon as it arrives;
// the concatenation of all deltas is also returned when the stream ends
// ("data: [DONE]"). If onDelta returns an error the stream is aborted and
// that error is returned along with the text accumulated so far.
func (d *DeepSeek) CompleteStream(ctx context.Context, model, prompt string, maxTokens int, onDelta func(string) error) (string, error) {
	if d.APIKey == "" {
		return "", fmt.Errorf("DEEPSEEK_API_KEY is not set: copy env.example to .env and add your key")
	}
	body, err := json.Marshal(chatRequest{
		Model:     model,
		Messages:  []chatMessage{{Role: "user", Content: prompt}},
		MaxTokens: maxTokens,
		Stream:    true,
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
	return full.String(), nil
}
