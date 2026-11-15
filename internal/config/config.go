// Package config loads rag-hybrid settings from the environment and an
// optional .env file (same variable names as the Python version's
// env.example, so an existing .env keeps working unchanged).
package config

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config carries every runtime knob. All fields have the same defaults as
// the Python Settings class unless an env var overrides them.
type Config struct {
	// OpenRouter (embeddings + reranking)
	OpenRouterAPIKey  string
	OpenRouterBaseURL string
	EmbedModel        string
	EmbedDim          int
	EmbedBatchSize    int
	RerankerModel     string

	// DeepSeek (generation)
	DeepSeekAPIKey      string
	LLMBaseURL          string
	LLMModel            string
	GenerationMaxTokens int

	// Qdrant
	QdrantURL        string
	QdrantAPIKey     string
	QdrantCollection string

	// Chunking / retrieval
	ChunkSize            int
	ChunkOverlap         int
	RetrievalTopK        int
	RerankTopK           int
	RerankScoreThreshold float64 // best rerank score below this -> deterministic "not in corpus" (0 disables)

	// Conversations
	DataDir       string // SQLite store directory (conversations)
	HistoryTurns  int    // window of recent turns used for rewriting/context
	RetentionDays int    // purge conversations older than N days (0 = never)

	// Web UI
	WebHost string
	WebPort int
}

// Load builds a Config from .env (optional) plus the process environment.
// Environment variables win over .env values.
func Load() (*Config, error) {
	if err := loadDotEnv(".env"); err != nil {
		return nil, fmt.Errorf("loading .env: %w", err)
	}
	c := &Config{
		OpenRouterAPIKey:     env("OPENROUTER_API_KEY", ""),
		OpenRouterBaseURL:    env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
		EmbedModel:           env("EMBED_MODEL", "qwen/qwen3-embedding-8b"),
		EmbedDim:             envInt("EMBED_DIM", 4096),
		EmbedBatchSize:       envInt("EMBED_BATCH_SIZE", 32),
		RerankerModel:        env("RERANKER_MODEL", "mongodb/rerank-3-lite"),
		DeepSeekAPIKey:       env("DEEPSEEK_API_KEY", ""),
		LLMBaseURL:           env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
		LLMModel:             env("LLM_MODEL", "deepseek-chat"),
		GenerationMaxTokens:  envInt("GENERATION_MAX_TOKENS", 700),
		QdrantURL:            env("QDRANT_URL", "http://localhost:6333"),
		QdrantAPIKey:         env("QDRANT_API_KEY", ""),
		QdrantCollection:     env("QDRANT_COLLECTION", "rag_hybrid"),
		ChunkSize:            envInt("CHUNK_SIZE", 400),
		ChunkOverlap:         envInt("CHUNK_OVERLAP", 50),
		RetrievalTopK:        envInt("RETRIEVAL_TOP_K", 20),
		RerankTopK:           envInt("RERANK_TOP_K", 5),
		RerankScoreThreshold: envFloat("RERANK_SCORE_THRESHOLD", 0.4),
		DataDir:              env("DATA_DIR", ".data"),
		HistoryTurns:         envInt("HISTORY_TURNS", 4),
		RetentionDays:        envInt("RETENTION_DAYS", 90),
		WebHost:              env("WEB_HOST", "127.0.0.1"),
		WebPort:              envInt("WEB_PORT", 8000),
	}
	if c.ChunkOverlap >= c.ChunkSize {
		return nil, fmt.Errorf("CHUNK_OVERLAP (%d) must be smaller than CHUNK_SIZE (%d)", c.ChunkOverlap, c.ChunkSize)
	}
	return c, nil
}

// loadDotEnv applies KEY=VALUE lines from path when present. Existing
// environment variables are never overwritten. Lines starting with # and
// blank lines are ignored.
func loadDotEnv(path string) error {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		if key != "" && os.Getenv(key) == "" {
			if err := os.Setenv(key, value); err != nil {
				return err
			}
		}
	}
	return scanner.Err()
}

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return fallback
}

func envFloat(key string, fallback float64) float64 {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil {
			return parsed
		}
	}
	return fallback
}
