// Package server serves the rag-hybrid web UI and JSON API with the
// standard library only: the chat page is embedded in the binary.
package server

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"rag-hybrid/internal/config"
	"rag-hybrid/internal/deepseek"
	"rag-hybrid/internal/engine"
	"rag-hybrid/internal/openrouter"
	"rag-hybrid/internal/qdrant"
	"rag-hybrid/internal/store"
)

//go:embed static/index.html
var indexHTML []byte

// Server holds the shared engine pieces behind the HTTP API.
type Server struct {
	cfg      *config.Config
	searcher *engine.Searcher
	indexer  *engine.Indexer
	store    engine.VectorStore
	convs    *store.Store
	done     chan struct{}
}

// New builds the server: the driving adapter AND its composition root. It
// constructs the driven adapters (qdrant vector store, OpenRouter embedder
// + reranker, DeepSeek LLM), injects them into the engine use cases, opens
// the SQLite conversation store and starts the retention loop; call Close
// to release both.
func New(cfg *config.Config) (*Server, error) {
	convStore, done, err := openStore(cfg.DataDir, cfg.RetentionDays)
	if err != nil {
		return nil, fmt.Errorf("opening conversation store: %w", err)
	}
	vectorStore := qdrant.NewClient(cfg.QdrantURL, cfg.QdrantAPIKey, cfg.QdrantCollection)
	open := openrouter.NewClient(cfg.OpenRouterAPIKey, cfg.OpenRouterBaseURL)
	llm := deepseek.New(cfg.DeepSeekAPIKey, cfg.LLMBaseURL)
	return &Server{
		cfg:      cfg,
		searcher: engine.NewSearcher(cfg, vectorStore, open, open, llm),
		indexer:  engine.NewIndexer(cfg, vectorStore, open),
		store:    vectorStore,
		convs:    convStore,
		done:     done,
	}, nil
}

// Close stops the retention loop and the store.
func (s *Server) Close() {
	close(s.done)
	_ = s.convs.Close()
}

// Handler returns the routed http.Handler with request logging.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", s.handleIndex)
	mux.HandleFunc("GET /api/health", s.handleHealth)
	mux.HandleFunc("GET /api/search", s.handleSearch)
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("POST /api/chat/stream", s.handleChatStream)
	mux.HandleFunc("POST /api/feedback", s.handleFeedback)
	mux.HandleFunc("GET /api/conversations", s.handleListConversations)
	mux.HandleFunc("DELETE /api/conversations", s.handleDeleteConversation)
	mux.HandleFunc("GET /api/conversations/{id}/turns", s.handleConversationTurns)
	mux.HandleFunc("POST /api/documents", s.handleDocuments)
	mux.HandleFunc("GET /api/documents", s.handleListDocuments)
	mux.HandleFunc("DELETE /api/documents", s.handleForget)
	return logRequests(mux)
}

// statusRecorder captures the response code for logging.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Flush propagates the Flusher interface through the middleware so SSE
// streaming works behind request logging.
func (r *statusRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// logRequests prints one line per request: method, path, status, duration.
func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		recorder := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(recorder, r)
		fmt.Printf("%s %s -> %d (%s)\n", r.Method, r.URL.RequestURI(), recorder.status, time.Since(start).Round(time.Millisecond))
	})
}

func (s *Server) handleIndex(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(indexHTML)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	health := map[string]any{
		"qdrant":     false,
		"points":     nil,
		"key":        s.cfg.DeepSeekAPIKey != "",
		"openrouter": s.cfg.OpenRouterAPIKey != "",
	}
	if points, err := s.store.PointsCount(ctx); err == nil {
		health["qdrant"] = true
		health["points"] = points
	}
	writeJSON(w, http.StatusOK, health)
}

type sourceView struct {
	Source  string  `json:"source"`
	Score   float64 `json:"score"`
	Preview string  `json:"preview"`
	Content string  `json:"content,omitempty"` // full text, only with ?full=1
}

func toSourceView(source string, score float64, content string, full bool) sourceView {
	view := sourceView{Source: source, Score: score, Preview: preview(content, 140)}
	if full {
		view.Content = content
	}
	return view
}

func preview(content string, max int) string {
	flat := strings.Join(strings.Fields(content), " ")
	if len(flat) > max {
		flat = flat[:max]
	}
	return flat
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	question := strings.TrimSpace(r.URL.Query().Get("q"))
	if question == "" {
		writeErr(w, http.StatusBadRequest, "missing query parameter: q")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()
	hits, err := s.searcher.Retrieve(ctx, question)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	results := make([]sourceView, 0, len(hits))
	full := r.URL.Query().Get("full") == "1"
	for _, hit := range hits {
		results = append(results, toSourceView(hit.Source, hit.Score, hit.Content, full))
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message        string `json:"message"`
		ConversationID string `json:"conversation_id"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	question := strings.TrimSpace(body.Message)
	if question == "" {
		writeErr(w, http.StatusBadRequest, "message must not be empty")
		return
	}

	conv, history, created, err := s.chatFlow(r, question, body.ConversationID)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	_ = s.convs.AppendTurn(r.Context(), conv.ID, "user", question, "[]")
	if created {
		title := question
		if len(title) > 60 {
			title = title[:60]
		}
		_ = s.convs.SetTitle(r.Context(), conv.ID, title)
	}

	ctx, cancel := context.WithTimeout(r.Context(), 180*time.Second)
	defer cancel()
	answer, _, err := s.searcher.AskWithContext(ctx, question, history)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	sources := make([]sourceView, 0, len(answer.Sources))
	for _, hit := range answer.Sources {
		sources = append(sources, toSourceView(hit.Source, hit.Score, hit.Content, true))
	}
	if sourcesJSON, err := json.Marshal(sources); err == nil {
		_ = s.convs.AppendTurn(r.Context(), conv.ID, "assistant", answer.Text, string(sourcesJSON))
	} else {
		_ = s.convs.AppendTurn(r.Context(), conv.ID, "assistant", answer.Text, "[]")
	}
	writeJSON(w, http.StatusOK, map[string]any{"answer": answer.Text, "sources": sources, "conversation_id": conv.ID})
}

func (s *Server) handleDocuments(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(engine.MaxUploadBytes); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid multipart form: "+err.Error())
		return
	}
	headers := r.MultipartForm.File["files"]
	if len(headers) == 0 {
		writeErr(w, http.StatusBadRequest, "no files under field 'files'")
		return
	}
	files := make([]engine.UploadFile, 0, len(headers))
	for _, header := range headers {
		file, err := header.Open()
		if err != nil {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("opening %s: %v", header.Filename, err))
			return
		}
		data, err := io.ReadAll(io.LimitReader(file, engine.MaxUploadBytes+1))
		file.Close()
		if err != nil {
			writeErr(w, http.StatusBadRequest, fmt.Sprintf("reading %s: %v", header.Filename, err))
			return
		}
		files = append(files, engine.UploadFile{Name: header.Filename, Data: data})
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()
	result, err := s.indexer.IndexUploads(ctx, files)
	if err != nil && result.Files == 0 {
		detail := err.Error()
		if len(result.Skipped) > 0 {
			detail = fmt.Sprintf("%s — %s", strings.Join(result.Skipped, "; "), detail)
		}
		writeErr(w, http.StatusBadRequest, detail)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"files":   result.Files,
		"chunks":  result.Chunks,
		"skipped": result.Skipped,
	})
}

// handleListDocuments returns every indexed source with its chunk count.
func (s *Server) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	stats, err := s.store.ListSources(ctx)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	total := 0
	for _, stat := range stats {
		total += stat.Chunks
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": stats, "chunks": total})
}

// handleForget removes every chunk of one source file: DELETE /api/documents?source=name
func (s *Server) handleForget(w http.ResponseWriter, r *http.Request) {
	source := strings.TrimSpace(r.URL.Query().Get("source"))
	if source == "" {
		writeErr(w, http.StatusBadRequest, "missing query parameter: source")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	if err := s.store.DeleteBySource(ctx, source); err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"forgotten": source})
}

func writeJSON(w http.ResponseWriter, code int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeErr(w http.ResponseWriter, code int, message string) {
	writeJSON(w, code, map[string]string{"detail": message})
}
