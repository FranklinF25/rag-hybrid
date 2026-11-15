// Conversation endpoints: durable chat with context-aware retrieval,
// SSE streaming, listing, deletion and retention.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"rag-hybrid/internal/engine"
	"rag-hybrid/internal/store"
)

// openStore opens (creating if needed) the SQLite conversation store and
// starts the retention loop.
func openStore(dataDir string, retentionDays int) (*store.Store, chan struct{}, error) {
	convStore, err := store.Open(filepath.Join(dataDir, "rag.db"))
	if err != nil {
		return nil, nil, err
	}
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(time.Hour)
		defer ticker.Stop()
		purge := func() {
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			_, _ = convStore.PurgeOlderThan(ctx, retentionDays)
		}
		purge() // once at startup, then hourly
		for {
			select {
			case <-ticker.C:
				purge()
			case <-done:
				return
			}
		}
	}()
	return convStore, done, nil
}

func (s *Server) handleListConversations(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	conversations, err := s.convs.ListConversations(ctx, 50)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	type view struct {
		ID        string `json:"id"`
		Title     string `json:"title"`
		UpdatedAt int64  `json:"updated_at"`
		Turns     int    `json:"turns"`
	}
	list := make([]view, 0, len(conversations))
	for _, c := range conversations {
		list = append(list, view{ID: c.ID, Title: c.Title, UpdatedAt: c.UpdatedAt, Turns: c.TurnCount})
	}
	writeJSON(w, http.StatusOK, map[string]any{"conversations": list})
}

func (s *Server) handleDeleteConversation(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(r.URL.Query().Get("id"))
	if id == "" {
		writeErr(w, http.StatusBadRequest, "missing query parameter: id")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	if err := s.convs.DeleteConversation(ctx, id); err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": id})
}

func (s *Server) handleConversationTurns(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	turns, err := s.convs.FullHistory(ctx, id)
	if err != nil {
		writeErr(w, http.StatusServiceUnavailable, err.Error())
		return
	}
	type turnView struct {
		Role    string       `json:"role"`
		Content string       `json:"content"`
		Sources []sourceView `json:"sources"`
	}
	views := make([]turnView, 0, len(turns))
	for _, t := range turns {
		var sources []sourceView
		_ = json.Unmarshal([]byte(t.Sources), &sources)
		views = append(views, turnView{Role: t.Role, Content: t.Content, Sources: sources})
	}
	writeJSON(w, http.StatusOK, map[string]any{"turns": views})
}

// chatFlow is the shared orchestration of /api/chat and /api/chat/stream:
// resolves (or creates) the conversation, loads the history window, and
// returns the pieces the callers need.
func (s *Server) chatFlow(r *http.Request, message, conversationID string) (conv store.Conversation, history []engine.Turn, created bool, err error) {
	ctx := r.Context()
	if conversationID != "" {
		conv = store.Conversation{ID: conversationID}
	} else {
		conv, err = s.convs.CreateConversation(ctx)
		if err != nil {
			return conv, nil, false, fmt.Errorf("creating conversation: %w", err)
		}
		created = true
	}
	rawTurns, err := s.convs.RecentTurns(ctx, conv.ID, s.cfg.HistoryTurns)
	if err != nil {
		return conv, nil, created, fmt.Errorf("loading history: %w", err)
	}
	history = make([]engine.Turn, 0, len(rawTurns))
	for _, t := range rawTurns {
		history = append(history, engine.Turn{Role: t.Role, Content: t.Content})
	}
	return conv, history, created, nil
}

// persistTurns saves the user turn (raw question) and the assistant turn
// (answer + sources), naming the conversation on its first question.
func (s *Server) persistTurns(r *http.Request, convID, question string, created bool, answer string, sources []sourceView) {
	ctx := r.Context()
	if created {
		title := question
		if len(title) > 60 {
			title = title[:60]
		}
		_ = s.convs.SetTitle(ctx, convID, title)
	}
	_ = s.convs.AppendTurn(ctx, convID, "user", question, "[]")
	if sourcesJSON, err := json.Marshal(sources); err == nil {
		_ = s.convs.AppendTurn(ctx, convID, "assistant", answer, string(sourcesJSON))
	} else {
		_ = s.convs.AppendTurn(ctx, convID, "assistant", answer, "[]")
	}
}

// handleChatStream: SSE. Events: conversation, rewritten, sources, delta*,
// done | error. Falls back to a single delta for threshold rejections.
func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message        string `json:"message"`
		ConversationID string `json:"conversation_id"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	question := strings.TrimSpace(body.Message)
	if question == "" {
		writeErr(w, http.StatusBadRequest, "message must not be empty")
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErr(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	send := func(event string, payload any) {
		data, _ := json.Marshal(payload)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
		flusher.Flush()
	}
	sendError := func(err error) {
		send("error", map[string]string{"detail": err.Error()})
	}

	conv, history, created, err := s.chatFlow(r, question, body.ConversationID)
	if err != nil {
		sendError(err)
		return
	}
	send("conversation", map[string]any{"id": conv.ID})
	_ = s.convs.AppendTurn(r.Context(), conv.ID, "user", question, "[]")
	if created {
		title := question
		if len(title) > 60 {
			title = title[:60]
		}
		_ = s.convs.SetTitle(r.Context(), conv.ID, title)
	}

	ctx, cancel := context.WithTimeout(r.Context(), 4*time.Minute)
	defer cancel()

	// Staged flow: rewrite + retrieve first, so the UI gets the rewritten
	// query and the sources before the token deltas arrive.
	rewritten, hits, localAnswer, err := s.searcher.PrepareQuery(ctx, question, history)
	if err != nil {
		sendError(err)
		return
	}
	send("rewritten", map[string]string{"query": rewritten})
	views := make([]sourceView, 0, len(hits))
	for _, hit := range hits {
		views = append(views, toSourceView(hit.Source, hit.Score, hit.Content, false))
	}
	send("sources", map[string]any{"sources": views})

	var answerText string
	if localAnswer != "" {
		// Deterministic not-in-corpus answer: a single delta, no LLM call.
		answerText = localAnswer
		send("delta", map[string]string{"text": answerText})
	} else {
		answerText, err = s.searcher.GenerateStream(ctx, question, history, hits, func(delta string) error {
			send("delta", map[string]string{"text": delta})
			return nil
		})
		if err != nil {
			sendError(err)
			return
		}
	}
	if sourcesJSON, err := json.Marshal(views); err == nil {
		_ = s.convs.AppendTurn(r.Context(), conv.ID, "assistant", answerText, string(sourcesJSON))
	} else {
		_ = s.convs.AppendTurn(r.Context(), conv.ID, "assistant", answerText, "[]")
	}
	send("done", map[string]string{"conversation_id": conv.ID})
}
