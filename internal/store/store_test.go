package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()
	s, err := Open(filepath.Join(t.TempDir(), "chat.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return s
}

func TestOpenCreatesSchemaAndReopens(t *testing.T) {
	dir := t.TempDir()
	// The parent directory does not exist yet; Open must create it.
	path := filepath.Join(dir, "nested", "chat.db")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("first Open: %v", err)
	}

	for _, name := range []string{"conversations", "turns", "idx_turns_conversation"} {
		var found int
		if err := s.db.QueryRow(
			`SELECT COUNT(*) FROM sqlite_master WHERE name = ?`, name,
		).Scan(&found); err != nil {
			t.Fatalf("query sqlite_master for %q: %v", name, err)
		}
		if found != 1 {
			t.Errorf("schema object %q: found %d in sqlite_master, want 1", name, found)
		}
	}

	// Persist one turn, close, and reopen the same path: both the schema
	// (IF NOT EXISTS) and the data must survive.
	ctx := context.Background()
	c, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation: %v", err)
	}
	if err := s.AppendTurn(ctx, c.ID, "user", "hola", "[]"); err != nil {
		t.Fatalf("AppendTurn: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2, err := Open(path)
	if err != nil {
		t.Fatalf("second Open: %v", err)
	}
	defer s2.Close()
	turns, err := s2.FullHistory(ctx, c.ID)
	if err != nil {
		t.Fatalf("FullHistory after reopen: %v", err)
	}
	if len(turns) != 1 || turns[0].Content != "hola" {
		t.Errorf("after reopen got %+v, want 1 turn with content %q", turns, "hola")
	}
}

func TestAppendTurnAndFullHistory(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	c, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation: %v", err)
	}
	if len(c.ID) != 32 {
		t.Errorf("conversation id length = %d, want 32", len(c.ID))
	}
	if c.Title != "" {
		t.Errorf("new conversation title = %q, want empty", c.Title)
	}
	if c.CreatedAt <= 0 || c.UpdatedAt != c.CreatedAt {
		t.Errorf("new conversation timestamps = (%d, %d), want equal and positive", c.CreatedAt, c.UpdatedAt)
	}

	sources := `[{"title":"RAG guide","url":"https://example.com/rag"}]`
	if err := s.AppendTurn(ctx, c.ID, "user", "que es RAG?", "[]"); err != nil {
		t.Fatalf("AppendTurn(user): %v", err)
	}
	if err := s.AppendTurn(ctx, c.ID, "assistant", "RAG es...", sources); err != nil {
		t.Fatalf("AppendTurn(assistant): %v", err)
	}

	turns, err := s.FullHistory(ctx, c.ID)
	if err != nil {
		t.Fatalf("FullHistory: %v", err)
	}
	if len(turns) != 2 {
		t.Fatalf("FullHistory returned %d turns, want 2", len(turns))
	}
	wantRoles := []string{"user", "assistant"}
	wantContents := []string{"que es RAG?", "RAG es..."}
	wantSources := []string{"[]", sources}
	for i, turn := range turns {
		if turn.Role != wantRoles[i] {
			t.Errorf("turn %d role = %q, want %q", i, turn.Role, wantRoles[i])
		}
		if turn.Content != wantContents[i] {
			t.Errorf("turn %d content = %q, want %q", i, turn.Content, wantContents[i])
		}
		if turn.Sources != wantSources[i] {
			t.Errorf("turn %d sources = %q, want %q", i, turn.Sources, wantSources[i])
		}
		if turn.CreatedAt <= 0 {
			t.Errorf("turn %d created_at = %d, want positive", i, turn.CreatedAt)
		}
	}
	if turns[0].ID >= turns[1].ID {
		t.Errorf("turns not ordered oldest to newest: ids %d then %d", turns[0].ID, turns[1].ID)
	}
}

func TestRecentTurns(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	c, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation: %v", err)
	}
	for i := 1; i <= 5; i++ {
		content := "t" + string(rune('0'+i))
		if err := s.AppendTurn(ctx, c.ID, "user", content, "[]"); err != nil {
			t.Fatalf("AppendTurn %d: %v", i, err)
		}
	}
	history, err := s.FullHistory(ctx, c.ID)
	if err != nil {
		t.Fatalf("FullHistory: %v", err)
	}
	if len(history) != 5 {
		t.Fatalf("FullHistory returned %d turns, want 5", len(history))
	}

	recent, err := s.RecentTurns(ctx, c.ID, 2)
	if err != nil {
		t.Fatalf("RecentTurns: %v", err)
	}
	if len(recent) != 2 {
		t.Fatalf("RecentTurns returned %d turns, want 2", len(recent))
	}
	wantIDs := []int64{history[3].ID, history[4].ID}
	wantContents := []string{"t4", "t5"}
	for i, turn := range recent {
		if turn.ID != wantIDs[i] {
			t.Errorf("recent[%d].ID = %d, want %d", i, turn.ID, wantIDs[i])
		}
		if turn.Content != wantContents[i] {
			t.Errorf("recent[%d].Content = %q, want %q", i, turn.Content, wantContents[i])
		}
	}
}

func TestSetTitleOnlyWhenEmpty(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	c, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation: %v", err)
	}
	if err := s.SetTitle(ctx, c.ID, "first question"); err != nil {
		t.Fatalf("SetTitle(first): %v", err)
	}
	// Second call must be ignored: the title is already set.
	if err := s.SetTitle(ctx, c.ID, "ignored second"); err != nil {
		t.Fatalf("SetTitle(second): %v", err)
	}

	list, err := s.ListConversations(ctx, 10)
	if err != nil {
		t.Fatalf("ListConversations: %v", err)
	}
	if len(list) != 1 || list[0].ID != c.ID {
		t.Fatalf("ListConversations = %+v, want one row for %q", list, c.ID)
	}
	if list[0].Title != "first question" {
		t.Errorf("title = %q, want %q", list[0].Title, "first question")
	}
}

func TestAppendTurnMissingConversation(t *testing.T) {
	s := newTestStore(t)
	if err := s.AppendTurn(context.Background(), "deadbeefdeadbeefdeadbeefdeadbeef", "user", "hi", "[]"); err == nil {
		t.Fatal("AppendTurn to nonexistent conversation: want error, got nil")
	}
}

func TestAppendTurnInvalidRole(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	c, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation: %v", err)
	}
	if err := s.AppendTurn(ctx, c.ID, "system", "hi", "[]"); err == nil {
		t.Fatal("AppendTurn with role system: want error, got nil")
	}
}

func TestDeleteConversationCascades(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	c, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation: %v", err)
	}
	if err := s.AppendTurn(ctx, c.ID, "user", "one", "[]"); err != nil {
		t.Fatalf("AppendTurn: %v", err)
	}
	if err := s.AppendTurn(ctx, c.ID, "assistant", "two", "[]"); err != nil {
		t.Fatalf("AppendTurn: %v", err)
	}

	if err := s.DeleteConversation(ctx, c.ID); err != nil {
		t.Fatalf("DeleteConversation: %v", err)
	}

	var turnRows int
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM turns WHERE conversation_id = ?`, c.ID,
	).Scan(&turnRows); err != nil {
		t.Fatalf("count turns: %v", err)
	}
	if turnRows != 0 {
		t.Errorf("%d turn rows remain after delete, want 0 (cascade)", turnRows)
	}
	turns, err := s.FullHistory(ctx, c.ID)
	if err != nil {
		t.Fatalf("FullHistory after delete: %v", err)
	}
	if len(turns) != 0 {
		t.Errorf("FullHistory after delete = %d turns, want 0", len(turns))
	}

	// Deleting again (missing id) must not error.
	if err := s.DeleteConversation(ctx, c.ID); err != nil {
		t.Errorf("DeleteConversation on missing id: %v", err)
	}
}

func TestListConversations(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	older, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation(older): %v", err)
	}
	for i := 0; i < 2; i++ {
		if err := s.AppendTurn(ctx, older.ID, "user", "turn", "[]"); err != nil {
			t.Fatalf("AppendTurn: %v", err)
		}
	}
	newer, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation(newer): %v", err)
	}
	// Both conversations were touched within the same second; force a
	// distinct updated_at so the ordering is deterministic.
	if _, err := s.db.ExecContext(ctx,
		`UPDATE conversations SET updated_at = ? WHERE id = ?`,
		time.Now().Unix()+100, newer.ID,
	); err != nil {
		t.Fatalf("backdate update: %v", err)
	}

	list, err := s.ListConversations(ctx, 10)
	if err != nil {
		t.Fatalf("ListConversations: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("ListConversations returned %d rows, want 2", len(list))
	}
	if list[0].ID != newer.ID {
		t.Errorf("list[0].ID = %q, want newest %q", list[0].ID, newer.ID)
	}
	if list[0].TurnCount != 0 {
		t.Errorf("newest TurnCount = %d, want 0 (LEFT JOIN, not NULL)", list[0].TurnCount)
	}
	if list[1].ID != older.ID {
		t.Errorf("list[1].ID = %q, want oldest %q", list[1].ID, older.ID)
	}
	if list[1].TurnCount != 2 {
		t.Errorf("oldest TurnCount = %d, want 2", list[1].TurnCount)
	}
}

func TestPurgeOlderThan(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	old, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation(old): %v", err)
	}
	if err := s.AppendTurn(ctx, old.ID, "user", "stale", "[]"); err != nil {
		t.Fatalf("AppendTurn: %v", err)
	}
	if _, err := s.db.ExecContext(ctx,
		`UPDATE conversations SET updated_at = ? WHERE id = ?`,
		time.Now().Unix()-2*86400, old.ID,
	); err != nil {
		t.Fatalf("backdate update: %v", err)
	}

	fresh, err := s.CreateConversation(ctx)
	if err != nil {
		t.Fatalf("CreateConversation(fresh): %v", err)
	}

	n, err := s.PurgeOlderThan(ctx, 1)
	if err != nil {
		t.Fatalf("PurgeOlderThan(1): %v", err)
	}
	if n != 1 {
		t.Errorf("PurgeOlderThan(1) deleted %d, want 1", n)
	}

	list, err := s.ListConversations(ctx, 10)
	if err != nil {
		t.Fatalf("ListConversations: %v", err)
	}
	if len(list) != 1 || list[0].ID != fresh.ID {
		t.Fatalf("after purge ListConversations = %+v, want only %q", list, fresh.ID)
	}
	var turnRows int
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM turns WHERE conversation_id = ?`, old.ID,
	).Scan(&turnRows); err != nil {
		t.Fatalf("count turns: %v", err)
	}
	if turnRows != 0 {
		t.Errorf("%d turn rows remain for purged conversation, want 0", turnRows)
	}

	// days = 0 (and negative) never purges.
	if n, err := s.PurgeOlderThan(ctx, 0); err != nil || n != 0 {
		t.Errorf("PurgeOlderThan(0) = (%d, %v), want (0, nil)", n, err)
	}
	if n, err := s.PurgeOlderThan(ctx, -5); err != nil || n != 0 {
		t.Errorf("PurgeOlderThan(-5) = (%d, %v), want (0, nil)", n, err)
	}
	list, err = s.ListConversations(ctx, 10)
	if err != nil {
		t.Fatalf("ListConversations after no-op purge: %v", err)
	}
	if len(list) != 1 || list[0].ID != fresh.ID {
		t.Errorf("no-op purge changed conversations: %+v, want only %q", list, fresh.ID)
	}
}
