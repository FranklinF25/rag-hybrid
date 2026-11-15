// Package store implements the persistent conversation store for the hybrid
// RAG chat application. Conversations and their turns are kept in an embedded
// SQLite database using the pure-Go driver (no cgo), so the binary stays
// statically linked.
package store

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// Conversation is a persisted chat conversation. Timestamps are unix seconds.
type Conversation struct {
	ID        string
	Title     string
	CreatedAt int64
	UpdatedAt int64
	TurnCount int
}

// Turn is a single message of a conversation. Sources holds the JSON encoding
// of the citations for an answer; the store treats it as an opaque string.
type Turn struct {
	ID        int64
	Role      string
	Content   string
	Sources   string
	CreatedAt int64
}

// Store is a persistent conversation store backed by SQLite. It is safe for
// concurrent use: database/sql pools connections and the WAL journal mode
// allows concurrent readers alongside a single writer.
type Store struct {
	db *sql.DB
}

// schema is applied on every Open; each statement is idempotent.
var schema = []string{
	`CREATE TABLE IF NOT EXISTS conversations (
		id         TEXT PRIMARY KEY,
		title      TEXT NOT NULL DEFAULT '',
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL
	)`,
	`CREATE TABLE IF NOT EXISTS turns (
		id              INTEGER PRIMARY KEY AUTOINCREMENT,
		conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
		role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
		content         TEXT NOT NULL,
		sources         TEXT NOT NULL DEFAULT '[]',
		created_at      INTEGER NOT NULL
	)`,
	`CREATE INDEX IF NOT EXISTS idx_turns_conversation ON turns(conversation_id, id)`,
}

// Open opens the SQLite database at path, creating the file and its parent
// directory if missing, and ensures the schema is present. Reopening an
// existing database is safe (CREATE ... IF NOT EXISTS).
func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("store: create database directory: %w", err)
	}
	// Per-connection pragmas ride along in the DSN so every pooled
	// connection gets them on open: enforce foreign keys (cascade deletes),
	// WAL journaling, and a busy timeout so concurrent access does not fail
	// immediately with SQLITE_BUSY.
	dsn := path + "?_pragma=foreign_keys(1)&_pragma=journal_mode(WAL)&_pragma=busy_timeout(10000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("store: open database: %w", err)
	}
	for _, stmt := range schema {
		if _, err := db.Exec(stmt); err != nil {
			db.Close()
			return nil, fmt.Errorf("store: apply schema: %w", err)
		}
	}
	// The DSN covers connections opened by the pool later; this also covers
	// the connection that just applied the schema.
	if _, err := db.Exec(`PRAGMA foreign_keys = ON`); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: enable foreign keys: %w", err)
	}
	return &Store{db: db}, nil
}

// Close closes the store's database.
func (s *Store) Close() error {
	return s.db.Close()
}

// CreateConversation creates a new conversation with a random 32-hex-char id,
// an empty title, and both timestamps set to now.
func (s *Store) CreateConversation(ctx context.Context) (Conversation, error) {
	id, err := newConversationID()
	if err != nil {
		return Conversation{}, fmt.Errorf("store: generate conversation id: %w", err)
	}
	now := time.Now().Unix()
	if _, err := s.db.ExecContext(ctx,
		`INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)`,
		id, "", now, now,
	); err != nil {
		return Conversation{}, fmt.Errorf("store: create conversation: %w", err)
	}
	return Conversation{ID: id, CreatedAt: now, UpdatedAt: now}, nil
}

// SetTitle sets the title of a conversation, but only while its current title
// is empty (the first user turn names the chat). Later calls are no-ops, and
// setting a title on a missing conversation is not an error.
func (s *Store) SetTitle(ctx context.Context, id, title string) error {
	if _, err := s.db.ExecContext(ctx,
		`UPDATE conversations SET title = ? WHERE id = ? AND title = ''`,
		title, id,
	); err != nil {
		return fmt.Errorf("store: set title: %w", err)
	}
	return nil
}

// AppendTurn appends a turn to the conversation and touches its updated_at.
// role must be "user" or "assistant" (enforced by the schema); sourcesJSON is
// stored as-is. Appending to a nonexistent conversation is an error.
func (s *Store) AppendTurn(ctx context.Context, conversationID, role, content, sourcesJSON string) error {
	var exists bool
	if err := s.db.QueryRowContext(ctx,
		`SELECT EXISTS(SELECT 1 FROM conversations WHERE id = ?)`, conversationID,
	).Scan(&exists); err != nil {
		return fmt.Errorf("store: check conversation: %w", err)
	}
	if !exists {
		return fmt.Errorf("store: conversation %q does not exist", conversationID)
	}
	now := time.Now().Unix()
	if _, err := s.db.ExecContext(ctx,
		`INSERT INTO turns (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)`,
		conversationID, role, content, sourcesJSON, now,
	); err != nil {
		return fmt.Errorf("store: append turn: %w", err)
	}
	if _, err := s.db.ExecContext(ctx,
		`UPDATE conversations SET updated_at = ? WHERE id = ?`, now, conversationID,
	); err != nil {
		return fmt.Errorf("store: touch conversation: %w", err)
	}
	return nil
}

// ListConversations returns up to limit conversations ordered by updated_at
// descending, each with its turn count.
func (s *Store) ListConversations(ctx context.Context, limit int) ([]Conversation, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(t.id)
		FROM conversations c
		LEFT JOIN turns t ON t.conversation_id = c.id
		GROUP BY c.id
		ORDER BY c.updated_at DESC, c.id DESC
		LIMIT ?`, limit)
	if err != nil {
		return nil, fmt.Errorf("store: list conversations: %w", err)
	}
	defer rows.Close()
	var out []Conversation
	for rows.Next() {
		var c Conversation
		if err := rows.Scan(&c.ID, &c.Title, &c.CreatedAt, &c.UpdatedAt, &c.TurnCount); err != nil {
			return nil, fmt.Errorf("store: scan conversation: %w", err)
		}
		out = append(out, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: list conversations: %w", err)
	}
	return out, nil
}

// RecentTurns returns the last limit turns of the conversation ordered
// oldest to newest.
func (s *Store) RecentTurns(ctx context.Context, conversationID string, limit int) ([]Turn, error) {
	return s.queryTurns(ctx, `
		SELECT id, role, content, sources, created_at
		FROM turns
		WHERE id IN (
			SELECT id FROM turns WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
		)
		ORDER BY id ASC`, conversationID, limit)
}

// FullHistory returns every turn of the conversation ordered oldest to
// newest. A missing conversation yields an empty slice, not an error.
func (s *Store) FullHistory(ctx context.Context, conversationID string) ([]Turn, error) {
	return s.queryTurns(ctx, `
		SELECT id, role, content, sources, created_at
		FROM turns
		WHERE conversation_id = ?
		ORDER BY id ASC`, conversationID)
}

// DeleteConversation deletes a conversation and, via the schema's ON DELETE
// CASCADE, all of its turns. Deleting a missing conversation is not an error.
func (s *Store) DeleteConversation(ctx context.Context, conversationID string) error {
	if _, err := s.db.ExecContext(ctx,
		`DELETE FROM conversations WHERE id = ?`, conversationID,
	); err != nil {
		return fmt.Errorf("store: delete conversation: %w", err)
	}
	return nil
}

// PurgeOlderThan deletes conversations whose updated_at is older than days
// days and returns how many were deleted. days <= 0 never purges anything.
func (s *Store) PurgeOlderThan(ctx context.Context, days int) (int64, error) {
	if days <= 0 {
		return 0, nil
	}
	cutoff := time.Now().Unix() - int64(days)*86400
	res, err := s.db.ExecContext(ctx,
		`DELETE FROM conversations WHERE updated_at < ?`, cutoff,
	)
	if err != nil {
		return 0, fmt.Errorf("store: purge conversations: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("store: count purged conversations: %w", err)
	}
	return n, nil
}

func (s *Store) queryTurns(ctx context.Context, query string, args ...any) ([]Turn, error) {
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("store: query turns: %w", err)
	}
	defer rows.Close()
	var out []Turn
	for rows.Next() {
		var t Turn
		if err := rows.Scan(&t.ID, &t.Role, &t.Content, &t.Sources, &t.CreatedAt); err != nil {
			return nil, fmt.Errorf("store: scan turn: %w", err)
		}
		out = append(out, t)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: query turns: %w", err)
	}
	return out, nil
}

// newConversationID returns a random 32-character lowercase hex string.
func newConversationID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}
