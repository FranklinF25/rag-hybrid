package engine

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"rag-hybrid/internal/bm25"
	"rag-hybrid/internal/config"
)

// MaxUploadBytes caps each uploaded file (10 MB, same as the Python UI).
const MaxUploadBytes = 10 * 1024 * 1024

// IndexResult reports what an indexing run wrote (plus what was skipped
// and why, for uploads).
type IndexResult struct {
	Files   int
	Chunks  int
	Skipped []string
}

// UploadFile is one in-memory uploaded document (web UI path).
type UploadFile struct {
	Name string
	Data []byte
}

// chunkRef pairs a chunk's text with the file it came from.
type chunkRef struct {
	content string
	source  string
}

// Indexer orchestrates: documents -> chunks -> dense embeddings (Embedder
// port) + BM25 sparse (local) -> VectorStore points.
type Indexer struct {
	cfg   *config.Config
	embed Embedder
	store VectorStore
}

// NewIndexer wires an Indexer from its driven ports; the composition root
// builds the concrete adapters.
func NewIndexer(cfg *config.Config, store VectorStore, embed Embedder) *Indexer {
	return &Indexer{cfg: cfg, embed: embed, store: store}
}

// IndexFiles loads path (file or directory, recursive) and indexes it.
func (ix *Indexer) IndexFiles(ctx context.Context, path string, recreate bool) (IndexResult, error) {
	documents, err := LoadDocuments(path)
	if err != nil {
		return IndexResult{}, err
	}
	chunks := documentsToChunks(documents, ix.cfg.ChunkSize, ix.cfg.ChunkOverlap)
	result, err := ix.writeChunks(ctx, chunks, recreate)
	result.Files = len(documents)
	return result, err
}

// IndexUploads indexes in-memory uploaded files. Unsupported suffixes,
// oversized files, empty files and invalid UTF-8 are skipped with reasons.
func (ix *Indexer) IndexUploads(ctx context.Context, files []UploadFile) (IndexResult, error) {
	var documents []Document
	var skipped []string
	for _, file := range files {
		if !SupportedSuffixes[strings.ToLower(filepath.Ext(file.Name))] {
			skipped = append(skipped, fmt.Sprintf("%s: unsupported type (allowed: .md, .txt, .rst)", file.Name))
			continue
		}
		if len(file.Data) > MaxUploadBytes {
			skipped = append(skipped, fmt.Sprintf("%s: larger than %d MB", file.Name, MaxUploadBytes/(1024*1024)))
			continue
		}
		// PDFs are binary: extract text BEFORE the UTF-8 validation.
		if strings.ToLower(filepath.Ext(file.Name)) == ".pdf" {
			document, err := pdfToDocument(file.Name, file.Data)
			if err != nil {
				skipped = append(skipped, err.Error())
				continue
			}
			documents = append(documents, document)
			continue
		}
		if !utf8.Valid(file.Data) {
			skipped = append(skipped, fmt.Sprintf("%s: not valid UTF-8", file.Name))
			continue
		}
		content := string(file.Data)
		if strings.TrimSpace(content) == "" {
			skipped = append(skipped, fmt.Sprintf("%s: empty", file.Name))
			continue
		}
		documents = append(documents, Document{Content: content, Source: file.Name})
	}
	if len(documents) == 0 {
		return IndexResult{Skipped: skipped}, fmt.Errorf("no indexable files among %d upload(s)", len(files))
	}
	chunks := documentsToChunks(documents, ix.cfg.ChunkSize, ix.cfg.ChunkOverlap)
	result, err := ix.writeChunks(ctx, chunks, false)
	result.Files = len(documents)
	result.Skipped = skipped
	return result, err
}

// documentsToChunks splits every document into overlapping word windows.
func documentsToChunks(documents []Document, size, overlap int) []chunkRef {
	var chunks []chunkRef
	for _, doc := range documents {
		for _, text := range ChunkText(doc.Content, size, overlap) {
			chunks = append(chunks, chunkRef{content: text, source: doc.Source})
		}
	}
	return chunks
}

// writeChunks embeds (batched) and upserts every chunk, creating the
// collection first (dropping it when recreate is set).
func (ix *Indexer) writeChunks(ctx context.Context, chunks []chunkRef, recreate bool) (IndexResult, error) {
	if recreate {
		if err := ix.store.DropCollection(ctx); err != nil {
			return IndexResult{}, fmt.Errorf("dropping collection: %w", err)
		}
	}
	if err := ix.store.EnsureCollection(ctx, ix.cfg.EmbedDim); err != nil {
		return IndexResult{}, fmt.Errorf("ensuring collection: %w", err)
	}
	if !recreate {
		// Sync per source (delete-then-insert): re-uploading a document
		// replaces its chunks instead of duplicating them.
		for _, source := range distinctSources(chunks) {
			if err := ix.store.DeleteBySource(ctx, source); err != nil {
				return IndexResult{}, fmt.Errorf("syncing %s: %w", source, err)
			}
		}
	}
	batch := max(ix.cfg.EmbedBatchSize, 1)
	written := 0
	for start := 0; start < len(chunks); start += batch {
		end := min(start+batch, len(chunks))
		group := chunks[start:end]

		texts := make([]string, len(group))
		for i, c := range group {
			texts[i] = c.content
		}
		embeddings, err := ix.embed.EmbedTexts(ctx, ix.cfg.EmbedModel, texts)
		if err != nil {
			return IndexResult{Chunks: written}, fmt.Errorf("embedding batch: %w", err)
		}
		if len(embeddings) != len(group) {
			return IndexResult{Chunks: written}, fmt.Errorf("embedding batch size mismatch: got %d want %d", len(embeddings), len(group))
		}

		points := make([]Point, len(group))
		for i, c := range group {
			if len(embeddings[i]) != ix.cfg.EmbedDim {
				return IndexResult{Chunks: written}, fmt.Errorf("embedding dimension mismatch: got %d want %d (EMBED_DIM?)", len(embeddings[i]), ix.cfg.EmbedDim)
			}
			sparse := bm25.Sparse(c.content)
			id, err := newID()
			if err != nil {
				return IndexResult{Chunks: written}, err
			}
			points[i] = Point{
				ID:            id,
				Dense:         embeddings[i],
				SparseIndices: sparse.Indices,
				SparseValues:  sparse.Values,
				Content:       c.content,
				Source:        c.source,
			}
		}
		if err := ix.store.Upsert(ctx, points); err != nil {
			return IndexResult{Chunks: written}, fmt.Errorf("upserting batch: %w", err)
		}
		written += len(points)
	}
	return IndexResult{Chunks: written}, nil
}

// distinctSources lists each source exactly once, in first-seen order.
func distinctSources(chunks []chunkRef) []string {
	seen := make(map[string]bool)
	var sources []string
	for _, c := range chunks {
		if !seen[c.source] {
			seen[c.source] = true
			sources = append(sources, c.source)
		}
	}
	return sources
}

// newID returns a random 32-char hex id for a Qdrant point.
func newID() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("generating point id: %w", err)
	}
	return hex.EncodeToString(buf), nil
}
