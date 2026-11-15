package engine

import "context"

// Driven ports: the interfaces this package (the domain core) depends on.
// The engine defines them; the adapter packages (qdrant, openrouter,
// deepseek) implement them with their concrete clients; the composition
// roots (cmd/rag-hybrid and the server package) inject the adapters.
// Dependencies point inward: this package imports no adapter.

// Hit is one retrieved chunk with its relevance score.
type Hit struct {
	ID      string
	Source  string
	Content string
	Score   float64
}

// Point is one document chunk ready for upsert into the vector store: a
// dense embedding, a sparse (BM25-style) vector and the payload fields
// stored alongside them.
type Point struct {
	ID            string
	Dense         []float64
	SparseIndices []uint32
	SparseValues  []float64
	Content       string
	Source        string
}

// SourceStat aggregates the chunk count of one indexed source file.
type SourceStat struct {
	Source string `json:"source"`
	Chunks int    `json:"chunks"`
}

// ScoredDoc is one rerank result: Index points into the documents slice
// that was passed to Rerank; Score is the cross-encoder relevance score.
type ScoredDoc struct {
	Index int
	Score float64
}

// VectorStore is the port for the vector database: hybrid retrieval for
// the query side, collection writes for indexing, and the per-source
// operations behind document sync, listing and forgetting.
type VectorStore interface {
	HybridQuery(ctx context.Context, dense []float64, sparseIndices []uint32, sparseValues []float64, limit int) ([]Hit, error)
	Upsert(ctx context.Context, points []Point) error
	EnsureCollection(ctx context.Context, dim int) error
	DropCollection(ctx context.Context) error
	DeleteBySource(ctx context.Context, source string) error
	ListSources(ctx context.Context) ([]SourceStat, error)
	PointsCount(ctx context.Context) (int, error)
}

// Embedder is the port for dense embedding models.
type Embedder interface {
	EmbedTexts(ctx context.Context, model string, texts []string) ([][]float64, error)
}

// Reranker is the port for reranking models.
type Reranker interface {
	Rerank(ctx context.Context, model, query string, documents []string, topN int) ([]ScoredDoc, error)
}

// LLM is the port for text generation (final answers and query
// rewriting): one-shot and streaming completions.
type LLM interface {
	Complete(ctx context.Context, model, prompt string, maxTokens int) (string, error)
	CompleteStream(ctx context.Context, model, prompt string, maxTokens int, onDelta func(string) error) (string, error)
}
