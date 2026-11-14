// Package engine implements the rag-hybrid flows: document loading,
// chunking, indexing (dense + sparse into Qdrant), hybrid retrieval with
// reranking, and answer generation via DeepSeek. It is the Go port of the
// Python src/rag_hybrid indexing/retrieval pipelines.
package engine

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Document is one source file to index.
type Document struct {
	Content string
	Source  string // relative or absolute path, kept as payload for citations
}

// SupportedSuffixes lists the file types accepted for indexing. PDFs go
// through a text-layer extractor (scanned PDFs are rejected with a reason).
var SupportedSuffixes = map[string]bool{".md": true, ".txt": true, ".rst": true, ".pdf": true}

// LoadDocuments reads every supported file under path (a file or a
// directory, recursive). It errors when nothing indexable is found.
func LoadDocuments(path string) ([]Document, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("stating %s: %w", path, err)
	}

	var files []string
	if !info.IsDir() {
		files = []string{path}
	} else {
		err = filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			if SupportedSuffixes[strings.ToLower(filepath.Ext(p))] {
				files = append(files, p)
			}
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("walking %s: %w", path, err)
		}
	}

	documents := make([]Document, 0, len(files))
	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", file, err)
		}
		if strings.ToLower(filepath.Ext(file)) == ".pdf" {
			document, err := pdfToDocument(file, content)
			if err != nil {
				return nil, err
			}
			documents = append(documents, document)
			continue
		}
		if strings.TrimSpace(string(content)) == "" {
			continue
		}
		documents = append(documents, Document{Content: string(content), Source: file})
	}
	if len(documents) == 0 {
		return nil, fmt.Errorf("no supported non-empty files (.md/.txt/.rst) found in: %s", path)
	}
	return documents, nil
}
