// Package engine: PDF text extraction. Text-layer PDFs are supported;
// scanned (image-only) PDFs yield no text and are skipped upstream with a
// clear reason. Whole-document extraction (no per-page markers in v1 —
// the plain-text API is document-level).
package engine

import (
	"bytes"
	"fmt"
	"io"
	"strings"

	pdfapi "github.com/ledongthuc/pdf"
)

// ExtractPDFText pulls the text layer of the whole document. It returns
// an error for structurally broken files and empty text for scanned
// documents (the caller decides how to report that).
func ExtractPDFText(data []byte) (string, error) {
	reader, err := pdfapi.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return "", fmt.Errorf("opening PDF: %w", err)
	}
	textReader, err := reader.GetPlainText()
	if err != nil {
		return "", fmt.Errorf("extracting PDF text: %w", err)
	}
	text, err := io.ReadAll(textReader)
	if err != nil {
		return "", fmt.Errorf("reading PDF text: %w", err)
	}
	return string(text), nil
}

// pdfToDocument converts extracted PDF text into a Document, failing
// explicitly when nothing could be extracted (usually a scanned file).
func pdfToDocument(name string, data []byte) (Document, error) {
	text, err := ExtractPDFText(data)
	if err != nil {
		return Document{}, err
	}
	if strings.TrimSpace(text) == "" {
		return Document{}, fmt.Errorf("%s: no extractable text (scanned/image PDF? OCR is not supported)", name)
	}
	return Document{Content: text, Source: name}, nil
}
