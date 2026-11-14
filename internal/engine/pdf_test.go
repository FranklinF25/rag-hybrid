package engine

import (
	"os"
	"strings"
	"testing"
)

func TestExtractPDFTextFromFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/sample.pdf")
	if err != nil {
		t.Fatalf("reading fixture: %v", err)
	}
	text, err := ExtractPDFText(data)
	if err != nil {
		t.Fatalf("extracting: %v", err)
	}
	if !strings.Contains(text, "Kappa frequency") || !strings.Contains(text, "42 hertz") {
		t.Fatalf("expected marker text missing, got: %q", text)
	}
}

func TestPDFToDocumentFixture(t *testing.T) {
	data, _ := os.ReadFile("testdata/sample.pdf")
	doc, err := pdfToDocument("sample.pdf", data)
	if err != nil {
		t.Fatalf("pdfToDocument: %v", err)
	}
	if doc.Source != "sample.pdf" || strings.TrimSpace(doc.Content) == "" {
		t.Fatalf("bad document: %+v", doc)
	}
}

func TestPDFGarbageFails(t *testing.T) {
	if _, err := pdfToDocument("broken.pdf", []byte("this is not a pdf")); err == nil {
		t.Fatal("expected an error for garbage bytes")
	}
}

func TestPDFNoTextFails(t *testing.T) {
	// A structurally-valid PDF with no text objects: hand-built minimal
	// document whose content stream only draws a rectangle.
	minimal := "%PDF-1.4\n" +
		"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n" +
		"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n" +
		"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Contents 4 0 R >> endobj\n" +
		"4 0 obj << /Length 44 >> stream\n1 0 0 RG 10 10 80 80 re S\nendstream endobj\n" +
		"trailer << /Root 1 0 R >>\n%%EOF\n"
	if _, err := pdfToDocument("scanned.pdf", []byte(minimal)); err == nil {
		t.Fatal("expected a no-text error (scanned-PDF path)")
	}
}
