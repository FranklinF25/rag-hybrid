package bm25

import (
	"math"
	"reflect"
	"slices"
	"testing"
)

func TestTokenizeSpanish(t *testing.T) {
	got := Tokenize("La Búsqueda Híbrida ¡rápido!")
	want := []string{"la", "búsqueda", "híbrida", "rápido"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Tokenize() = %q, want %q", got, want)
	}
}

func TestTokenizeEdgeCases(t *testing.T) {
	tests := []struct {
		name string
		text string
		want []string
	}{
		{"empty string", "", []string{}},
		{"only punctuation", "!!!", []string{}},
		{"digits kept in word", "gpt4", []string{"gpt4"}},
		{"dot splits number", "3.14", []string{"3", "14"}},
		{"underscore kept", "snake_case", []string{"snake_case"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Tokenize(tt.text)
			if got == nil {
				t.Fatalf("Tokenize(%q) returned nil slice, want non-nil", tt.text)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Tokenize(%q) = %q, want %q", tt.text, got, tt.want)
			}
		})
	}
}

func TestSparseDeterminism(t *testing.T) {
	text := "La Búsqueda Híbrida ¡rápido!"
	first := Sparse(text)
	second := Sparse(text)
	if !reflect.DeepEqual(first, second) {
		t.Errorf("Sparse() not deterministic:\nfirst  = %+v\nsecond = %+v", first, second)
	}
}

func TestSparseWordOrderIrrelevant(t *testing.T) {
	a := Sparse("denso y disperso")
	b := Sparse("disperso y denso")
	if !reflect.DeepEqual(a, b) {
		t.Errorf("word order changed the vector:\na = %+v\nb = %+v", a, b)
	}
}

func TestSparseSublinearTF(t *testing.T) {
	got := Sparse("qdrant qdrant qdrant una")
	if len(got.Indices) != 2 || len(got.Values) != 2 {
		t.Fatalf("got %d dimensions, want 2: %+v", len(got.Indices), got)
	}
	values := slices.Clone(got.Values)
	slices.Sort(values)
	want := []float64{1.0, 1 + math.Log(3)}
	for i := range want {
		if math.Abs(values[i]-want[i]) > 1e-12 {
			t.Errorf("value[%d] = %v, want %v (values = %v)", i, values[i], want[i], values)
		}
	}
}

func TestSparseIndicesAscendingUnique(t *testing.T) {
	texts := []string{
		"",
		"la búsqueda híbrida",
		"qdrant qdrant qdrant una",
		"El cóndor vuela sobre los Andes, ¡qué vista! 123 456",
	}
	for _, text := range texts {
		v := Sparse(text)
		if len(v.Indices) != len(v.Values) {
			t.Errorf("Sparse(%q): len(Indices)=%d, len(Values)=%d, want equal",
				text, len(v.Indices), len(v.Values))
		}
		for i := 1; i < len(v.Indices); i++ {
			if v.Indices[i-1] >= v.Indices[i] {
				t.Errorf("Sparse(%q): indices not strictly ascending at %d: %v",
					text, i, v.Indices)
			}
		}
	}
}

func TestSparseEmptyText(t *testing.T) {
	for _, text := range []string{"", "   ", "\t\n", "!!! ¡¡? …"} {
		v := Sparse(text)
		if v.Indices == nil || v.Values == nil {
			t.Errorf("Sparse(%q) returned nil slice(s): %+v", text, v)
		}
		if len(v.Indices) != 0 || len(v.Values) != 0 {
			t.Errorf("Sparse(%q) = %+v, want empty vector", text, v)
		}
	}
}

func TestSparseSharedTokenOverlap(t *testing.T) {
	a := Sparse("cómo funciona la fusión")
	b := Sparse("la fusión combina rankings")
	shared := 0
	for _, index := range a.Indices {
		if slices.Contains(b.Indices, index) {
			shared++
		}
	}
	if shared == 0 {
		t.Errorf("no shared indices between %v and %v", a.Indices, b.Indices)
	}
}

func TestMerge(t *testing.T) {
	a := SparseVector{Indices: []uint32{1, 3}, Values: []float64{1.5, 2.0}}
	b := SparseVector{Indices: []uint32{2, 3}, Values: []float64{0.5, 1.0}}
	got := Merge([]SparseVector{a, b})
	want := SparseVector{Indices: []uint32{1, 2, 3}, Values: []float64{1.5, 0.5, 3.0}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Merge() = %+v, want %+v", got, want)
	}

	empty := Merge(nil)
	if empty.Indices == nil || empty.Values == nil || len(empty.Indices) != 0 {
		t.Errorf("Merge(nil) = %+v, want empty non-nil vector", empty)
	}
}
