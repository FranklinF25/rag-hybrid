// Package bm25 implements a BM25-style sparse text encoder.
//
// It follows Qdrant's recommended BM25 recipe: hash tokens client-side into
// a sparse bag-of-words with sublinear term frequency, and let the server
// apply IDF weighting via the collection's Modifier.IDF. Hashed
// bag-of-words + sublinear TF + server-side Qdrant IDF together form the
// BM25 scoring function; no corpus statistics are needed client-side, so
// query and document encoding stay symmetric without a shared vocabulary
// file.
//
// v1 limitations (accepted, as in the Python original): no stemming and no
// stop-word removal, so morphological variants do not match. Add them here
// if recall on keyword queries becomes a problem.
package bm25

import (
	"crypto/md5"
	"encoding/binary"
	"math"
	"regexp"
	"slices"
	"strings"
)

// tokenRE matches runs of unicode letters, digits, and underscore; every
// other character (whitespace, punctuation, symbols) is a separator.
var tokenRE = regexp.MustCompile(`[\p{L}\p{N}_]+`)

// hashSpace is the size of the hashed index space, 2^31: every token maps
// to a stable index below this bound.
const hashSpace = 1 << 31

// SparseVector is a sparse embedding: strictly ascending, unique dimension
// indices with parallel values.
type SparseVector struct {
	Indices []uint32
	Values  []float64
}

// Tokenize splits text into lowercase unicode word tokens (unicode letters,
// digits, and underscore; accents survive lowercasing). Empty or
// separator-only input yields an empty, non-nil slice.
func Tokenize(text string) []string {
	raw := tokenRE.FindAllString(text, -1)
	tokens := make([]string, 0, len(raw))
	for _, token := range raw {
		tokens = append(tokens, strings.ToLower(token))
	}
	return tokens
}

// Sparse encodes text as a hashed bag-of-words with sublinear term
// frequency: each distinct token contributes value 1 + ln(tf) at a stable
// index (see tokenIndex). IDF weighting and document-length normalization
// are applied server-side by Qdrant's IDF modifier, so the values here are
// pure term frequencies. When two distinct tokens hash to the same index
// their values are summed — the Python original kept duplicate indices, but
// unique indices are required for a well-formed sparse vector, so summing
// is this port's documented divergence. Empty or token-free text yields an
// empty, non-nil vector.
func Sparse(text string) SparseVector {
	counts := make(map[string]int)
	for _, token := range Tokenize(text) {
		counts[token]++
	}

	weighted := make(map[uint32]float64, len(counts))
	for token, tf := range counts {
		weighted[tokenIndex(token)] += 1 + math.Log(float64(tf))
	}
	return newSparseVector(weighted)
}

// Merge sums values per index across the given vectors and returns a single
// vector with strictly ascending, unique indices.
func Merge(vectors []SparseVector) SparseVector {
	sum := make(map[uint32]float64)
	for _, vector := range vectors {
		for i, index := range vector.Indices {
			sum[index] += vector.Values[i]
		}
	}
	return newSparseVector(sum)
}

// tokenIndex maps a token to a stable sparse dimension: the first 8 hex
// characters of the MD5 of the token's UTF-8 bytes, parsed as an integer and
// reduced modulo 2^31. The first 8 hex characters of an MD5 digest are
// exactly its first 4 bytes big-endian. MD5 is used only as a stable hash,
// never for security.
func tokenIndex(token string) uint32 {
	sum := md5.Sum([]byte(token))
	return binary.BigEndian.Uint32(sum[:4]) % hashSpace
}

// newSparseVector materializes an index-to-value map as a SparseVector with
// strictly ascending indices and parallel values; both slices are non-nil.
func newSparseVector(values map[uint32]float64) SparseVector {
	indices := make([]uint32, 0, len(values))
	for index := range values {
		indices = append(indices, index)
	}
	slices.Sort(indices)

	sorted := make([]float64, len(indices))
	for i, index := range indices {
		sorted[i] = values[index]
	}
	return SparseVector{Indices: indices, Values: sorted}
}
