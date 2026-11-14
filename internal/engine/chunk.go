package engine

import "strings"

// ChunkText splits text into overlapping word windows: each chunk holds up
// to size words and consecutive chunks share overlap words (the splitter's
// step is size-overlap). This mirrors the Python DocumentSplitter
// (split_by=word, split_length=size, split_overlap=overlap).
func ChunkText(text string, size, overlap int) []string {
	if size <= 0 {
		size = 1
	}
	if overlap < 0 || overlap >= size {
		overlap = 0
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	step := size - overlap
	var chunks []string
	for start := 0; start < len(words); start += step {
		end := start + size
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[start:end], " "))
		if end == len(words) {
			break
		}
	}
	return chunks
}
