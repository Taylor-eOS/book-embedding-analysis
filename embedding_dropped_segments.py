import numpy as np

final_cache_path = "embedding_cache_final.npz"
draft_cache_path = "embedding_cache_draft.npz"
DROP_THRESHOLD = 0.35
SNIPPET_CHARS = 80
HISTOGRAM_BINS = 20

def load_cache(path):
    data = np.load(path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms

def compute_best_matches(draft_embeddings, final_embeddings):
    draft_normalized = normalize_rows(draft_embeddings)
    final_normalized = normalize_rows(final_embeddings)
    similarity_matrix = draft_normalized @ final_normalized.T
    best_indices = np.argmax(similarity_matrix, axis=1)
    best_scores = similarity_matrix[np.arange(similarity_matrix.shape[0]), best_indices]
    return best_indices, best_scores

def snippet(segments, index):
    text = str(segments[index]).strip().replace("\n", " ")
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS] + "..."

def print_score_histogram(best_scores):
    print("\nBEST-MATCH SCORE DISTRIBUTION (draft segments)")
    lo = float(best_scores.min())
    hi = float(best_scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    bin_edges = np.linspace(lo, hi, HISTOGRAM_BINS + 1)
    counts, _ = np.histogram(best_scores, bins=bin_edges)
    max_count = max(counts.max(), 1)
    for i in range(HISTOGRAM_BINS):
        bar_len = int(40 * counts[i] / max_count)
        bar = "#" * bar_len
        print(f"  {bin_edges[i]:.3f} - {bin_edges[i + 1]:.3f}  {counts[i]:>5}  {bar}")

def print_likely_dropped(draft_segments, final_segments, best_scores, best_indices, threshold):
    order = np.argsort(best_scores)
    dropped = [i for i in order if best_scores[i] < threshold]
    print(f"\nLIKELY UNSEEN SEGMENTS (best-match score below {threshold})")
    print(f"count: {len(dropped)} of {len(draft_segments)} draft segments\n")
    for i in dropped:
        print(f"draft {i:>5}  best_score {best_scores[i]:.4f}  closest_final {best_indices[i]:>5}")
        print(f"    draft:  {snippet(draft_segments, int(i))}")
        print(f"    closest: {snippet(final_segments, int(best_indices[i]))}")

def main():
    draft_segments, draft_embeddings = load_cache(draft_cache_path)
    final_segments, final_embeddings = load_cache(final_cache_path)
    best_indices, best_scores = compute_best_matches(draft_embeddings, final_embeddings)
    print(f"draft segments: {len(draft_segments)}   final segments: {len(final_segments)}")
    print_score_histogram(best_scores)
    print_likely_dropped(draft_segments, final_segments, best_scores, best_indices, DROP_THRESHOLD)

if __name__ == "__main__":
    main()
