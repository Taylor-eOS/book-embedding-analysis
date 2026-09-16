import numpy as np
from embedding_report import detect_chapter_boundaries

cache_path = "embedding_cache.npz"
SIMILARITY_THRESHOLD = 0.90
TOP_PAIR_COUNT = 30
SNIPPET_CHARS = 70

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms

def boundaries_to_chapter_ranges(boundaries, total_segments):
    ranges = []
    start = 0
    for b in boundaries:
        ranges.append((start, b))
        start = b
    ranges.append((start, total_segments))
    return ranges

def compute_chapter_centroids(embeddings_normalized, chapter_ranges):
    centroids = []
    for start, end in chapter_ranges:
        chapter_vectors = embeddings_normalized[start:end]
        centroid = chapter_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            centroid_norm = 1.0
        centroids.append(centroid / centroid_norm)
    return np.array(centroids)

def compute_similarity_matrix(centroids):
    return centroids @ centroids.T

def chapter_snippet(segments, chapter_ranges, chapter_index):
    start, end = chapter_ranges[chapter_index]
    text = segments[start].strip().replace("\n", " ")
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS] + "..."

def find_best_match_per_chapter(similarity_matrix):
    n = similarity_matrix.shape[0]
    matches = []
    for i in range(n):
        row = similarity_matrix[i].copy()
        row[i] = -1.0
        best_j = int(np.argmax(row))
        matches.append((i, best_j, float(row[best_j])))
    return matches

def find_top_pairs(similarity_matrix, top_n):
    n = similarity_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(similarity_matrix[i, j])))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top_n]

def find_clusters(similarity_matrix, threshold):
    n = similarity_matrix.shape[0]
    visited = [False] * n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        frontier = [i]
        while frontier:
            current = frontier.pop()
            for j in range(n):
                if not visited[j] and similarity_matrix[current, j] >= threshold:
                    visited[j] = True
                    group.append(j)
                    frontier.append(j)
        if len(group) > 1:
            clusters.append(sorted(group))
    return clusters

def print_most_redundant_chapters(segments, chapter_ranges, matches):
    ranked = sorted(matches, key=lambda m: m[2], reverse=True)
    print(f"\nTOP {len(ranked)} MOST REDUNDANT SEGMENTS")
    print("(each chapter compared against every other chapter in the book)\n")
    for i, j, sim in ranked:
        print(f"chapter {i:>3}  closest match: chapter {j:>3}  similarity {sim:.4f}")
        print(f"    [{i}] {chapter_snippet(segments, chapter_ranges, i)}")
        print(f"    [{j}] {chapter_snippet(segments, chapter_ranges, j)}")

def print_top_pairs(segments, chapter_ranges, pairs):
    print(f"\nTOP {len(pairs)} MOST SIMILAR PAIRS")
    print("(ranked globally across all possible pairs)\n")
    for i, j, sim in pairs:
        print(f"chapter {i:>3} <-> chapter {j:>3}  similarity {sim:.4f}")
        print(f"    [{i}] {chapter_snippet(segments, chapter_ranges, i)}")
        print(f"    [{j}] {chapter_snippet(segments, chapter_ranges, j)}")

def print_clusters(segments, chapter_ranges, clusters, threshold):
    print(f"\nCLUSTERS (mutual similarity >= {threshold})")
    if not clusters:
        print("No clusters found at this threshold.")
        return
    for cluster in clusters:
        label = ", ".join(str(c) for c in cluster)
        print(f"\ncluster: {label}")
        for c in cluster:
            print(f"    [{c}] {chapter_snippet(segments, chapter_ranges, c)}")

def main():
    segments, embeddings = load_cache()
    if len(segments) < 2:
        print("Need at least 2 segments to analyze chapter redundancy.")
        return
    boundaries = detect_chapter_boundaries(len(segments))
    chapter_ranges = boundaries_to_chapter_ranges(boundaries, len(segments))
    if len(chapter_ranges) < 2:
        print("Need at least 2 chapters to analyze redundancy.")
        return
    embeddings_normalized = normalize_rows(embeddings)
    centroids = compute_chapter_centroids(embeddings_normalized, chapter_ranges)
    similarity_matrix = compute_similarity_matrix(centroids)
    matches = find_best_match_per_chapter(similarity_matrix)
    top_pairs = find_top_pairs(similarity_matrix, TOP_PAIR_COUNT)
    clusters = find_clusters(similarity_matrix, SIMILARITY_THRESHOLD)
    print_most_redundant_chapters(segments, chapter_ranges, matches)
    print_top_pairs(segments, chapter_ranges, top_pairs)
    print_clusters(segments, chapter_ranges, clusters, SIMILARITY_THRESHOLD)

if __name__ == "__main__":
    main()
