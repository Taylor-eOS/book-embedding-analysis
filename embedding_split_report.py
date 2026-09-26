import numpy as np

cache_path = "embedding_cache.npz"
input_text_path = "input.txt"
MIN_SUBCHAPTER_SEGMENTS = 3
MAX_SPLITS_PER_CHAPTER = 2
SIGNIFICANCE_PERMUTATIONS = 300
SIGNIFICANCE_LEVEL = 0.05
RANDOM_SEED = 0
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

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def detect_chapter_lengths(total_segments):
    with open(input_text_path, "r", encoding="utf-8") as file:
        text = file.read()
    paragraphs = text.split("\n\n")
    if len(paragraphs) != total_segments:
        raise ValueError(f"{input_text_path} splits into {len(paragraphs)} paragraphs on blank lines but the cache has {total_segments} segments.")
    if not paragraphs[0].startswith(" "):
        raise ValueError(f"The first paragraph of {input_text_path} does not start with a space, so the first chapter boundary is undefined.")
    lengths = []
    current_length = 0
    for paragraph in paragraphs:
        if paragraph.startswith(" "):
            if current_length > 0:
                lengths.append(current_length)
            current_length = 1
        else:
            current_length += 1
    lengths.append(current_length)
    return lengths

def chapter_lengths_to_ranges(lengths, total_segments):
    ranges = []
    start = 0
    for length in lengths:
        end = start + length
        ranges.append((start, end))
        start = end
    if start != total_segments:
        raise ValueError(f"Detected chapter lengths sum to {start} segments but the cache has {total_segments}.")
    return ranges

def within_cluster_dispersion(embeddings_normalized):
    if embeddings_normalized.shape[0] == 0:
        return 0.0
    centroid = normalize_vector(embeddings_normalized.mean(axis=0))
    distances = 1.0 - embeddings_normalized @ centroid
    return float(distances.sum())

def best_split_dispersion(embeddings_normalized, min_side):
    n = embeddings_normalized.shape[0]
    if n < min_side * 2:
        return None
    best_i = None
    best_dispersion = None
    for i in range(min_side, n - min_side + 1):
        left_dispersion = within_cluster_dispersion(embeddings_normalized[:i])
        right_dispersion = within_cluster_dispersion(embeddings_normalized[i:])
        combined = left_dispersion + right_dispersion
        if best_dispersion is None or combined < best_dispersion:
            best_dispersion = combined
            best_i = i
    if best_i is None:
        return None
    return best_i, best_dispersion

def permutation_p_value(embeddings_normalized, min_side, observed_dispersion, rng, n_permutations):
    n = embeddings_normalized.shape[0]
    at_least_as_good = 0
    for _ in range(n_permutations):
        shuffled = embeddings_normalized[rng.permutation(n)]
        result = best_split_dispersion(shuffled, min_side)
        if result is None:
            continue
        _, shuffled_dispersion = result
        if shuffled_dispersion <= observed_dispersion:
            at_least_as_good += 1
    return (at_least_as_good + 1) / (n_permutations + 1)

def best_significant_split(embeddings_normalized, min_side, significance_level, rng, n_permutations):
    n = embeddings_normalized.shape[0]
    whole_dispersion = within_cluster_dispersion(embeddings_normalized)
    if whole_dispersion == 0:
        return None
    result = best_split_dispersion(embeddings_normalized, min_side)
    if result is None:
        return None
    split_index, best_dispersion = result
    improvement_ratio = (whole_dispersion - best_dispersion) / whole_dispersion
    p_value = permutation_p_value(embeddings_normalized, min_side, best_dispersion, rng, n_permutations)
    if p_value > significance_level:
        return None
    return split_index, improvement_ratio, p_value

def recursive_splits(embeddings_normalized, offset, min_side, significance_level, rng, n_permutations, max_splits):
    if max_splits <= 0:
        return []
    result = best_significant_split(embeddings_normalized, min_side, significance_level, rng, n_permutations)
    if result is None:
        return []
    split_index, improvement_ratio, p_value = result
    global_index = split_index + offset
    left_splits = recursive_splits(embeddings_normalized[:split_index], offset, min_side, significance_level, rng, n_permutations, max_splits - 1)
    right_splits = recursive_splits(embeddings_normalized[split_index:], global_index, min_side, significance_level, rng, n_permutations, max_splits - 1)
    return left_splits + [(global_index, improvement_ratio, p_value)] + right_splits

def snippet(segments, index):
    text = segments[index].strip().replace("\n", " ")
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS] + "..."

def print_chapter_result(chapter_number, start, end, segments, splits):
    length = end - start
    print(f"\nchapter {chapter_number:>3}  segments {start:>5}-{end - 1:<5}  ({length} subchapters)")
    if not splits:
        print("    no split recommended")
        return
    for global_index, improvement_ratio, p_value in splits:
        local_position = global_index - start
        print(f"    suggested split before subchapter {local_position} (segment {global_index})  dispersion_improvement {improvement_ratio:.1%}  p={p_value:.3f}")
        print(f"        before: {snippet(segments, global_index - 1)}")
        print(f"        after:  {snippet(segments, global_index)}")

def main():
    segments, embeddings = load_cache()
    chapter_lengths = detect_chapter_lengths(embeddings.shape[0])
    chapter_ranges = chapter_lengths_to_ranges(chapter_lengths, embeddings.shape[0])
    embeddings_normalized = normalize_rows(embeddings)
    rng = np.random.default_rng(RANDOM_SEED)
    for chapter_number, (start, end) in enumerate(chapter_ranges, 1):
        chapter_embeddings = embeddings_normalized[start:end]
        splits = recursive_splits(chapter_embeddings, start, MIN_SUBCHAPTER_SEGMENTS, SIGNIFICANCE_LEVEL, rng, SIGNIFICANCE_PERMUTATIONS, MAX_SPLITS_PER_CHAPTER)
        splits.sort(key=lambda triple: triple[0])
        print_chapter_result(chapter_number, start, end, segments, splits)

if __name__ == "__main__":
    main()
