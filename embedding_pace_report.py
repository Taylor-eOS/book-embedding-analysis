import numpy as np

cache_path = "embedding_cache.npz"
LENGTH_UNIT = "words"
SMOOTHING_WINDOW = 5
TOP_FLAG_COUNT = 25
MIN_LENGTH_FOR_RATIO = 5
SNIPPET_CHARS = 90

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms

def cosine_distance(a, b):
    return float(1.0 - np.dot(a, b))

def compute_segment_lengths(segments, unit):
    lengths = np.zeros(len(segments))
    for i, s in enumerate(segments):
        if unit == "words":
            lengths[i] = len(s.split())
        else:
            lengths[i] = len(s)
    return lengths

def compute_step_distances(embeddings_normalized):
    n = embeddings_normalized.shape[0]
    distances = np.zeros(n)
    for i in range(1, n):
        distances[i] = cosine_distance(embeddings_normalized[i - 1], embeddings_normalized[i])
    return distances

def compute_density_ratio(step_distances, lengths, min_length):
    n = len(step_distances)
    ratio = np.full(n, np.nan)
    for i in range(1, n):
        if lengths[i] < min_length:
            continue
        ratio[i] = step_distances[i] / lengths[i]
    return ratio

def compute_smoothed_pace(step_distances, window):
    n = len(step_distances)
    smoothed = np.zeros(n)
    for i in range(n):
        low = max(0, i - window + 1)
        smoothed[i] = step_distances[low:i + 1].mean()
    return smoothed

def zscore_ignoring_nan(values):
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.zeros_like(values)
    mean = valid.mean()
    std = valid.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std

def snippet(segments, index):
    text = str(segments[index]).strip().replace("\n", " ")
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS] + "..."

def print_pace_curve(step_distances, smoothed_pace, bucket_count=40):
    n = len(step_distances)
    print("\nSEMANTIC PACE CURVE (smoothed step distance across the book)")
    print("(higher bars mean the book is moving through meaning faster at that point)\n")
    bucket_size = max(1, n // bucket_count)
    max_val = smoothed_pace.max() if smoothed_pace.max() > 0 else 1.0
    for start in range(0, n, bucket_size):
        end = min(n, start + bucket_size)
        bucket_mean = smoothed_pace[start:end].mean()
        bar_len = int(50 * bucket_mean / max_val)
        bar = "#" * bar_len
        print(f"  segments {start:>5}-{end - 1:<5}  {bucket_mean:.4f}  {bar}")

def print_dense_outliers(segments, density_ratio, lengths, step_distances):
    print(f"\nTOP {TOP_FLAG_COUNT} DENSEST SEGMENTS")
    print("(large semantic jump packed into relatively few words, possibly under-explained)\n")
    density_z = zscore_ignoring_nan(density_ratio)
    eligible = [i for i in range(len(segments)) if not np.isnan(density_ratio[i])]
    order = sorted(eligible, key=lambda i: density_z[i], reverse=True)[:TOP_FLAG_COUNT]
    for index in order:
        print(f"segment {index:>5}  density_z {density_z[index]:.3f}  step_dist {step_distances[index]:.4f}  length {int(lengths[index])}")
        print(f"    {snippet(segments, index)}")

def print_inert_outliers(segments, density_ratio, lengths, step_distances):
    print(f"\nTOP {TOP_FLAG_COUNT} MOST INERT SEGMENTS")
    print("(long relative to the semantic ground actually covered, possibly padding)\n")
    density_z = zscore_ignoring_nan(density_ratio)
    eligible = [i for i in range(len(segments)) if not np.isnan(density_ratio[i])]
    order = sorted(eligible, key=lambda i: density_z[i])[:TOP_FLAG_COUNT]
    for index in order:
        print(f"segment {index:>5}  density_z {density_z[index]:.3f}  step_dist {step_distances[index]:.4f}  length {int(lengths[index])}")
        print(f"    {snippet(segments, index)}")

def main():
    segments, embeddings = load_cache()
    if len(segments) < SMOOTHING_WINDOW + 2:
        print(f"Need at least {SMOOTHING_WINDOW + 2} segments for this analysis.")
        return
    embeddings_normalized = normalize_rows(embeddings)
    lengths = compute_segment_lengths(segments, LENGTH_UNIT)
    step_distances = compute_step_distances(embeddings_normalized)
    smoothed_pace = compute_smoothed_pace(step_distances, SMOOTHING_WINDOW)
    density_ratio = compute_density_ratio(step_distances, lengths, MIN_LENGTH_FOR_RATIO)
    print_pace_curve(step_distances, smoothed_pace)
    print_dense_outliers(segments, density_ratio, lengths, step_distances)
    print_inert_outliers(segments, density_ratio, lengths, step_distances)

if __name__ == "__main__":
    main()
