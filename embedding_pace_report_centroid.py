import numpy as np

cache_path = "embedding_cache.npz"
LENGTH_UNIT = "words"
SMOOTHING_WINDOW = 5
TOP_FLAG_COUNT = 20
MIN_LENGTH_FOR_RATIO = 7
SNIPPET_CHARS = 80
LOCAL_CENTROID_WINDOW = 3
LOCAL_ZSCORE_WINDOW = 61
TREND_POLY_DEGREE = 2

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

def compute_local_centroids(embeddings_normalized, window):
    n = embeddings_normalized.shape[0]
    centroids = np.zeros_like(embeddings_normalized)
    for i in range(n):
        low = max(0, i - window)
        high = i
        if low == high:
            centroids[i] = embeddings_normalized[i]
            continue
        block = embeddings_normalized[low:high]
        mean_vec = block.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        centroids[i] = mean_vec / norm if norm > 0 else embeddings_normalized[i]
    return centroids

def compute_step_distances(embeddings_normalized, local_centroids):
    n = embeddings_normalized.shape[0]
    distances = np.zeros(n)
    for i in range(1, n):
        distances[i] = cosine_distance(local_centroids[i - 1], embeddings_normalized[i])
    return distances

def fit_length_trend(step_distances, lengths, min_length, degree):
    mask = (lengths >= min_length) & (np.arange(len(lengths)) > 0)
    x = lengths[mask]
    y = step_distances[mask]
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

def compute_length_residuals(step_distances, lengths, min_length, trend_poly):
    n = len(step_distances)
    residuals = np.full(n, np.nan)
    for i in range(1, n):
        if lengths[i] < min_length:
            continue
        predicted = trend_poly(lengths[i])
        residuals[i] = step_distances[i] - predicted
    return residuals

def compute_smoothed_pace(step_distances, window):
    n = len(step_distances)
    smoothed = np.zeros(n)
    for i in range(n):
        low = max(0, i - window + 1)
        smoothed[i] = step_distances[low:i + 1].mean()
    return smoothed

def rolling_zscore_ignoring_nan(values, window):
    n = len(values)
    z = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        if np.isnan(values[i]):
            continue
        low = max(0, i - half)
        high = min(n, i + half + 1)
        block = values[low:high]
        valid = block[~np.isnan(block)]
        if valid.size < 2:
            continue
        mean = valid.mean()
        std = valid.std()
        if std == 0:
            z[i] = 0.0
            continue
        z[i] = (values[i] - mean) / std
    return z

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

def print_dense_outliers(segments, residual_z, lengths, step_distances):
    print(f"\nTOP {TOP_FLAG_COUNT} DENSEST SEGMENTS")
    print("(semantic jump well above what this segment's length would predict, possibly under-explained)\n")
    eligible = [i for i in range(len(segments)) if not np.isnan(residual_z[i])]
    order = sorted(eligible, key=lambda i: residual_z[i], reverse=True)[:TOP_FLAG_COUNT]
    for index in order:
        print(f"segment {index:>5}  residual_z {residual_z[index]:.3f}  step_dist {step_distances[index]:.4f}  length {int(lengths[index])}")
        print(f"    {snippet(segments, index)}")

def print_inert_outliers(segments, residual_z, lengths, step_distances):
    print(f"\nTOP {TOP_FLAG_COUNT} MOST INERT SEGMENTS")
    print("(semantic jump well below what this segment's length would predict, possibly padding)\n")
    eligible = [i for i in range(len(segments)) if not np.isnan(residual_z[i])]
    order = sorted(eligible, key=lambda i: residual_z[i])[:TOP_FLAG_COUNT]
    for index in order:
        print(f"segment {index:>5}  residual_z {residual_z[index]:.3f}  step_dist {step_distances[index]:.4f}  length {int(lengths[index])}")
        print(f"    {snippet(segments, index)}")

def main():
    segments, embeddings = load_cache()
    if len(segments) < SMOOTHING_WINDOW + 2:
        print(f"Need at least {SMOOTHING_WINDOW + 2} segments for this analysis.")
        return
    embeddings_normalized = normalize_rows(embeddings)
    lengths = compute_segment_lengths(segments, LENGTH_UNIT)
    local_centroids = compute_local_centroids(embeddings_normalized, LOCAL_CENTROID_WINDOW)
    step_distances = compute_step_distances(embeddings_normalized, local_centroids)
    smoothed_pace = compute_smoothed_pace(step_distances, SMOOTHING_WINDOW)
    trend_poly = fit_length_trend(step_distances, lengths, MIN_LENGTH_FOR_RATIO, TREND_POLY_DEGREE)
    length_residuals = compute_length_residuals(step_distances, lengths, MIN_LENGTH_FOR_RATIO, trend_poly)
    residual_z = rolling_zscore_ignoring_nan(length_residuals, LOCAL_ZSCORE_WINDOW)
    print_pace_curve(step_distances, smoothed_pace)
    print_dense_outliers(segments, residual_z, lengths, step_distances)
    print_inert_outliers(segments, residual_z, lengths, step_distances)

if __name__ == "__main__":
    main()
