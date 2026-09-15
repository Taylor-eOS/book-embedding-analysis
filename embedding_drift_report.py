import numpy as np

cache_path = "embedding_cache.npz"
TREND_WINDOWS = [3, 5, 8]
CONFIRM_WINDOW = 3
TOP_FLAG_COUNT = 25
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

def compute_step_vectors(embeddings_normalized):
    n = embeddings_normalized.shape[0]
    steps = np.zeros_like(embeddings_normalized)
    for i in range(1, n):
        step = embeddings_normalized[i] - embeddings_normalized[i - 1]
        steps[i] = normalize_vector(step)
    return steps

def compute_endorsement_scores_for_window(embeddings_normalized, steps, window):
    n = embeddings_normalized.shape[0]
    scores = np.full(n, np.nan)
    for i in range(2, n):
        window_start = max(0, i - window)
        if window_start >= i - 1:
            continue
        trend = normalize_vector(embeddings_normalized[i - 1] - embeddings_normalized[window_start])
        scores[i] = float(np.dot(trend, steps[i]))
    return scores

def compute_averaged_endorsement(embeddings_normalized, steps, windows):
    per_window = [compute_endorsement_scores_for_window(embeddings_normalized, steps, w) for w in windows]
    stacked = np.vstack(per_window)
    averaged = np.nanmean(stacked, axis=0)
    return averaged

def zscore_ignoring_nan(values):
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.zeros_like(values)
    mean = valid.mean()
    std = valid.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std

def compute_confirmation_scores(endorsement_scores, confirm_window):
    n = endorsement_scores.shape[0]
    confirmation = np.full(n, np.nan)
    for i in range(n):
        window_end = min(n, i + 1 + confirm_window)
        if window_end <= i + 1:
            continue
        following = endorsement_scores[i + 1:window_end]
        following = following[~np.isnan(following)]
        if following.size == 0:
            continue
        confirmation[i] = float(following.mean())
    return confirmation

def snippet(segments, index):
    text = segments[index].strip().replace("\n", " ")
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS] + "..."

def print_likely_cuts(segments, endorsement_z, confirmation_z, endorsement_raw, confirmation_raw):
    print(f"\nTOP {TOP_FLAG_COUNT} LIKELY CUT CANDIDATES")
    print("(unendorsed jump AND not confirmed by what follows, meaning nothing set it up and nothing continues it)\n")
    n = len(segments)
    eligible = [i for i in range(n) if not np.isnan(endorsement_z[i]) and not np.isnan(confirmation_z[i])]
    combined = {i: endorsement_z[i] + confirmation_z[i] for i in eligible}
    order = sorted(eligible, key=lambda i: combined[i])[:TOP_FLAG_COUNT]
    for index in order:
        print(f"segment {index:>5}  endorsement_z {endorsement_z[index]:.3f}  confirmation_z {confirmation_z[index]:.3f}  " f"raw_endorsement {endorsement_raw[index]:.3f}  raw_confirmation {confirmation_raw[index]:.3f}")
        print(f"    prev: {snippet(segments, index - 1)}")
        print(f"    this: {snippet(segments, index)}")

def print_unendorsed_only(segments, endorsement_z, endorsement_raw):
    print(f"\nTOP {TOP_FLAG_COUNT} UNENDORSED JUMPS (single-step view, before checking what follows)")
    print("(low or negative cosine similarity between the multi-window trend direction and the actual step)\n")
    n = len(segments)
    eligible = [i for i in range(n) if not np.isnan(endorsement_z[i])]
    order = sorted(eligible, key=lambda i: endorsement_z[i])[:TOP_FLAG_COUNT]
    for index in order:
        print(f"segment {index:>5}  endorsement_z {endorsement_z[index]:.3f}  raw {endorsement_raw[index]:.3f}")
        print(f"    prev: {snippet(segments, index - 1)}")
        print(f"    this: {snippet(segments, index)}")

def main():
    segments, embeddings = load_cache()
    min_required = max(TREND_WINDOWS) + CONFIRM_WINDOW + 2
    if len(segments) < min_required:
        print(f"Need at least {min_required} segments for this analysis given the configured windows.")
        return
    embeddings_normalized = normalize_rows(embeddings)
    steps = compute_step_vectors(embeddings_normalized)
    endorsement_raw = compute_averaged_endorsement(embeddings_normalized, steps, TREND_WINDOWS)
    endorsement_z = zscore_ignoring_nan(endorsement_raw)
    confirmation_raw = compute_confirmation_scores(endorsement_raw, CONFIRM_WINDOW)
    confirmation_z = zscore_ignoring_nan(confirmation_raw)
    print_likely_cuts(segments, endorsement_z, confirmation_z, endorsement_raw, confirmation_raw)
    print_unendorsed_only(segments, endorsement_z, endorsement_raw)

if __name__ == "__main__":
    main()
