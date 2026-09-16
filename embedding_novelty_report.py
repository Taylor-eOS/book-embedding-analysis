import json
import numpy as np

cache_path = "embedding_cache.npz"
output_json_path = "novelty_timeline.json"
output_html_path = "novelty_timeline.html"
html_template_path = "novelty_timeline_template.html"
CONTEXT_WINDOW = 5
CONTEXT_RANK = 2
MIN_CONTEXT = 3
SNIPPET_CHARS = 80
NUMBER_OF_SPIKES = 20

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms

def context_basis(context_embeddings, top_k):
    mean = context_embeddings.mean(axis=0)
    centered = context_embeddings - mean
    if not np.any(centered):
        return mean, np.zeros((1, context_embeddings.shape[1]))
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = s.max() * 1e-8 if s.size > 0 else 0.0
    available_rank = int(np.sum(s > tolerance))
    rank = max(1, min(top_k, available_rank))
    return mean, vt[:rank]

def residual_fraction(vector, mean, basis):
    centered_vector = vector - mean
    denom = np.linalg.norm(centered_vector)
    if denom == 0:
        return 0.0
    projection = basis.T @ (basis @ centered_vector)
    residual = centered_vector - projection
    return float(np.linalg.norm(residual) / denom)

def compute_novelty_scores(embeddings_normalized, window, min_context, top_k):
    n = embeddings_normalized.shape[0]
    scores = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window)
        context = embeddings_normalized[start:i]
        if context.shape[0] < min_context:
            continue
        mean, basis = context_basis(context, top_k)
        scores[i] = residual_fraction(embeddings_normalized[i], mean, basis)
    return scores

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

def build_timeline_records(segments, novelty_raw, novelty_z):
    records = []
    for i in range(len(segments)):
        raw = novelty_raw[i]
        z = novelty_z[i]
        record = {
            "index": i,
            "raw": None if np.isnan(raw) else round(float(raw), 5),
            "z": None if np.isnan(raw) else round(float(z), 5),
            "snippet": snippet(segments, i),
        }
        records.append(record)
    return records

def build_payload(records, window, min_context):
    return {
        "context_window": window,
        "min_context": min_context,
        "segments": records,
    }

def save_json(payload):
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

def save_html(payload):
    with open(html_template_path, "r", encoding="utf-8") as f:
        template = f.read()
    embedded = json.dumps(payload, ensure_ascii=False)
    html = template.replace("__TIMELINE_DATA__", embedded)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)

def print_top_spikes(records, top_n=NUMBER_OF_SPIKES):
    eligible = [r for r in records if r["z"] is not None]
    ranked = sorted(eligible, key=lambda r: r["z"], reverse=True)[:top_n]
    print(f"\nTOP {len(ranked)} NOVELTY SPIKES")
    print("(segments least explained by the preceding context window)")
    print(f"Context rank: {CONTEXT_RANK}\n")
    for r in ranked:
        print(f"segment {r['index']:>5}  novelty_z {r['z']:.3f}  raw {r['raw']:.3f}")
        print(f"    {r['snippet']}")

def main():
    segments, embeddings = load_cache()
    if len(segments) < CONTEXT_WINDOW + MIN_CONTEXT:
        print(f"Need at least {CONTEXT_WINDOW + MIN_CONTEXT} segments given the configured window.")
        return
    embeddings_normalized = normalize_rows(embeddings)
    novelty_raw = compute_novelty_scores(embeddings_normalized, CONTEXT_WINDOW, MIN_CONTEXT, CONTEXT_RANK)
    novelty_z = zscore_ignoring_nan(novelty_raw)
    records = build_timeline_records(segments, novelty_raw, novelty_z)
    payload = build_payload(records, CONTEXT_WINDOW, MIN_CONTEXT)
    save_json(payload)
    save_html(payload)
    print_top_spikes(records)
    print(f"\nWrote {output_json_path} and {output_html_path}")

if __name__ == "__main__":
    main()
