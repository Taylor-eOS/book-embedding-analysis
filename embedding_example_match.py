import os
import numpy as np
from sentence_transformers import SentenceTransformer

cache_path = "embedding_cache.npz"
examples_path = "examples.txt"
model_name = "google/embeddinggemma-300m"
SPLIT_ON = "\n\n"
TRUST_CODE = False
TOKEN_VAR = os.environ.get("HF_TOKEN")
BATCH_SIZE = 16
AGGREGATION = "max"
TOP_FLAG_COUNT = 40
SNIPPET_CHARS = 100

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def load_example_segments():
    with open(examples_path, "r", encoding="utf-8") as f:
        text = f.read()
    segments = [s.strip() for s in text.split(SPLIT_ON) if s.strip()]
    return segments

def normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms

def embed_segments(segments, model):
    embeddings = model.encode(
        segments,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )
    return embeddings

def load_model():
    hf_token = TOKEN_VAR
    model = SentenceTransformer(model_name, device="cpu", trust_remote_code=TRUST_CODE)
    return model

def compute_similarity_matrix(segment_embeddings, example_embeddings):
    segments_normalized = normalize_rows(segment_embeddings)
    examples_normalized = normalize_rows(example_embeddings)
    return segments_normalized @ examples_normalized.T

def aggregate_scores(similarity_matrix, aggregation):
    if aggregation == "max":
        return similarity_matrix.max(axis=1)
    if aggregation == "mean":
        return similarity_matrix.mean(axis=1)
    raise ValueError(f"Unknown aggregation mode: {aggregation}")

def best_example_indices(similarity_matrix):
    return np.argmax(similarity_matrix, axis=1)

def snippet(text, chars=SNIPPET_CHARS):
    cleaned = str(text).strip().replace("\n", " ")
    if len(cleaned) <= chars:
        return cleaned
    return cleaned[:chars] + "..."

def print_top_matches(segments, examples, scores, best_indices, mean_scores, aggregation, top_n):
    print(f"\nTOP {top_n} SEGMENTS MOST SIMILAR TO THE EXAMPLES (aggregation: {aggregation})\n")
    order = np.argsort(scores)[::-1][:top_n]
    for rank, index in enumerate(order, start=1):
        closest_example = best_indices[index]
        print(f"{rank:>3}. segment {index:>5}  score {scores[index]:.4f}  mean_score {mean_scores[index]:.4f}  closest_example {closest_example}")
        print(f"     segment: {snippet(segments[index])}")
        print(f"     example: {snippet(examples[closest_example])}")

def main():
    segments, segment_embeddings = load_cache()
    examples = load_example_segments()
    if len(examples) == 0:
        print(f"No segments found in {examples_path}.")
        return
    model = load_model()
    example_embeddings = embed_segments(examples, model)
    similarity_matrix = compute_similarity_matrix(segment_embeddings, example_embeddings)
    max_scores = aggregate_scores(similarity_matrix, "max")
    mean_scores = aggregate_scores(similarity_matrix, "mean")
    scores = max_scores if AGGREGATION == "max" else mean_scores
    best_indices = best_example_indices(similarity_matrix)
    print(f"{len(segments)} candidate segments, {len(examples)} example segments")
    print_top_matches(segments, examples, scores, best_indices, mean_scores, AGGREGATION, TOP_FLAG_COUNT)

if __name__ == "__main__":
    main()
