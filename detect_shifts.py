import pysbd
import numpy as np
from sentence_transformers import SentenceTransformer

input_path = "input.txt"
model_name = "Qwen/Qwen3-Embedding-0.6B"
language = "en"
context_size = 3

def main():
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = [s.strip() for s in segmenter.segment(text) if s.strip()]
    if len(sentences) < context_size + 1:
        print(f"Need at least {context_size + 1} sentences to compare.")
        return
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
    scores = []
    indices = []
    for i in range(context_size, len(sentences)):
        context_embeddings = embeddings[i - context_size:i]
        context_mean = context_embeddings.mean(axis=0)
        context_mean = context_mean / np.linalg.norm(context_mean)
        similarity = float(embeddings[i] @ context_mean)
        scores.append(similarity)
        indices.append(i)
    order = np.argsort(scores)
    print(f"\nSentences ranked from most to least different than the previous {context_size} sentences:\n")
    for pos in order:
        idx = indices[pos]
        print(f"{scores[pos]:.4f} {sentences[idx][:40]}")
    most_different_pos = order[0]
    most_different_idx = indices[most_different_pos]
    print("\nMost different sentence from its preceding context:")
    print(sentences[most_different_idx])

if __name__ == "__main__":
    main()
