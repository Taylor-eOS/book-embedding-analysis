import pysbd
from sentence_transformers import SentenceTransformer
import json

input_path = "input.txt"
report_path = "report.json"
model_name = "Qwen/Qwen3-Embedding-0.6B"
language = "en"
BATCH_SIZE = 16
SENTENCE_DISTANCE = 1
NUM_RESULTS = 10
REPORT_RESULTS = 200
PRINTED_CONTEXT = 80

def load_sentences():
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    segmenter = pysbd.Segmenter(language=language, clean=False)
    return [s.strip() for s in segmenter.segment(text) if s.strip()]

def embed_sentences(sentences):
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(sentences, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)

def compute_pairs(embeddings):
    similarity_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) < SENTENCE_DISTANCE:
                continue
            pairs.append((float(similarity_matrix[i, j]), i, j))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs

def print_top_pairs(pairs, sentences):
    print(f"\nTop {NUM_RESULTS} most similar sentence pairs (possible conceptual repetitions):\n")
    for score, i, j in pairs[:NUM_RESULTS]:
        print(f"{score:.4f}  [{i}] {sentences[i][:PRINTED_CONTEXT]}")
        print(f"          [{j}] {sentences[j][:PRINTED_CONTEXT]}\n")

def write_report(pairs, sentences):
    data = [
        {"score": round(score, 4), "sentence_a": sentences[i], "sentence_b": sentences[j]}
        for score, i, j in pairs[:REPORT_RESULTS]
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Report file created: {report_path}")

def main():
    sentences = load_sentences()
    if len(sentences) < 2:
        print("Need at least 2 sentences to compare.")
        return
    embeddings = embed_sentences(sentences)
    pairs = compute_pairs(embeddings)
    print_top_pairs(pairs, sentences)
    write_report(pairs, sentences)

if __name__ == "__main__":
    main()
