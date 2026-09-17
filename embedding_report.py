import os
import numpy as np
from embedding_cache import get_segments_and_embeddings, input_path, SPLIT_ON

TOP_JUMP_COUNT = 30
TOP_BOUNDARY_NEIGHBORS = 2
CHAPTER_MARKER = SPLIT_ON + " "

def detect_chapter_boundaries(total_segments):
    if not os.path.exists(input_path):
        print(f"\nNo input file found at '{input_path}'.")
        print("Proceeding without existing chapter boundaries.\n")
        return []
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    raw_pieces = text.split(SPLIT_ON)
    boundaries = []
    segment_count = 0
    for i, piece in enumerate(raw_pieces[:-1]):
        if not piece.strip():
            continue
        segment_count += 1
        next_piece = raw_pieces[i + 1]
        if next_piece.startswith(" ") and next_piece.strip():
            boundaries.append(segment_count)
    return curate_chapter_boundaries(boundaries, total_segments)

def curate_chapter_boundaries(boundaries, total_segments):
    curated = []
    seen = set()
    previous = 0
    for b in boundaries:
        if b in seen:
            raise ValueError(f"duplicate chapter boundary at segment {b}")
        if b <= previous:
            raise ValueError(f"chapter boundary at segment {b} is not strictly " f"after the previous boundary at {previous}")
        if b >= total_segments:
            raise ValueError(f"chapter boundary at segment {b} falls at or beyond " f"the last segment ({total_segments - 1})")
        seen.add(b)
        curated.append(b)
        previous = b
    print(f"\ndetected {len(curated)} exact chapter boundaries from " f"'{input_path}' via the '{CHAPTER_MARKER!r}' marker:")
    print(", ".join(str(b) for b in curated))
    return curated

def compute_adjacent_distances(embeddings):
    return np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)

def percentile_rank(value, population):
    return float((population < value).sum()) / len(population) * 100.0

def print_boundary_neighborhood(segments, diffs, boundaries):
    if not boundaries:
        return
    print("\nLOCAL GAP DIAGNOSTICS")
    total_segments = len(segments)
    for b in boundaries:
        print(f"\nknown boundary {b}")
        start = max(1, b - TOP_BOUNDARY_NEIGHBORS)
        end = min(total_segments - 1, b + TOP_BOUNDARY_NEIGHBORS)
        for candidate in range(start, end + 1):
            distance = diffs[candidate - 1]
            marker = "  <-- CHAPTER BOUNDARY" if candidate == b else ""
            print(f"  gap {candidate:>4}: " f"segments {candidate - 1} | {candidate}  " f"distance {distance:.4f}{marker}")

def print_top_jumps(segments, diffs, existing_boundaries):
    print(f"\ntop {TOP_JUMP_COUNT} largest exact adjacent-segment jumps " "in the book:")
    print("(boundary is exactly before the listed segment)\n")
    order = np.argsort(diffs)[::-1][:TOP_JUMP_COUNT]
    for idx in sorted(order):
        boundary = idx + 1
        distance = diffs[idx]
        pct = percentile_rank(distance, diffs)
        if boundary in existing_boundaries:
            status = "EXISTING CHAPTER BOUNDARY"
        else:
            status = "not an existing chapter boundary"
        before_snippet = segments[boundary - 1][-50:].replace("\n", " ")
        after_snippet = segments[boundary][:50].replace("\n", " ")
        print(f"boundary {boundary:>4}  " f"distance {distance:.4f}  " f"(top {100 - pct:.1f}%)  " f"[{status}]")
        print(f"    ...{before_snippet} || {after_snippet}...")

def print_existing_boundary_ranking(segments, diffs, boundaries):
    if not boundaries:
        return
    ranked = []
    for b in boundaries:
        distance = diffs[b - 1]
        pct = percentile_rank(distance, diffs)
        ranked.append((b, distance, pct))
    ranked.sort(key=lambda x: x[1], reverse=True)
    print("\nEXISTING CHAPTER BOUNDARIES RANKED BY THEIR EXACT GAP:")
    for b, distance, pct in ranked:
        before_snippet = segments[b - 1][-50:].replace("\n", " ")
        after_snippet = segments[b][:50].replace("\n", " ")
        print(f"\nboundary {b:>4}  " f"distance {distance:.4f}  " f"(top {100 - pct:.1f}% of all gaps)")
        print(f"    ...{before_snippet} || {after_snippet}...")

def main():
    segments, embeddings = get_segments_and_embeddings()
    if len(segments) < 2:
        print("Need at least 2 segments to analyze meaning shifts.")
        return
    existing_boundaries = detect_chapter_boundaries(len(segments))
    diffs = compute_adjacent_distances(embeddings)
    print(f"\nadjacent-segment distances over {len(diffs)} exact gaps:")
    print(f"min {diffs.min():.4f}  " f"max {diffs.max():.4f}  " f"mean {diffs.mean():.4f}  " f"std {diffs.std():.4f}")
    print_boundary_neighborhood(segments, diffs, existing_boundaries,)
    print_existing_boundary_ranking(segments, diffs, existing_boundaries,)
    print_top_jumps(segments, diffs, existing_boundaries,)

if __name__ == "__main__":
    print("Did you export the Hugging Face token?")
    main()

