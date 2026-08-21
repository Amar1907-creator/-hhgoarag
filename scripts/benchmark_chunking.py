#!/usr/bin/env python3
"""Compare chunking strategies on the real corpus, with the real encoder.

The task specification asks for thought about how the dataset is split, not one
scheme asserted to be best. This re-chunks the same passages six ways, embeds
each with the same model, indexes each identically, and evaluates all of them
against the same queries -- so the strategy is chosen by measurement.

A retrieved chunk is credited to the passage it came from, which is what makes
strategies producing different numbers of chunks comparable at all.

    python3 scripts/benchmark_chunking.py \
        --corpus data/processed/hi-train-5k-corpus.jsonl \
        --evaluation data/processed/hi-validation-5k-evaluation.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.strategies import DESCRIPTIONS, STRATEGIES, chunk  # noqa: E402
from src.documents.index import FlatIPIndex  # noqa: E402

DEFAULT_STRATEGIES = ("whole", "fixed", "sentence", "sliding", "semantic", "metadata")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(rankings: dict[str, list[str]], positives: dict[str, set[str]]) -> dict:
    cutoffs = (1, 5, 10)
    result = {"queries": len(rankings)}
    for cutoff in cutoffs:
        result[f"recall_at_{cutoff}"] = sum(
            bool(set(rankings[q][:cutoff]) & positives[q]) for q in rankings) / max(len(rankings), 1)
    reciprocal = []
    for query, ranked in rankings.items():
        rank = next((i for i, pid in enumerate(ranked, 1) if pid in positives[query]), None)
        reciprocal.append(0.0 if rank is None else 1.0 / rank)
    result["mrr"] = sum(reciprocal) / max(len(reciprocal), 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=6000,
                        help="passages to include besides every gold passage")
    parser.add_argument("--queries", type=int, default=0, help="0 uses every evaluation query")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target", type=int, default=700)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docs/CHUNKING_BENCHMARK.md"))
    args = parser.parse_args()

    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    for name in names:
        if name not in STRATEGIES:
            raise SystemExit(f"unknown strategy {name!r}; available: {', '.join(STRATEGIES)}")

    corpus = load_jsonl(args.corpus)
    queries = load_jsonl(args.evaluation)
    if args.queries:
        queries = queries[: args.queries]
    gold = {pid for row in queries for pid in row["positive_passage_ids"]}

    # Every gold passage must be present or the comparison is meaningless; the
    # rest is a head sample to keep the run affordable.
    by_id = {row["passage_id"]: row for row in corpus}
    sample = [by_id[pid] for pid in gold if pid in by_id]
    for row in corpus:
        if len(sample) >= args.sample + len(gold):
            break
        if row["passage_id"] not in gold:
            sample.append(row)
    print(f"{len(sample):,} passages ({len(gold)} of them gold), {len(queries)} queries, "
          f"{len(names)} strategies\n", flush=True)

    from src.retrieval.embedding import SentenceTransformerE5
    embedder = SentenceTransformerE5(args.model, device=args.device, batch_size=args.batch_size)
    query_vectors = embedder.embed_queries([row["query"] for row in queries])
    positives = {str(row["query_id"]): set(row["positive_passage_ids"]) for row in queries}

    results = []
    for name in names:
        started = time.perf_counter()
        texts, owners = [], []
        for row in sample:
            options = {"target": args.target}
            if name == "metadata":
                # Structure the corpus actually carries: language is the only
                # section-like field on an MSMARCO passage.
                options["heading"] = row.get("language", "")
            for piece in chunk(row["text"], name, **options):
                texts.append(piece.text)
                owners.append(row["passage_id"])
        chunked = time.perf_counter() - started

        started = time.perf_counter()
        vectors = []
        for start in range(0, len(texts), args.batch_size):
            vectors.append(embedder.embed_passages(texts[start:start + args.batch_size]))
        import numpy as np
        matrix = np.vstack(vectors).astype(np.float32)
        embedded = time.perf_counter() - started

        index = FlatIPIndex(embedder.dimension)
        index.add([f"c{i}" for i in range(len(texts))], matrix)

        rankings, searches = {}, []
        for row, vector in zip(queries, query_vectors):
            began = time.perf_counter()
            hits = index.search(vector.reshape(1, -1), args.top_k * 3)[0]
            searches.append((time.perf_counter() - began) * 1e3)
            seen, ranked = set(), []
            for chunk_id, _ in hits:                     # credit the parent passage
                owner = owners[int(chunk_id[1:])]
                if owner not in seen:
                    seen.add(owner)
                    ranked.append(owner)
                if len(ranked) >= args.top_k:
                    break
            rankings[str(row["query_id"])] = ranked

        metrics = evaluate(rankings, positives)
        results.append({
            "strategy": name, "description": DESCRIPTIONS[name], "chunks": len(texts),
            "chunks_per_passage": len(texts) / len(sample),
            "avg_chars": statistics.fmean(len(t) for t in texts) if texts else 0,
            "index_mb": matrix.nbytes / 1e6, "chunk_seconds": chunked, "embed_seconds": embedded,
            "search_p50_ms": percentile(searches, .50), "search_p95_ms": percentile(searches, .95),
            **metrics})
        print(f"  {name:<9} {len(texts):>7,} chunks  R@5 {metrics['recall_at_5']:.4f}  "
              f"R@10 {metrics['recall_at_10']:.4f}  MRR {metrics['mrr']:.4f}  "
              f"embed {embedded:.0f}s", flush=True)

    best = max(results, key=lambda r: r["recall_at_10"])
    lines = ["# Chunking strategy benchmark", "",
             f"{len(sample):,} passages including all {len(gold)} gold passages, "
             f"{len(queries)} queries, `{args.model}`. Every strategy uses the same passages, "
             "the same encoder and the same index type; a retrieved chunk is credited to the "
             "passage it came from, which is what makes different chunk counts comparable.", "",
             "| Strategy | Chunks | per passage | avg chars | Recall@1 | Recall@5 | Recall@10 | MRR "
             "| Vectors MB | Embed s | Search p50 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in results:
        lines.append(
            f"| `{row['strategy']}` | {row['chunks']:,} | {row['chunks_per_passage']:.2f} "
            f"| {row['avg_chars']:.0f} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} "
            f"| {row['recall_at_10']:.4f} | {row['mrr']:.4f} | {row['index_mb']:.1f} "
            f"| {row['embed_seconds']:.0f} | {row['search_p50_ms']:.2f} ms |")
    lines += ["", "## What each strategy is", ""]
    for row in results:
        lines.append(f"- **`{row['strategy']}`** — {row['description']}")
    lines += ["", f"**Best Recall@10: `{best['strategy']}` at {best['recall_at_10']:.4f}.**", "",
              "Read the cost column alongside the quality one: a strategy that wins by a hair "
              "while tripling the vector count is not obviously the right choice for a latency "
              "budget, and the table is here so that trade is visible rather than assumed.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
