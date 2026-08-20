#!/usr/bin/env python3
"""Build or load a local dense index and benchmark a separately supplied evaluation set."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.embedding import SentenceTransformerE5
from src.retrieval.evaluation import evaluate
from src.retrieval.index import FaissHNSWIndex
from src.retrieval.metadata import load_metadata, resolve


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values); position = (len(ordered) - 1) * percent; low = int(position); high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timings(values: list[float]) -> dict[str, float]:
    return {"count": len(values), "min_ms": min(values) * 1e3, "mean_ms": statistics.fmean(values) * 1e3, "p50_ms": percentile(values, .50) * 1e3, "p70_ms": percentile(values, .70) * 1e3, "p100_ms": max(values) * 1e3}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True, help="Phase 1 train-only canonical corpus JSONL")
    parser.add_argument("--evaluation", type=Path, required=True, help="JSONL: query_id, query, positive_passage_ids")
    parser.add_argument("--index-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small"); parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=100); parser.add_argument("--warmup", type=int, default=10); parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(); embedder = SentenceTransformerE5(args.model)
    corpus = read_jsonl(args.corpus); metadata = load_metadata(args.corpus)
    if args.rebuild:
        index = FaissHNSWIndex(embedder.dimension)
        batch = 64
        for start in range(0, len(corpus), batch):
            rows = corpus[start:start + batch]; index.add([row["passage_id"] for row in rows], embedder.embed_passages([row["text"] for row in rows]))
        index.save(args.index_dir)
    else: index = FaissHNSWIndex.load(args.index_dir)
    queries = read_jsonl(args.evaluation)[:args.queries]
    if len(queries) < args.warmup: raise ValueError("evaluation set is smaller than warmup")
    for row in queries[:args.warmup]: index.search(embedder.embed_queries([row["query"]]), args.top_k)
    embedding_times: list[float] = []; search_times: list[float] = []; metadata_times: list[float] = []; total_times: list[float] = []; rankings: dict[str, list[str]] = {}; positives: dict[str, set[str]] = {}
    for row in queries:
        started = time.perf_counter(); point = time.perf_counter(); vector = embedder.embed_queries([row["query"]]); embedding_times.append(time.perf_counter() - point)
        point = time.perf_counter(); hits = index.search(vector, args.top_k)[0]; search_times.append(time.perf_counter() - point)
        point = time.perf_counter(); ranking = [passage_id for passage_id, _ in hits]; resolve(metadata, ranking); metadata_times.append(time.perf_counter() - point)
        total_times.append(time.perf_counter() - started); rankings[str(row["query_id"])] = ranking; positives[str(row["query_id"])] = set(row["positive_passage_ids"])
    report = {"model": args.model, "embedding_dimension": embedder.dimension, "corpus_passages": len(corpus), "evaluation_queries": len(queries), "top_k": args.top_k, "metrics": evaluate(rankings, positives), "warm_latency": {"query_embedding": timings(embedding_times), "ann_search": timings(search_times), "metadata_lookup": timings(metadata_times), "total_retrieval": timings(total_times)}, "index_bytes": (args.index_dir / "index.faiss").stat().st_size, "metadata_rows": len(metadata)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
