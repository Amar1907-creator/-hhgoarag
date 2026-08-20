#!/usr/bin/env python3
"""Build or load a local dense index and benchmark a separately supplied evaluation set.

Index construction streams the corpus once, reports progress, checkpoints on a
configurable interval, and can resume after an interruption. The number of
vectors already in the index is the single source of truth for how far a build
got, so a resumed build can neither skip nor duplicate a passage.
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.embedding import SentenceTransformerE5
from src.retrieval.evaluation import evaluate
from src.retrieval.index import FaissHNSWIndex
from src.retrieval.metadata import MemoryMetadataStore, OffsetMetadataStore

STATE_FILENAME = "build_state.json"


@contextmanager
def deferred_interrupt(log):
    """Hold SIGINT until the loop reaches a safe point.

    Without this, Ctrl-C lands wherever the interpreter happens to be: inside
    faiss's add (leaving vectors without IDs) or midway through writing a
    checkpoint. Deferring it means an interrupt is always handled between
    batches, where the on-disk state is known to be consistent.
    """
    requested = {"value": False}

    def handler(signum, frame):
        if not requested["value"]:
            print("\n[index] interrupt received; finishing the current batch, then checkpointing...",
                  file=log, flush=True)
        requested["value"] = True

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:
        previous = None  # not on the main thread, e.g. under a test runner
    try:
        yield requested
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values); position = (len(ordered) - 1) * percent; low = int(position); high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def timings(values: list[float]) -> dict[str, float]:
    return {"count": len(values), "min_ms": min(values) * 1e3, "mean_ms": statistics.fmean(values) * 1e3,
            "p50_ms": percentile(values, .50) * 1e3, "p70_ms": percentile(values, .70) * 1e3,
            "p95_ms": percentile(values, .95) * 1e3, "p100_ms": max(values) * 1e3}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def stream_corpus(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (passage_id, text) without holding the corpus in memory."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                yield row["passage_id"], row["text"]


def count_corpus_lines(path: Path) -> int:
    """Cheap byte scan so progress reporting can show a percentage and ETA."""
    total = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            total += block.count(b"\n")
    return total


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60: return f"{seconds}s"
    if seconds < 3600: return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def corpus_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size}


def read_state(index_dir: Path) -> dict | None:
    state_path = index_dir / STATE_FILENAME
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def write_state(index_dir: Path, state: dict) -> None:
    """Written after the index so it can never claim more than the index holds."""
    temporary = index_dir / (STATE_FILENAME + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(index_dir / STATE_FILENAME)


def checkpoint(index: FaissHNSWIndex, index_dir: Path, state: dict) -> None:
    index.save(index_dir)
    state = dict(state, passages_done=index.index.ntotal, updated=datetime.now(timezone.utc).isoformat())
    write_state(index_dir, state)


def build_index(*, corpus_path: Path, embedder, index_dir: Path, batch_size: int, checkpoint_every: int,
                progress_every: int, resume: bool, log=sys.stderr) -> dict:
    """Embed the corpus into a FAISS index, checkpointing as it goes."""
    fingerprint = corpus_fingerprint(corpus_path)
    state = {"model": embedder.model_name, "dimension": embedder.dimension, "corpus": fingerprint,
             "batch_size": batch_size, "passages_done": 0}
    resumed_from = 0

    if resume and (index_dir / "index.faiss").exists():
        index = FaissHNSWIndex.load(index_dir, repair=True, log=log)
        previous = read_state(index_dir)
        if previous:
            if previous.get("model") != embedder.model_name:
                raise SystemExit(f"cannot resume: index was built with model {previous.get('model')!r}, not {embedder.model_name!r}")
            if previous.get("corpus", {}).get("bytes") != fingerprint["bytes"]:
                raise SystemExit("cannot resume: the corpus file changed since the checkpoint")
        if index.dimension != embedder.dimension:
            raise SystemExit(f"cannot resume: index dimension {index.dimension} != model dimension {embedder.dimension}")
        resumed_from = index.index.ntotal
        print(f"[index] resuming from {resumed_from:,} passages already in the index", file=log, flush=True)
    else:
        index = FaissHNSWIndex(embedder.dimension)

    total_passages = count_corpus_lines(corpus_path)
    pending_ids: list[str] = []
    pending_texts: list[str] = []
    processed = 0
    batches = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal pending_ids, pending_texts, batches
        if not pending_ids:
            return
        index.add(pending_ids, embedder.embed_passages(pending_texts))
        pending_ids, pending_texts = [], []
        batches += 1

    def report_progress() -> None:
        done = index.index.ntotal
        rate = (done - resumed_from) / max(time.perf_counter() - started, 1e-9)
        remaining = max(total_passages - done, 0)
        eta = format_duration(remaining / rate) if rate > 0 else "unknown"
        percent = (100 * done / total_passages) if total_passages else 0
        print(f"[index] {done:,}/{total_passages:,} passages ({percent:.1f}%) "
              f"{rate:.0f} passages/s eta {eta}", file=log, flush=True)

    try:
        with deferred_interrupt(log) as interrupt_requested:
            for passage_id, text in stream_corpus(corpus_path):
                processed += 1
                if processed <= resumed_from:
                    # Verify the resume point instead of trusting the line count.
                    if processed == resumed_from and index.ids[resumed_from - 1] != passage_id:
                        raise SystemExit("cannot resume: corpus order does not match the checkpointed index")
                    continue
                pending_ids.append(passage_id)
                pending_texts.append(text)
                if len(pending_ids) >= batch_size:
                    flush()
                    if progress_every and batches % progress_every == 0:
                        report_progress()
                    if checkpoint_every and batches % checkpoint_every == 0:
                        checkpoint(index, index_dir, state)
                    # Safe point: on-disk and in-memory state agree here.
                    if interrupt_requested["value"]:
                        raise KeyboardInterrupt
            flush()
    except KeyboardInterrupt:
        # Deliberately do NOT flush here: the interrupt may have come from the
        # embedder mid-batch, and retrying it would raise again before the
        # checkpoint is written. The pending batch is simply recomputed on
        # resume, since index.ntotal is what resume trusts.
        checkpoint(index, index_dir, state)
        print(f"[index] interrupted; checkpointed {index.index.ntotal:,} passages.\n"
              f"[index] resume with the same command plus --resume", file=log, flush=True)
        raise

    checkpoint(index, index_dir, state)
    elapsed = time.perf_counter() - started
    embedded = index.index.ntotal - resumed_from
    print(f"[index] complete: {index.index.ntotal:,} passages in {format_duration(elapsed)} "
          f"({embedded / elapsed:.0f} passages/s this run)", file=log, flush=True)
    return {"index": index, "seconds": elapsed, "resumed_from": resumed_from,
            "passages_per_second": (embedded / elapsed) if elapsed else 0, "corpus_passages": index.index.ntotal}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True, help="Phase 1 train-only canonical corpus JSONL")
    parser.add_argument("--evaluation", type=Path, required=True, help="JSONL: query_id, query, positive_passage_ids")
    parser.add_argument("--index-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--device", default=None, help="torch device for the encoder, e.g. cpu or mps")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries", default="100", help="number of evaluation queries, or 'all'")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64, help="passages embedded per batch")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="save the index every N batches (0 disables)")
    parser.add_argument("--progress-every", type=int, default=20, help="report progress every N batches (0 disables)")
    parser.add_argument("--metadata", choices=("offset", "memory"), default="offset",
                        help="offset keeps only byte offsets in RAM; memory holds the whole sidecar")
    parser.add_argument("--rankings-out", type=Path, help="optional JSONL of per-query rankings for later comparison")
    parser.add_argument("--rebuild", action="store_true", help="build the index instead of loading it")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted --rebuild")
    parser.add_argument("--overwrite-index", action="store_true", help="allow --rebuild to replace an existing index")
    args = parser.parse_args()

    if args.rebuild and not args.resume and not args.overwrite_index and (args.index_dir / "index.faiss").exists():
        raise SystemExit(
            f"refusing to rebuild over the existing index at {args.index_dir}.\n"
            "Use --resume to continue it, --overwrite-index to replace it, or --index-dir NEW_DIR."
        )

    embedder = SentenceTransformerE5(args.model, device=args.device, batch_size=args.batch_size)

    build_report = None
    if args.rebuild:
        build_report = build_index(corpus_path=args.corpus, embedder=embedder, index_dir=args.index_dir,
                                   batch_size=args.batch_size, checkpoint_every=args.checkpoint_every,
                                   progress_every=args.progress_every, resume=args.resume)
        index = build_report["index"]
    else:
        index = FaissHNSWIndex.load(args.index_dir)

    store = OffsetMetadataStore.build(args.corpus) if args.metadata == "offset" else MemoryMetadataStore(args.corpus)

    queries = read_jsonl(args.evaluation)
    if args.queries != "all":
        queries = queries[: int(args.queries)]
    if len(queries) < args.warmup: raise ValueError("evaluation set is smaller than warmup")
    for row in queries[:args.warmup]: index.search(embedder.embed_queries([row["query"]]), args.top_k)

    embedding_times: list[float] = []; search_times: list[float] = []; metadata_times: list[float] = []; total_times: list[float] = []
    rankings: dict[str, list[str]] = {}; positives: dict[str, set[str]] = {}
    for row in queries:
        started = time.perf_counter(); point = time.perf_counter(); vector = embedder.embed_queries([row["query"]]); embedding_times.append(time.perf_counter() - point)
        point = time.perf_counter(); hits = index.search(vector, args.top_k)[0]; search_times.append(time.perf_counter() - point)
        point = time.perf_counter(); ranking = [passage_id for passage_id, _ in hits]; store.resolve(ranking); metadata_times.append(time.perf_counter() - point)
        total_times.append(time.perf_counter() - started); rankings[str(row["query_id"])] = ranking; positives[str(row["query_id"])] = set(row["positive_passage_ids"])

    if args.rankings_out:
        args.rankings_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rankings_out.open("w", encoding="utf-8") as handle:
            for query_id, ranking in rankings.items():
                handle.write(json.dumps({"query_id": query_id, "ranking": ranking,
                                         "positive_passage_ids": sorted(positives[query_id])}, ensure_ascii=False) + "\n")

    report = {"model": args.model, "device": args.device, "embedding_dimension": embedder.dimension,
              "corpus": str(args.corpus), "corpus_passages": index.index.ntotal, "evaluation": str(args.evaluation),
              "evaluation_queries": len(queries), "top_k": args.top_k, "batch_size": args.batch_size,
              "metadata_backend": store.backend, "metrics": evaluate(rankings, positives),
              "warm_latency": {"query_embedding": timings(embedding_times), "ann_search": timings(search_times),
                               "metadata_lookup": timings(metadata_times), "total_retrieval": timings(total_times)},
              "index_bytes": (args.index_dir / "index.faiss").stat().st_size, "metadata_rows": len(store),
              "timestamp": datetime.now(timezone.utc).isoformat()}
    if build_report:
        report["index_build"] = {"seconds": build_report["seconds"], "passages_per_second": build_report["passages_per_second"],
                                 "resumed_from": build_report["resumed_from"], "checkpoint_every_batches": args.checkpoint_every}
    store.close()
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
