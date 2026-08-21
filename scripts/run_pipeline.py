#!/usr/bin/env python3
"""One command: corpus -> evaluation -> index -> benchmark, verified between stages.

Each stage is skipped when its artifacts already exist, so an interrupted run is
resumed by re-running the same command. Nothing under an existing prefix is
overwritten: every stage writes under --prefix, and the underlying scripts
refuse to clobber artifacts they did not create.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_evaluation import coverage, load_corpus_ids  # noqa: E402

REVISION = "bf5cdc1f26e581e519018e434db14edd1b77602b"
ROOT = Path(__file__).resolve().parents[1]


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def run(argv: list[str]) -> float:
    """Run a stage with live output. Raises SystemExit on failure."""
    print(f"$ {' '.join(argv)}\n", flush=True)
    started = time.perf_counter()
    result = subprocess.run(argv, cwd=ROOT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(f"stage failed with exit code {result.returncode}: {' '.join(argv)}")
    return elapsed


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hi")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--limit", type=int, default=5000, help="train records to ingest")
    parser.add_argument("--prefix", default=None, help="artifact prefix; default hi-train-{limit//1000}k")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--device", default=None, help="encoder device, e.g. cpu or mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-queries", type=int, default=40,
                        help="stop if the evaluation set is smaller than this")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run stages whose artifacts exist")
    args = parser.parse_args()

    thousands = args.limit // 1000
    prefix = args.prefix or f"{args.config}-train-{thousands}k" if thousands else f"{args.config}-train-{args.limit}"
    eval_prefix = prefix.replace("-train-", "-validation-")
    processed = Path("data/processed")
    manifests = Path("data/manifests")

    corpus = processed / f"{prefix}-corpus.jsonl"
    evaluation = processed / f"{eval_prefix}-evaluation.jsonl"
    index_dir = processed / f"{prefix}-index"
    benchmark = manifests / f"{prefix}-benchmark.json"
    rankings = processed / f"{prefix}-rankings.jsonl"

    timings: dict[str, float] = {}
    started_all = time.perf_counter()

    if not args.skip_preflight:
        banner("STAGE 0  preflight")
        timings["preflight"] = run([sys.executable, "scripts/preflight.py",
                                    "--config", args.config, "--revision", args.revision, "--limit", "5"])

    banner(f"STAGE 1  corpus  ({args.limit:,} train records -> {corpus})")
    if corpus.exists() and not args.force:
        print(f"already built ({count_lines(corpus):,} passages); skipping")
    else:
        cmd = [sys.executable, "scripts/build_corpus.py", "--config", args.config,
               "--revision", args.revision, "--limit", str(args.limit), "--output-prefix", prefix]
        if args.force:
            cmd.append("--overwrite")
        timings["corpus"] = run(cmd)

    corpus_manifest = read_json(manifests / f"{prefix}-build.json")
    corpus_lines = count_lines(corpus)
    errors_path = processed / f"{prefix}-validation-errors.jsonl"
    print(f"\nverify: {corpus_lines:,} corpus lines")
    print(f"verify: manifest unique_passages = {corpus_manifest.get('unique_passages', 'n/a'):,}"
          if isinstance(corpus_manifest.get("unique_passages"), int) else "verify: manifest missing")
    if corpus_manifest.get("unique_passages") not in (None, corpus_lines):
        raise SystemExit("corpus line count does not match the build manifest")
    print(f"verify: duplicate rate {corpus_manifest.get('duplicate_percentage', 0):.2f}%")
    print(f"verify: validation errors {count_lines(errors_path) if errors_path.exists() else 0}")
    print(f"verify: dedup rows {corpus_manifest.get('dedup_rows_at_end', 'n/a')}")

    banner(f"STAGE 2  validation evaluation  (-> {evaluation})")
    if evaluation.exists() and not args.force:
        print(f"already built ({count_lines(evaluation):,} queries); skipping")
    else:
        cmd = [sys.executable, "scripts/build_evaluation.py", "--corpus", str(corpus),
               "--config", args.config, "--revision", args.revision, "--output-prefix", eval_prefix]
        if args.force:
            cmd.append("--overwrite")
        timings["evaluation"] = run(cmd)

    banner("STAGE 3  coverage gate")
    report = coverage(evaluation, load_corpus_ids(corpus))
    for key, value in report.items():
        print(f"  {key:<32} {value:,.2f}" if isinstance(value, float) else f"  {key:<32} {value:,}")
    if report["evaluation_queries"] < args.min_queries:
        suggested = args.limit * 3
        raise SystemExit(
            f"\nSTOP: only {report['evaluation_queries']} evaluation queries, below --min-queries "
            f"{args.min_queries}. A benchmark on this few queries has confidence intervals wider "
            f"than any model difference worth measuring.\nBuild a larger corpus and try again:\n"
            f"  python3 scripts/run_pipeline.py --limit {suggested}\n"
            f"(the {prefix} artifacts are kept; the larger run uses its own prefix)")
    if report["query_coverage_pct"] < 99.0:
        raise SystemExit(f"\nSTOP: query coverage {report['query_coverage_pct']:.2f}% -- the evaluation "
                         f"builder should only emit queries whose positives are in the corpus. "
                         f"Corpus and evaluation are out of sync.")
    print(f"\ngate passed: {report['evaluation_queries']} queries, "
          f"{report['query_coverage_pct']:.1f}% query coverage")

    banner(f"STAGE 4  index + benchmark  (-> {index_dir})")
    if benchmark.exists() and index_dir.joinpath("index.faiss").exists() and not args.force:
        print("already built; skipping")
    else:
        cmd = [sys.executable, "scripts/benchmark_retrieval.py", "--corpus", str(corpus),
               "--evaluation", str(evaluation), "--index-dir", str(index_dir),
               "--output", str(benchmark), "--rankings-out", str(rankings),
               "--model", args.model, "--queries", "all", "--warmup", "10",
               "--batch-size", str(args.batch_size), "--top-k", str(args.top_k),
               "--checkpoint-every", "50", "--progress-every", "20"]
        if args.device:
            cmd += ["--device", args.device]
        if index_dir.joinpath("index.faiss").exists():
            cmd.append("--resume" if not args.force else "--overwrite-index")
        cmd.append("--rebuild")
        timings["index_and_benchmark"] = run(cmd)

    banner("STAGE 5  verification")
    result = read_json(benchmark)
    ids = json.loads((index_dir / "ids.json").read_text())
    checks = {
        "index ntotal == corpus passages": result.get("corpus_passages") == corpus_lines,
        "ids count == corpus passages": len(ids) == corpus_lines,
        "embedding dimension": result.get("embedding_dimension"),
        "evaluation queries benchmarked": result.get("evaluation_queries"),
    }
    for name, value in checks.items():
        print(f"  {name:<36} {value}")
    if not checks["index ntotal == corpus passages"] or not checks["ids count == corpus passages"]:
        raise SystemExit("index and corpus are misaligned")

    metrics = result.get("metrics", {})
    latency = result.get("warm_latency", {}).get("total_retrieval", {})
    print(f"\n  Recall@1  {metrics.get('recall_at_1', 0):.4f}")
    print(f"  Recall@5  {metrics.get('recall_at_5', 0):.4f}")
    print(f"  Recall@10 {metrics.get('recall_at_10', 0):.4f}")
    print(f"  MRR       {metrics.get('mrr', 0):.4f}")
    print(f"  latency   p50 {latency.get('p50_ms', 0):.2f} ms   p95 {latency.get('p95_ms', 0):.2f} ms   "
          f"p100 {latency.get('p100_ms', 0):.2f} ms")

    timings["total"] = time.perf_counter() - started_all
    run_manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "config": args.config,
        "revision": args.revision, "limit": args.limit, "prefix": prefix,
        "corpus": str(corpus), "evaluation": str(evaluation), "index_dir": str(index_dir),
        "benchmark": str(benchmark), "coverage": report, "metrics": metrics,
        "stage_seconds": {name: round(value, 1) for name, value in timings.items()},
    }
    path = manifests / f"{prefix}-pipeline.json"
    path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    banner(f"PIPELINE COMPLETE in {timings['total'] / 60:.1f} min -> {path}")
    print("Next: python3 scripts/project_status.py --prefix " + prefix)
    print("Demo: python3 scripts/demo.py --corpus " + str(corpus) + " --index-dir " + str(index_dir))


if __name__ == "__main__":
    main()
