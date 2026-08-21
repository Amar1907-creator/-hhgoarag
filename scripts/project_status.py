#!/usr/bin/env python3
"""Assemble the project status report from artifacts that actually exist.

Every number is read from a manifest or counted from a file. Anything missing
is reported as missing rather than estimated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_evaluation import coverage, load_corpus_ids  # noqa: E402

MISSING = "not built"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def run_tests(root: Path) -> str:
    try:
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                                cwd=root, capture_output=True, text=True, timeout=900)
    except Exception as exc:
        return f"could not run ({exc})"
    tail = [line for line in result.stderr.strip().splitlines() if line.strip()]
    summary = next((line for line in reversed(tail) if line.startswith("Ran ")), "")
    verdict = "OK" if result.returncode == 0 else tail[-1] if tail else "FAILED"
    return f"{summary} -> {verdict}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="hi-train-5k")
    parser.add_argument("--eval-prefix", default=None)
    parser.add_argument("--output", type=Path, default=Path("docs/PROJECT_STATUS.md"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_prefix = args.eval_prefix or args.prefix.replace("-train-", "-validation-")
    processed, manifests = Path("data/processed"), Path("data/manifests")

    corpus = processed / f"{args.prefix}-corpus.jsonl"
    evaluation = processed / f"{eval_prefix}-evaluation.jsonl"
    index_dir = processed / f"{args.prefix}-index"
    build = read_json(manifests / f"{args.prefix}-build.json")
    bench = read_json(manifests / f"{args.prefix}-benchmark.json")
    eval_manifest = read_json(manifests / f"{eval_prefix}-evaluation-build.json")

    corpus_lines = count_lines(corpus)
    cover = coverage(evaluation, load_corpus_ids(corpus)) if (corpus_lines and evaluation.exists()) else {}
    metrics = bench.get("metrics", {})
    # Metrics computed against an evaluation set whose positives are mostly absent
    # from the corpus measure the corpus, not the retriever. Label them, loudly.
    coverage_ok = cover.get("query_coverage_pct", 0) >= 95.0 if cover else False
    invalid = "" if coverage_ok else "INVALID (low coverage) "
    total_latency = bench.get("warm_latency", {}).get("total_retrieval", {})
    index_bytes = (index_dir / "index.faiss").stat().st_size if (index_dir / "index.faiss").exists() else None
    tests = "skipped" if args.skip_tests else run_tests(root)

    def num(value, spec=",", fallback=MISSING):
        return format(value, spec) if isinstance(value, (int, float)) else fallback

    def latency_of(block, key):
        """Older benchmark manifests predate p95; report that rather than crash."""
        value = (block or {}).get(key)
        if isinstance(value, (int, float)):
            return f"{value:.2f} ms"
        return "not recorded (manifest predates this metric)" if block else MISSING

    rag_ready = (root / "src" / "rag" / "pipeline.py").exists()
    from src.rag.generator import RECOMMENDED_SMALL, installed_models
    local_models = installed_models()
    runtime = (f"local model via Ollama ({local_models[0]})" if local_models
               else f"no local model installed; extractive fallback in use "
                    f"(optional: ollama pull {RECOMMENDED_SMALL})")

    rows = [
        ("1. Dataset", f"ai4bharat/MSMARCO-XI @ {build.get('revision', MISSING)[:12]} "
                       f"({build.get('configuration', {}).get('loader_config', '?')}/"
                       f"{build.get('configuration', {}).get('split', '?')})"),
        ("2. Corpus size", f"{num(corpus_lines)} unique passages from "
                           f"{num(build.get('records_processed'))} records "
                           f"({build.get('duplicate_percentage', 0):.2f}% duplicates)"),
        ("3. Index size", f"{num(bench.get('corpus_passages'))} vectors, "
                          f"{index_bytes / 1e6:.1f} MB, FAISS HNSW inner-product"
                          if index_bytes else MISSING),
        ("4. Model", f"{bench.get('model', MISSING)} "
                     f"({num(bench.get('embedding_dimension'))} dimensions)"),
        ("5. Evaluation queries", num(cover.get("evaluation_queries"))),
        ("6. Positive coverage", f"{cover['positive_coverage_pct']:.2f}% of "
                                 f"{cover['total_positive_ids']:,} positive IDs; "
                                 f"{cover['query_coverage_pct']:.2f}% of queries"
                                 if cover else MISSING),
        ("7. Recall@5", f"{invalid}{metrics['recall_at_5']:.4f}" if "recall_at_5" in metrics else MISSING),
        ("8. Recall@10", f"{invalid}{metrics['recall_at_10']:.4f}" if "recall_at_10" in metrics else MISSING),
        ("9. MRR", f"{invalid}{metrics['mrr']:.4f}" if "mrr" in metrics else MISSING),
        ("10. p50 latency", latency_of(total_latency, "p50_ms")),
        ("11. p95 latency", latency_of(total_latency, "p95_ms")),
        ("12. Answer generation", f"{'implemented' if rag_ready else MISSING}; {runtime}; "
                                  f"requires no API key"),
        ("13. Tests", tests),
    ]

    limitations = [
        "The evaluation set only contains validation queries whose gold passage is byte-identical "
        "to a passage in the train corpus. That is a biased sample of queries, so Recall here is a "
        "pipeline health measure, not a claim about Hindi retrieval quality in general.",
        f"Corpus covers {num(build.get('records_processed'))} of 778,638 Hindi train records.",
        "Single language (Hindi). No BM25, fusion, or reranking; dense retrieval only.",
        "Voice input uses the browser's own hi-IN recognition, so it needs Chrome, "
        "Edge or Safari; typing works everywhere.",
        "Generated answers need a local Ollama model; without one the system quotes "
        "retrieved evidence verbatim, which is still fully grounded.",
    ]
    if not metrics:
        limitations.insert(0, "No benchmark has been run for this prefix yet.")
    elif not coverage_ok:
        limitations.insert(0, f"THE RETRIEVAL METRICS ABOVE ARE NOT VALID. Only "
                              f"{cover.get('query_coverage_pct', 0):.2f}% of evaluation queries have a "
                              f"positive passage inside the indexed corpus, so Recall and MRR here "
                              f"measure corpus coverage, not retrieval quality. Rebuild the evaluation "
                              f"set against this corpus before quoting any of these numbers.")

    demo = "./run.sh          # builds anything missing, then opens the app"
    lines = [f"# Project status: {args.prefix}", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    for name, value in rows:
        lines.append(f"- **{name}**: {value}")
    lines += ["", "## 14. Remaining limitations", ""]
    lines += [f"- {item}" for item in limitations]
    lines += ["", "## 15. Final demo command", "", "```bash", demo, "```", ""]
    if eval_manifest:
        lines += [f"Evaluation scan: {eval_manifest.get('evaluation_queries', 0)} queries from "
                  f"{eval_manifest.get('records_seen', 0):,} validation records "
                  f"(join rate {100 * eval_manifest.get('join_rate', 0):.3f}%).", ""]

    text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)
    print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()
