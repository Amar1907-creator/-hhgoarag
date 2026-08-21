#!/usr/bin/env python3
"""Join an evaluation set against a corpus and report positive coverage.

An evaluation set whose positives are not in the indexed corpus cannot be
answered by any retriever, so its Recall and MRR are meaningless. This is the
check that catches that before a benchmark is believed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_corpus_ids(corpus: Path) -> set[str]:
    with corpus.open(encoding="utf-8") as handle:
        return {json.loads(line)["passage_id"] for line in handle if line.strip()}


def coverage(evaluation: Path, corpus_ids: set[str]) -> dict:
    with evaluation.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    positives = [pid for row in rows for pid in row["positive_passage_ids"]]
    present = [pid for pid in positives if pid in corpus_ids]
    matched = [row for row in rows if any(pid in corpus_ids for pid in row["positive_passage_ids"])]
    return {
        "corpus_passages": len(corpus_ids),
        "evaluation_queries": len(rows),
        "total_positive_ids": len(positives),
        "positive_ids_in_corpus": len(present),
        "queries_with_corpus_positive": len(matched),
        "positive_coverage_pct": (100 * len(present) / len(positives)) if positives else 0.0,
        "query_coverage_pct": (100 * len(matched) / len(rows)) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--min-query-coverage", type=float, default=95.0)
    parser.add_argument("--min-queries", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = coverage(args.evaluation, load_corpus_ids(args.corpus))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"corpus passages                      : {report['corpus_passages']:,}")
        print(f"total evaluation queries             : {report['evaluation_queries']:,}")
        print(f"total positive IDs                   : {report['total_positive_ids']:,}")
        print(f"positive IDs present in corpus       : {report['positive_ids_in_corpus']:,}")
        print(f"queries with >=1 corpus positive     : {report['queries_with_corpus_positive']:,}")
        print(f"positive coverage percentage         : {report['positive_coverage_pct']:.2f}%")
        print(f"query coverage percentage            : {report['query_coverage_pct']:.2f}%")

    problems = []
    if report["query_coverage_pct"] < args.min_query_coverage:
        problems.append(f"query coverage {report['query_coverage_pct']:.2f}% is below "
                        f"{args.min_query_coverage}%: the benchmark would measure the corpus, not the retriever")
    if report["evaluation_queries"] < args.min_queries:
        problems.append(f"only {report['evaluation_queries']} queries, fewer than the required {args.min_queries}")
    for problem in problems:
        print(f"PROBLEM: {problem}", file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
