#!/usr/bin/env python3
"""Derive validation evaluation JSONL rows from train corpus passage IDs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.deduplicate import passage_id
from src.data.loader import DATASET_ID, load_split
from src.data.normalize import normalize_text
from src.data.schema import validate_record


def load_corpus_passage_ids(corpus_path: Path) -> set[str]:
    passage_ids: set[str] = set()
    with corpus_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            passage_ids.add(row["passage_id"])
    return passage_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True, help="Phase 1 train-only canonical corpus JSONL")
    parser.add_argument("--config", required=True, help="dataset language config, e.g. hi")
    parser.add_argument("--revision", required=True, help="40-character dataset commit SHA")
    parser.add_argument("--limit", type=int, help="optional cap on emitted evaluation queries")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/{config}-validation-evaluation.jsonl"),
        help="output JSONL path; {config} is expanded from --config",
    )
    args = parser.parse_args()
    args.output = Path(str(args.output).format(config=args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    corpus_ids = load_corpus_passage_ids(args.corpus)
    stats = Counter()
    started = time.perf_counter()
    emitted = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for record_number, record in enumerate(
            load_split(split="validation", config=args.config, revision=args.revision),
            start=1,
        ):
            stats["records_seen"] += 1
            result = validate_record(record)
            if not result.valid:
                stats["malformed_records"] += 1
                continue

            language = record["target_lang"]
            positive_passage_ids: list[str] = []
            for position, text in enumerate(record["passages"]["Translated_passages"]):
                if not record["passages"]["is_selected"][position]:
                    continue
                normalized = normalize_text(text)
                if not normalized:
                    stats["empty_selected_passages"] += 1
                    continue
                stable_id = passage_id(language, normalized)
                if stable_id in corpus_ids:
                    positive_passage_ids.append(stable_id)
                else:
                    stats["selected_not_in_train_corpus"] += 1

            if not positive_passage_ids:
                stats["queries_without_corpus_positive"] += 1
                continue

            handle.write(
                json.dumps(
                    {
                        "query_id": record["query_id"],
                        "query": record["query"],
                        "positive_passage_ids": positive_passage_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["evaluation_queries"] += 1
            emitted += 1
            if args.limit and emitted >= args.limit:
                break

    elapsed = time.perf_counter() - started
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": args.revision,
        "processing_timestamp": datetime.now(UTC).isoformat(),
        "configuration": {
            "loader_config": args.config,
            "split": "validation",
            "limit": args.limit,
            "corpus": str(args.corpus),
            "normalization": "nfkc-collapse-whitespace-v1",
            "positive_join": "validation is_selected labels intersected with train corpus passage_id set",
        },
        "records_seen": stats["records_seen"],
        **stats,
        "elapsed_seconds": elapsed,
        "evaluation": str(args.output),
    }
    manifest_path = Path("data/manifests") / f"{args.config}-validation-evaluation-build.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
