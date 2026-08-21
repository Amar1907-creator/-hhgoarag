#!/usr/bin/env python3
"""Derive validation evaluation JSONL rows from train corpus passage IDs.

Artifact names come from --output-prefix, so an evaluation set built against a
smaller or larger corpus cannot overwrite an existing one. Both the JSONL and
its manifest follow the prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.deduplicate import passage_id
from src.data.loader import DATASET_ID, load_split
from src.data.normalize import normalize_text
from src.data.schema import validate_record


def output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "evaluation": output_dir / f"{prefix}-evaluation.jsonl",
        "manifest": Path("data/manifests") / f"{prefix}-evaluation-build.json",
    }


def assert_safe_to_write(paths: dict[str, Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    clashes = [path for path in paths.values() if path.exists()]
    if clashes:
        listing = "\n  ".join(str(path) for path in clashes)
        raise SystemExit(
            "refusing to overwrite existing evaluation artifacts:\n  "
            f"{listing}\n"
            "Use --output-prefix NEW_PREFIX for a new evaluation set, or --overwrite to replace these."
        )


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
    parser.add_argument("--corpus", type=Path, required=True, help="train-only canonical corpus JSONL")
    parser.add_argument("--config", required=True, help="dataset language config, e.g. hi")
    parser.add_argument("--revision", required=True, help="40-character dataset commit SHA")
    parser.add_argument("--limit", type=int, help="optional cap on emitted evaluation queries")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-prefix", help="artifact prefix; defaults to {config}-validation")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing evaluation artifacts")
    args = parser.parse_args()

    prefix = args.output_prefix or f"{args.config}-validation"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir, prefix)
    assert_safe_to_write(paths, overwrite=args.overwrite)

    corpus_ids = load_corpus_passage_ids(args.corpus)
    stats: Counter = Counter()
    started = time.perf_counter()
    emitted = 0

    with paths["evaluation"].open("w", encoding="utf-8") as handle:
        for record_number, record in enumerate(
            load_split(split="validation", config=args.config, revision=args.revision), start=1
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

            handle.write(json.dumps({
                "query_id": record["query_id"],
                "query": record["query"],
                "positive_passage_ids": positive_passage_ids,
            }, ensure_ascii=False) + "\n")
            stats["evaluation_queries"] += 1
            emitted += 1
            if args.limit and emitted >= args.limit:
                break

    elapsed = time.perf_counter() - started
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": args.revision,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "loader_config": args.config,
            "split": "validation",
            "limit": args.limit,
            "output_prefix": prefix,
            "corpus": str(args.corpus),
            "corpus_passages": len(corpus_ids),
            "normalization": "nfkc-collapse-whitespace-v1",
            "positive_join": "validation is_selected labels intersected with train corpus passage_id set",
        },
        "records_seen": stats["records_seen"],
        **stats,
        "elapsed_seconds": elapsed,
        "evaluation": str(paths["evaluation"]),
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
