#!/usr/bin/env python3
"""Build an exact-deduplicated corpus and separate provenance/evaluation artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.deduplicate import ExactDeduplicator, passage_id
from src.data.loader import DATASET_ID, load_split
from src.data.normalize import normalize_text
from src.data.schema import validate_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--revision", required=True); parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int); parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / f"{args.config}-{args.split}-corpus.jsonl"; provenance_path = args.output_dir / f"{args.config}-{args.split}-provenance.jsonl"; errors_path = args.output_dir / f"{args.config}-{args.split}-validation-errors.jsonl"
    deduper = ExactDeduplicator(args.output_dir / f"{args.config}-{args.split}-seen.sqlite3"); stats = Counter(); started = time.perf_counter()
    with corpus_path.open("w", encoding="utf-8") as corpus, provenance_path.open("w", encoding="utf-8") as provenance, errors_path.open("w", encoding="utf-8") as errors:
        for record_number, record in enumerate(load_split(split=args.split, config=args.config, revision=args.revision), start=1):
            result = validate_record(record)
            if not result.valid:
                stats["malformed_records"] += 1; errors.write(json.dumps({"record_number": record_number, "query_id": record.get("query_id") if isinstance(record, dict) else None, "errors": result.errors, "warnings": result.warnings}, ensure_ascii=False) + "\n"); continue
            language = record["target_lang"]
            for position, text in enumerate(record["passages"]["Translated_passages"]):
                normalized = normalize_text(text)
                if not normalized:
                    stats["empty_passages"] += 1; continue
                stable_id = passage_id(language, normalized); stats["total_passages"] += 1
                is_unique = deduper.add(stable_id)
                if is_unique:
                    stats["unique_passages"] += 1
                    corpus.write(json.dumps({"passage_id": stable_id, "text": text, "language": language, "source_id": str(record["query_id"]), "metadata": {"normalization": "nfkc-collapse-whitespace-v1"}}, ensure_ascii=False) + "\n")
                else: stats["duplicate_passages"] += 1
                provenance.write(json.dumps({"passage_id": stable_id, "query_id": record["query_id"], "candidate_position": position, "is_selected": int(record["passages"]["is_selected"][position]), "split": args.split}, ensure_ascii=False) + "\n")
            if args.limit and record_number >= args.limit: break
    deduper.close(); elapsed = time.perf_counter() - started; total = stats["total_passages"]
    manifest = {"dataset_id": DATASET_ID, "revision": args.revision, "processing_timestamp": datetime.now(UTC).isoformat(), "configuration": {"loader_config": args.config, "split": args.split, "limit": args.limit, "normalization": "nfkc-collapse-whitespace-v1", "deduplication": "exact-language-normalized-text-sha256-v1"}, "records_processed": record_number if 'record_number' in locals() else 0, **stats, "duplicate_percentage": (100 * stats["duplicate_passages"] / total) if total else 0, "elapsed_seconds": elapsed, "records_per_second": (record_number / elapsed) if 'record_number' in locals() and elapsed else 0, "corpus": str(corpus_path), "provenance": str(provenance_path), "validation_errors": str(errors_path)}
    manifest_path = Path("data/manifests") / f"{args.config}-{args.split}-build.json"; manifest_path.parent.mkdir(parents=True, exist_ok=True); manifest_path.write_text(json.dumps(manifest, indent=2) + "\n"); print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
