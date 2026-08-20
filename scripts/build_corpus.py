#!/usr/bin/env python3
"""Build an exact-deduplicated corpus and separate provenance/evaluation artifact.

A build is FRESH by default: output files are truncated and the SQLite dedup
state is reset so the two always agree. Reusing stale dedup state against a
truncated corpus silently yields an empty corpus, so that combination is only
reachable through the explicit --append multi-part mode.
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
from src.data.deduplicate import ExactDeduplicator, passage_id
from src.data.loader import DATASET_ID, load_split
from src.data.normalize import normalize_text
from src.data.schema import validate_record

NORMALIZATION = "nfkc-collapse-whitespace-v1"
DEDUPLICATION = "exact-language-normalized-text-sha256-v1"


def output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    """All artifact paths for a build, derived from one prefix."""
    return {
        "corpus": output_dir / f"{prefix}-corpus.jsonl",
        "provenance": output_dir / f"{prefix}-provenance.jsonl",
        "errors": output_dir / f"{prefix}-validation-errors.jsonl",
        "seen": output_dir / f"{prefix}-seen.sqlite3",
    }


def existing_outputs(paths: dict[str, Path]) -> list[Path]:
    return [path for path in paths.values() if path.exists()]


def assert_safe_to_write(paths: dict[str, Path], *, overwrite: bool, append: bool) -> None:
    """Refuse to clobber artifacts unless the caller said so explicitly."""
    if overwrite or append:
        return
    clashes = existing_outputs(paths)
    if clashes:
        listing = "\n  ".join(str(path) for path in clashes)
        raise SystemExit(
            "refusing to overwrite existing build artifacts:\n  "
            f"{listing}\n"
            "Use --output-prefix NEW_PREFIX for a new build, --overwrite to replace "
            "these files, or --append to extend a multi-part build."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="dataset language config, e.g. hi")
    parser.add_argument("--revision", required=True, help="40-character dataset commit SHA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, help="optional cap on source records processed")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output-prefix",
        help="artifact filename prefix; defaults to {config}-{split}. Use a distinct "
        "prefix to build a new corpus without touching an existing one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing existing artifacts with a fresh build",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="advanced: continue a multi-part build. Appends to existing artifacts and "
        "KEEPS the dedup state instead of resetting it.",
    )
    args = parser.parse_args()

    prefix = args.output_prefix or f"{args.config}-{args.split}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir, prefix)
    assert_safe_to_write(paths, overwrite=args.overwrite, append=args.append)

    file_mode = "a" if args.append else "w"
    deduper = ExactDeduplicator(paths["seen"], reset=not args.append)
    seen_at_start = deduper.count()
    stats: Counter = Counter()
    record_number = 0
    started = time.perf_counter()

    with paths["corpus"].open(file_mode, encoding="utf-8") as corpus, \
         paths["provenance"].open(file_mode, encoding="utf-8") as provenance, \
         paths["errors"].open(file_mode, encoding="utf-8") as errors:
        for record_number, record in enumerate(
            load_split(split=args.split, config=args.config, revision=args.revision), start=1
        ):
            result = validate_record(record)
            if not result.valid:
                stats["malformed_records"] += 1
                errors.write(json.dumps({
                    "record_number": record_number,
                    "query_id": record.get("query_id") if isinstance(record, dict) else None,
                    "errors": result.errors,
                    "warnings": result.warnings,
                }, ensure_ascii=False) + "\n")
                continue

            language = record["target_lang"]
            for position, text in enumerate(record["passages"]["Translated_passages"]):
                normalized = normalize_text(text)
                if not normalized:
                    stats["empty_passages"] += 1
                    continue
                stable_id = passage_id(language, normalized)
                stats["total_passages"] += 1
                if deduper.add(stable_id):
                    stats["unique_passages"] += 1
                    corpus.write(json.dumps({
                        "passage_id": stable_id,
                        "text": text,
                        "language": language,
                        "source_id": str(record["query_id"]),
                        "metadata": {"normalization": NORMALIZATION},
                    }, ensure_ascii=False) + "\n")
                else:
                    stats["duplicate_passages"] += 1
                provenance.write(json.dumps({
                    "passage_id": stable_id,
                    "query_id": record["query_id"],
                    "candidate_position": position,
                    "is_selected": int(record["passages"]["is_selected"][position]),
                    "split": args.split,
                }, ensure_ascii=False) + "\n")

            if args.limit and record_number >= args.limit:
                break

    seen_at_end = deduper.count()
    deduper.close()
    elapsed = time.perf_counter() - started
    total = stats["total_passages"]

    manifest = {
        "dataset_id": DATASET_ID,
        "revision": args.revision,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "loader_config": args.config,
            "split": args.split,
            "limit": args.limit,
            "output_prefix": prefix,
            "mode": "append" if args.append else "fresh",
            "dedup_reset": not args.append,
            "normalization": NORMALIZATION,
            "deduplication": DEDUPLICATION,
        },
        "records_processed": record_number,
        **stats,
        "duplicate_percentage": (100 * stats["duplicate_passages"] / total) if total else 0,
        "dedup_rows_at_start": seen_at_start,
        "dedup_rows_at_end": seen_at_end,
        "elapsed_seconds": elapsed,
        "records_per_second": (record_number / elapsed) if elapsed else 0,
        "corpus": str(paths["corpus"]),
        "provenance": str(paths["provenance"]),
        "validation_errors": str(paths["errors"]),
        "dedup_state": str(paths["seen"]),
    }
    manifest_path = Path("data/manifests") / f"{prefix}-build.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
