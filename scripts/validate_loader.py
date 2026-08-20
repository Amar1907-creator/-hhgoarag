#!/usr/bin/env python3
"""Validate the pinned remote Parquet loader on a tiny bounded sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import DATASET_ID, load_sample
from src.data.schema import validate_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="dataset language config, e.g. hi")
    parser.add_argument("--revision", required=True, help="40-character dataset commit SHA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    records = load_sample(split=args.split, config=args.config, revision=args.revision, limit=args.limit)
    report = {
        "dataset_id": DATASET_ID,
        "revision": args.revision,
        "split": args.split,
        "config": args.config,
        "records_loaded": len(records),
        "records": [],
    }
    for index, record in enumerate(records, start=1):
        result = validate_record(record)
        report["records"].append(
            {
                "index": index,
                "valid": result.valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "query_id": record.get("query_id"),
                "target_lang": record.get("target_lang"),
                "query_preview": (record.get("query") or "")[:120],
                "passage_count": len(record.get("passages", {}).get("Translated_passages", [])),
            }
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not records or not all(row["valid"] for row in report["records"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
