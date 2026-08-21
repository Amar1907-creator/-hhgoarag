#!/usr/bin/env python3
"""Stream a bounded dataset sample and write factual inspection measurements."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import DATASET_ID, load_split
from src.data.normalize import normalize_text
from src.data.schema import validate_record


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="dataset language config, e.g. hi")
    parser.add_argument("--revision", required=True, help="40-character dataset commit SHA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/inspection.json"))
    args = parser.parse_args()
    started = time.perf_counter(); validation = Counter(); languages = Counter(); lengths: list[int] = []
    for count, record in enumerate(load_split(split=args.split, config=args.config, revision=args.revision), start=1):
        result = validate_record(record)
        validation["valid" if result.valid else "malformed"] += 1
        validation["warnings"] += len(result.warnings)
        if result.valid:
            languages[record["target_lang"]] += 1
            for text in record["passages"]["Translated_passages"]:
                if text.strip(): lengths.append(len(normalize_text(text)))
        if count >= args.limit: break
    elapsed = time.perf_counter() - started
    report = {"dataset_id": DATASET_ID, "revision": args.revision, "processing_timestamp": datetime.now(UTC).isoformat(), "config": {"loader_config": args.config, "split": args.split, "limit": args.limit, "normalization": "nfkc-collapse-whitespace-v1"},
              "sample_records": count if 'count' in locals() else 0, "elapsed_seconds": elapsed,
              "records_per_second": (count / elapsed) if 'count' in locals() and elapsed else 0,
              "validation": dict(validation), "target_languages": dict(languages),
              "passage_character_lengths": {"count": len(lengths), "min": min(lengths, default=0), "mean": sum(lengths) / len(lengths) if lengths else 0,
              "p50": percentile(lengths, .50), "p75": percentile(lengths, .75), "p90": percentile(lengths, .90), "p95": percentile(lengths, .95), "p99": percentile(lengths, .99), "max": max(lengths, default=0)},
              "measurement_method": "Unicode NFKC + collapsed whitespace character count; bounded streaming sample"}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
