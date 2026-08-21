#!/usr/bin/env python3
"""Read pinned MSMARCO-XI Parquet metadata without downloading data row groups."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import DATASET_ID


def inspect_file(item: tuple[str, int, str]) -> dict:
    path, byte_size, revision = item
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    filesystem_path = f"datasets/{DATASET_ID}@{revision}/{path}"
    with HfFileSystem().open(filesystem_path, "rb") as handle:
        metadata = pq.ParquetFile(handle).metadata
    stem = Path(path).stem
    return {"path": path, "split": Path(path).parts[0], "file_size_bytes": byte_size,
            "rows": metadata.num_rows, "row_groups": metadata.num_row_groups,
            "language_file_code": stem[:-5] if stem.endswith("train") else stem[:-3],
            "schema": str(metadata.schema)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/repository-inventory.json"))
    args = parser.parse_args()
    if len(args.revision) != 40:
        raise SystemExit("revision must be a 40-character commit SHA")
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(DATASET_ID, revision=args.revision, files_metadata=True)
    jobs = [(item.rfilename, item.size, args.revision) for item in info.siblings if item.rfilename.endswith(".parquet")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        files = list(executor.map(inspect_file, jobs))
    files.sort(key=lambda row: row["path"])
    totals: dict[str, int] = {}
    for row in files:
        totals[row["split"]] = totals.get(row["split"], 0) + row["rows"]
    report = {"dataset_id": DATASET_ID, "revision": info.sha, "processing_timestamp": datetime.now(UTC).isoformat(),
              "method": "Hugging Face Hub file metadata + remote Parquet footer only; no data row groups downloaded",
              "files": files, "rows_by_split": totals, "total_rows": sum(totals.values()), "total_file_bytes": sum(row["file_size_bytes"] for row in files)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("dataset_id", "revision", "rows_by_split", "total_rows", "total_file_bytes")}, indent=2))


if __name__ == "__main__":
    main()
