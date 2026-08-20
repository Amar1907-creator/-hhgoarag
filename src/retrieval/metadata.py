"""Metadata sidecar access; metadata is intentionally outside the FAISS index."""

from __future__ import annotations

import json
from pathlib import Path


def load_metadata(corpus_jsonl: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    with corpus_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            passage_id = row["passage_id"]
            if passage_id in metadata: raise ValueError(f"duplicate passage_id in corpus: {passage_id}")
            metadata[passage_id] = {key: value for key, value in row.items() if key != "text"}
    return metadata


def resolve(metadata: dict[str, dict], passage_ids: list[str]) -> list[dict]:
    return [metadata[passage_id] for passage_id in passage_ids if passage_id in metadata]
