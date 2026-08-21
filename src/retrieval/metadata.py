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


class MemoryMetadataStore:
    """Whole sidecar in RAM. Fine for a pilot corpus, roughly 1 GB at 1M passages."""

    backend = "memory"

    def __init__(self, corpus_jsonl: Path) -> None:
        self.metadata = load_metadata(corpus_jsonl)

    def resolve(self, passage_ids: list[str]) -> list[dict]:
        return resolve(self.metadata, passage_ids)

    def close(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self.metadata)


class OffsetMetadataStore:
    """Byte-offset index into the corpus; only the k retrieved rows are parsed.

    Holds one offset per passage instead of one dict per passage, which keeps a
    million-passage corpus in the low hundreds of MB rather than about a GB.
    """

    backend = "offset"

    def __init__(self, corpus_jsonl: Path, offsets: dict[str, int]) -> None:
        self.corpus_jsonl = corpus_jsonl
        self.offsets = offsets
        self.handle = corpus_jsonl.open("rb")

    @classmethod
    def build(cls, corpus_jsonl: Path) -> "OffsetMetadataStore":
        offsets: dict[str, int] = {}
        with corpus_jsonl.open("rb") as handle:
            offset = 0
            for line in handle:
                stripped = line.strip()
                if stripped:
                    passage_id = json.loads(stripped)["passage_id"]
                    if passage_id in offsets:
                        raise ValueError(f"duplicate passage_id in corpus: {passage_id}")
                    offsets[passage_id] = offset
                offset += len(line)
        return cls(corpus_jsonl, offsets)

    def resolve(self, passage_ids: list[str]) -> list[dict]:
        rows: list[dict] = []
        for passage_id in passage_ids:
            offset = self.offsets.get(passage_id)
            if offset is None:
                continue
            self.handle.seek(offset)
            row = json.loads(self.handle.readline().decode("utf-8"))
            rows.append({key: value for key, value in row.items() if key != "text"})
        return rows

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self) -> "OffsetMetadataStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.offsets)
