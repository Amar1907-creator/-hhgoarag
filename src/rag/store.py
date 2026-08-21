"""Offset-indexed passage text lookup for the answer layer.

The retrieval benchmark deliberately keeps text out of its metadata sidecar.
Generation needs the text, so this store keeps one byte offset per passage and
reads only the handful of rows that were actually retrieved.
"""

from __future__ import annotations

import json
from pathlib import Path


class PassageTextStore:
    def __init__(self, corpus_jsonl: Path, offsets: dict[str, int]) -> None:
        self.corpus_jsonl = corpus_jsonl
        self.offsets = offsets
        self.handle = corpus_jsonl.open("rb")

    @classmethod
    def build(cls, corpus_jsonl: Path) -> "PassageTextStore":
        offsets: dict[str, int] = {}
        with corpus_jsonl.open("rb") as handle:
            offset = 0
            for line in handle:
                stripped = line.strip()
                if stripped:
                    offsets[json.loads(stripped)["passage_id"]] = offset
                offset += len(line)
        return cls(corpus_jsonl, offsets)

    def texts(self, passage_ids: list[str]) -> dict[str, str]:
        found: dict[str, str] = {}
        for passage_id in passage_ids:
            offset = self.offsets.get(passage_id)
            if offset is None:
                continue
            self.handle.seek(offset)
            found[passage_id] = json.loads(self.handle.readline().decode("utf-8"))["text"]
        return found

    def close(self) -> None:
        self.handle.close()

    def __len__(self) -> int:
        return len(self.offsets)
