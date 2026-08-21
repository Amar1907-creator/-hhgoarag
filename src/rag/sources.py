"""Knowledge sources: where retrieved evidence comes from.

The Hindi corpus and an uploaded document are different worlds -- different
scale, lifetime, index type and citation shape -- so they are separate objects
behind one small interface rather than one class with a mode flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from src.rag.evidence import DOCUMENT_MIN_SCORE


class KnowledgeSource(Protocol):
    key: str
    label: str
    kind: str
    min_score: float | None

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    def hydrate(self, ids: list[str]) -> dict[str, dict]: ...
    def size(self) -> int: ...


class CorpusSource:
    """The pre-built Hindi passage corpus."""

    kind = "corpus"
    min_score = None            # use the global default

    def __init__(self, index, texts, *, key: str = "corpus", label: str = "Hindi corpus") -> None:
        self.index = index
        self.texts = texts
        self.key = key
        self.label = label

    def search(self, vector, top_k):
        return self.index.search(vector, top_k)[0]

    def hydrate(self, ids):
        return {passage_id: {"text": text, "source_kind": "corpus"}
                for passage_id, text in self.texts.texts(ids).items()}

    def size(self):
        return getattr(self.index.index, "ntotal", 0)


class DocumentSource:
    """One uploaded document. Citations carry its name and page number."""

    kind = "document"

    def __init__(self, index, chunks: dict[str, dict], *, document_id: str, name: str) -> None:
        self.index = index
        self.chunks = chunks
        self.document_id = document_id
        self.name = name
        self.key = f"document:{document_id}"
        self.label = name
        self.min_score = DOCUMENT_MIN_SCORE

    @classmethod
    def load(cls, store, document_id: str, name: str | None = None) -> "DocumentSource":
        from src.documents.index import FlatIPIndex
        record = store.get(document_id)
        chunks = store.load_chunks(document_id)
        index = FlatIPIndex.load(Path(store.index_dir(document_id)))
        return cls(index, chunks, document_id=document_id,
                   name=name or (record.name if record else document_id))

    def search(self, vector, top_k):
        return self.index.search(vector, top_k)[0]

    def hydrate(self, ids):
        out: dict[str, dict] = {}
        for chunk_id in ids:
            row = self.chunks.get(chunk_id)
            if not row:
                continue
            out[chunk_id] = {"text": row["text"], "source_kind": "document",
                             "document_id": row["document_id"], "document": row["document_name"],
                             "page": row["page"], "chunk_id": row["chunk_id"]}
        return out

    def size(self):
        return len(self.chunks)
