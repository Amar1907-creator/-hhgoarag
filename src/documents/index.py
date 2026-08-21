"""Per-document vector index.

Exact inner-product search, not HNSW: a document is hundreds of chunks, not a
million, so an approximate index would add tuning risk and graph overhead to buy
microseconds nobody can measure. Deliberately separate from the Hindi corpus
index -- different lifetime, different scale, different failure modes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


class FlatIPIndex:
    """Cosine similarity over L2-normalised float32 vectors, exhaustively."""

    def __init__(self, dimension: int) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("install faiss-cpu to index documents") from exc
        self.faiss = faiss
        self.dimension = dimension
        self.ids: list[str] = []
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors must be float32 with the configured dimension")
        if len(chunk_ids) != len(vectors) or len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk IDs must be unique and match vectors")
        self.index.add(vectors)
        self.ids.extend(chunk_ids)

    def search(self, query_vectors: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]
        scores, positions = self.index.search(np.asarray(query_vectors, dtype=np.float32),
                                              min(top_k, self.index.ntotal))
        return [[(self.ids[position], float(score))
                 for position, score in zip(row_positions, row_scores) if position >= 0]
                for row_positions, row_scores in zip(positions, scores)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "index.faiss.tmp"
        self.faiss.write_index(self.index, str(temporary))
        (directory / "ids.json").write_text(json.dumps(self.ids))
        os.replace(temporary, directory / "index.faiss")

    @classmethod
    def load(cls, directory: Path) -> "FlatIPIndex":
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install faiss-cpu to index documents") from exc
        index = faiss.read_index(str(directory / "index.faiss"))
        result = cls.__new__(cls)
        result.faiss = faiss
        result.index = index
        result.dimension = index.d
        result.ids = json.loads((directory / "ids.json").read_text())
        if index.ntotal != len(result.ids):
            raise ValueError(f"document index is corrupt: {index.ntotal} vectors, {len(result.ids)} ids")
        return result
