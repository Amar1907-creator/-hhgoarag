"""Local FAISS HNSW index with deterministic persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissHNSWIndex:
    """Cosine-similarity ANN over already L2-normalized float32 vectors."""

    def __init__(self, dimension: int, *, m: int = 32, ef_search: int = 64) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("install faiss-cpu to use the ANN index") from exc
        self.faiss = faiss
        self.dimension = dimension
        self.ids: list[str] = []
        self.index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search

    def add(self, passage_ids: list[str], vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors must be float32 with the configured dimension")
        if len(passage_ids) != len(vectors) or len(set(passage_ids)) != len(passage_ids):
            raise ValueError("passage IDs must be unique and match vectors")
        self.index.add(vectors); self.ids.extend(passage_ids)

    def search(self, query_vectors: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        scores, positions = self.index.search(np.asarray(query_vectors, dtype=np.float32), top_k)
        return [[(self.ids[position], float(score)) for position, score in zip(row_positions, row_scores) if position >= 0] for row_positions, row_scores in zip(positions, scores)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "ids.json").write_text(json.dumps(self.ids) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "FaissHNSWIndex":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("install faiss-cpu to load the ANN index") from exc
        index = faiss.read_index(str(directory / "index.faiss")); result = cls.__new__(cls)
        result.faiss = faiss; result.index = index; result.dimension = index.d; result.ids = json.loads((directory / "ids.json").read_text())
        if index.ntotal != len(result.ids): raise ValueError("index/vector ID mapping is corrupt")
        return result
