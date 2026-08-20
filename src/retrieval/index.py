"""Local FAISS HNSW index with deterministic, crash-safe persistence."""

from __future__ import annotations

import json
import os
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
        """Write through temporary files so an interrupted checkpoint cannot
        leave a half-written index behind.

        ids.json is published BEFORE index.faiss on purpose. A hard kill between
        the two renames then leaves trailing ids whose vectors never landed,
        which load(repair=True) can truncate. The opposite order would leave
        vectors with no IDs, which is not repairable.
        """
        directory.mkdir(parents=True, exist_ok=True)
        temporary_index = directory / "index.faiss.tmp"
        temporary_ids = directory / "ids.json.tmp"
        self.faiss.write_index(self.index, str(temporary_index))
        temporary_ids.write_text(json.dumps(self.ids) + "\n")
        os.replace(temporary_ids, directory / "ids.json")
        os.replace(temporary_index, directory / "index.faiss")

    @classmethod
    def load(cls, directory: Path, *, repair: bool = False, log=None) -> "FaissHNSWIndex":
        """Load a persisted index.

        repair=True is for resuming a build that was killed hard (SIGKILL, OOM,
        power loss) between the two renames in save(). Trailing IDs with no
        vectors are dropped, and the affected passages are simply re-embedded by
        the resuming build. Benchmarking loads stay strict.
        """
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("install faiss-cpu to load the ANN index") from exc
        index = faiss.read_index(str(directory / "index.faiss")); result = cls.__new__(cls)
        result.faiss = faiss; result.index = index; result.dimension = index.d; result.ids = json.loads((directory / "ids.json").read_text())
        if index.ntotal != len(result.ids):
            if repair and len(result.ids) > index.ntotal:
                dropped = len(result.ids) - index.ntotal
                result.ids = result.ids[: index.ntotal]
                if log is not None:
                    print(f"[index] repaired checkpoint: dropped {dropped:,} trailing IDs whose "
                          f"vectors were never written; they will be re-embedded", file=log, flush=True)
            else:
                raise ValueError(
                    f"index/vector ID mapping is corrupt: {index.ntotal} vectors, {len(result.ids)} ids"
                )
        return result
