"""Embedding interface and optional SentenceTransformers implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dimension: int

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerE5:
    """Multilingual E5 adapter; document/query prefixes are part of its contract."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install sentence-transformers to use a production embedder") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _embed(self, prefix: str, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode([prefix + text for text in texts], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("query: ", texts)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("passage: ", texts)
