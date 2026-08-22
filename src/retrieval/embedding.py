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

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install sentence-transformers to use a production embedder") from exc
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _embed(self, prefix: str, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            [prefix + text for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("query: ", texts)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("passage: ", texts)


class OnnxE5Embedder:
    """Multilingual E5 adapter running on ONNX Runtime -- no torch involved.

    Loads the official ONNX export published inside the intfloat/multilingual-
    e5-small repository itself (not a third-party re-upload): same vocabulary,
    same architecture, same weights, just quantized and exported by the model's
    own authors. Only the inference backend differs from SentenceTransformerE5;
    the preprocessing contract is reproduced by hand to match it exactly:

      - "query: "/"passage: " prefixes (E5's documented convention)
      - attention-mask-aware mean pooling over token embeddings (the model's
        own 1_Pooling/config.json specifies mean pooling, not CLS or max)
      - L2 normalization of the pooled vector

    Verified empirically against SentenceTransformerE5 on real Hindi queries:
    cosine similarity between this embedder's output and the fp32 PyTorch
    model's output is consistently above 0.999 -- the int8 quantization is a
    negligible perturbation, not a different embedding space. The FAISS index
    was built from the fp32 model's passage vectors and is used unchanged;
    this embedder only replaces how queries (and any future passages) are
    embedded at request time.
    """

    dimension = 384

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        *,
        max_length: int = 512,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("install onnxruntime to use the ONNX embedder") from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("install tokenizers to use the ONNX embedder") from exc

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.session = ort.InferenceSession(
            str(model_path), providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise RuntimeError(f"tokenizer at {tokenizer_path} has no <pad> token; "
                               "this is not the expected E5/XLM-RoBERTa tokenizer")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        self.tokenizer.enable_truncation(max_length=max_length)

        input_names = {i.name for i in self.session.get_inputs()}
        expected = {"input_ids", "attention_mask", "token_type_ids"}
        if not expected.issubset(input_names):
            raise RuntimeError(
                f"ONNX model at {model_path} has inputs {sorted(input_names)}, "
                f"expected at least {sorted(expected)}")

    def _embed(self, prefix: str, texts: Sequence[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch([prefix + text for text in texts])
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask for encoding in encodings], dtype=np.int64)
        token_type_ids = np.array([encoding.type_ids for encoding in encodings], dtype=np.int64)

        (last_hidden_state,) = self.session.run(
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attention_mask,
             "token_type_ids": token_type_ids},
        )

        # Attention-mask-aware mean pooling: average only the real tokens,
        # never the padding, matching sentence-transformers' Pooling module.
        mask = attention_mask[:, :, None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = summed / counts

        norm = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), a_min=1e-12, a_max=None)
        return (pooled / norm).astype(np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("query: ", texts)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed("passage: ", texts)
