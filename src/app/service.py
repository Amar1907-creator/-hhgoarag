"""Application state: expensive things are loaded exactly once, at startup.

The embedding model, the FAISS index and the passage store are all process
lifetime objects. Nothing here is constructed per request.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from src.documents.ingest import ingest_pdf
from src.documents.store import STATUS_FAILED, STATUS_READY, DocumentRecord, DocumentStore
from src.rag.generator import build_generator
from src.rag.pipeline import RagPipeline
from src.rag.sources import DocumentSource

DEFAULT_PREFIX = os.environ.get("HHGOARAG_PREFIX", "hi-train-5k")
DEFAULT_EMBEDDING_MODEL = os.environ.get("HHGOARAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
PROCESSED = Path("data/processed")
MANIFESTS = Path("data/manifests")


def resolve_device(requested: str | None = None) -> str:
    """Prefer Apple Silicon acceleration when it is actually usable."""
    if requested:
        return requested
    override = os.environ.get("HHGOARAG_DEVICE")
    if override:
        return override
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def discover_prefix(preferred: str = DEFAULT_PREFIX) -> str | None:
    """Use the requested corpus if built, else the largest one that is."""
    if (PROCESSED / f"{preferred}-corpus.jsonl").exists():
        return preferred
    candidates = []
    for path in sorted(PROCESSED.glob("*-corpus.jsonl")):
        prefix = path.name[: -len("-corpus.jsonl")]
        if (PROCESSED / f"{prefix}-index" / "index.faiss").exists():
            candidates.append((path.stat().st_size, prefix))
    return max(candidates)[1] if candidates else None


@dataclass
class ServiceStatus:
    ready: bool = False
    prefix: str = ""
    corpus_passages: int = 0
    index_vectors: int = 0
    embedding_model: str = ""
    embedding_dimension: int = 0
    device: str = ""
    generator: str = ""
    generator_available: bool = False
    generator_detail: dict = field(default_factory=dict)
    requires_api_key: bool = False
    load_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)
    error: str = ""
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["uptime_seconds"] = round(time.time() - self.started_at, 1)
        payload.pop("started_at")
        return payload


class Service:
    """Holds the loaded pipeline. Thread-safe for the single-process server."""

    def __init__(self, *, prefix: str | None = None, generator_kind: str = "auto",
                 device: str | None = None, top_k: int = 10, min_score: float | None = None,
                 pipeline: RagPipeline | None = None, status: ServiceStatus | None = None,
                 documents: DocumentStore | None = None) -> None:
        self.top_k = top_k
        self.min_score = min_score
        self.generator_kind = generator_kind
        self.requested_prefix = prefix or DEFAULT_PREFIX
        self.requested_device = device
        self.pipeline = pipeline
        self.status = status or ServiceStatus()
        self._lock = threading.Lock()
        self.documents = documents or DocumentStore()
        # One worker: ingestion is CPU-bound embedding work, and queueing uploads
        # is better than thrashing the machine the retrieval index is serving from.
        self._ingest_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._pending: dict[str, DocumentRecord] = {}

    # -- loading ---------------------------------------------------------
    def load(self) -> ServiceStatus:
        started = time.perf_counter()
        prefix = discover_prefix(self.requested_prefix)
        if prefix is None:
            self.status.error = (
                f"no built corpus found under {PROCESSED}/. Build one first:\n"
                f"  python3 scripts/run_pipeline.py --limit 5000")
            return self.status

        corpus = PROCESSED / f"{prefix}-corpus.jsonl"
        index_dir = PROCESSED / f"{prefix}-index"
        device = resolve_device(self.requested_device)
        generator = build_generator(self.generator_kind)

        try:
            self.pipeline = RagPipeline.load(corpus=corpus, index_dir=index_dir,
                                             model=DEFAULT_EMBEDDING_MODEL, device=device,
                                             generator=generator, top_k=self.top_k,
                                             min_score=self.min_score)
        except Exception as exc:
            self.status.error = f"{type(exc).__name__}: {exc}"
            return self.status

        with corpus.open("rb") as handle:
            corpus_passages = sum(1 for line in handle if line.strip())
        underlying = getattr(generator, "primary", generator)

        self.status = ServiceStatus(
            ready=True, prefix=prefix, corpus_passages=corpus_passages,
            index_vectors=self.pipeline.index.index.ntotal,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            embedding_dimension=self.pipeline.embedder.dimension, device=device,
            generator=getattr(generator, "name", "unknown"),
            generator_available=getattr(underlying, "available", True),
            generator_detail=underlying.status() if hasattr(underlying, "status") else {},
            requires_api_key=False,
            load_seconds=round(time.perf_counter() - started, 2),
            metrics=self.read_metrics(prefix),
        )
        self.restore_documents()
        if self.status.index_vectors != corpus_passages:
            self.status.ready = False
            self.status.error = (f"index/corpus mismatch: {self.status.index_vectors} vectors "
                                 f"vs {corpus_passages} passages")
        return self.status

    # -- uploaded documents ----------------------------------------------
    def restore_documents(self) -> int:
        """Re-attach documents indexed in an earlier run, so uploads survive restarts."""
        restored = 0
        if self.pipeline is None:
            return 0
        for record in self.documents.list():
            if record.status != STATUS_READY:
                continue
            try:
                source = DocumentSource.load(self.documents, record.document_id, record.name)
            except Exception:
                continue
            self.pipeline.add_source(source)
            restored += 1
        return restored

    def ingest_document(self, data: bytes, name: str) -> DocumentRecord:
        """Start ingestion in the background and return the initial record."""
        if self.pipeline is None:
            raise RuntimeError(self.status.error or "service is not loaded")
        from src.documents.extract import content_id
        document_id = content_id(data)
        existing = self.documents.get(document_id)
        if existing and existing.status == STATUS_READY:
            self.attach_document(document_id)
            return existing

        record = DocumentRecord(document_id=document_id, name=name, bytes=len(data))
        self.documents.save_record(record)
        self._pending[document_id] = record
        self._ingest_pool.submit(self._run_ingest, data, name, document_id)
        return record

    def _run_ingest(self, data: bytes, name: str, document_id: str) -> None:
        try:
            record = ingest_pdf(data, name, embedder=self.pipeline.embedder, store=self.documents)
            if record.status == STATUS_READY:
                self.attach_document(document_id)
        except Exception as exc:  # a bad upload must never take the server down
            record = self.documents.get(document_id) or DocumentRecord(document_id=document_id, name=name)
            record.status, record.reason = STATUS_FAILED, "ingest_error"
            record.message = f"{type(exc).__name__}: {exc}"
            self.documents.save_record(record)
        finally:
            self._pending.pop(document_id, None)

    def attach_document(self, document_id: str) -> bool:
        record = self.documents.get(document_id)
        if not record or record.status != STATUS_READY or self.pipeline is None:
            return False
        try:
            self.pipeline.add_source(DocumentSource.load(self.documents, document_id, record.name))
        except Exception:
            return False
        return True

    def delete_document(self, document_id: str) -> bool:
        if self.pipeline is not None:
            self.pipeline.remove_source(f"document:{document_id}")
        return self.documents.delete(document_id)

    def knowledge_sources(self) -> list[dict]:
        if self.pipeline is None:
            return []
        return self.pipeline.describe_sources()

    @staticmethod
    def read_metrics(prefix: str) -> dict:
        """Benchmark headline numbers, suppressed when coverage makes them invalid."""
        path = MANIFESTS / f"{prefix}-benchmark.json"
        pipeline_path = MANIFESTS / f"{prefix}-pipeline.json"
        if not path.exists():
            return {}
        try:
            bench = json.loads(path.read_text())
        except Exception:
            return {}
        coverage = {}
        if pipeline_path.exists():
            try:
                coverage = json.loads(pipeline_path.read_text()).get("coverage", {})
            except Exception:
                coverage = {}
        valid = coverage.get("query_coverage_pct", 0) >= 95.0 if coverage else None
        latency = bench.get("warm_latency", {}).get("total_retrieval", {})
        return {
            "recall_at_1": bench.get("metrics", {}).get("recall_at_1"),
            "recall_at_5": bench.get("metrics", {}).get("recall_at_5"),
            "recall_at_10": bench.get("metrics", {}).get("recall_at_10"),
            "mrr": bench.get("metrics", {}).get("mrr"),
            "evaluation_queries": bench.get("evaluation_queries"),
            "query_coverage_pct": coverage.get("query_coverage_pct"),
            "metrics_valid": valid,
            "p50_ms": latency.get("p50_ms"), "p95_ms": latency.get("p95_ms"),
            "p100_ms": latency.get("p100_ms"),
            "index_bytes": bench.get("index_bytes"),
        }

    # -- serving ---------------------------------------------------------
    def answer(self, question: str, top_k: int | None = None, source: str | None = None) -> dict:
        if self.pipeline is None:
            raise RuntimeError(self.status.error or "service is not loaded")
        with self._lock:
            previous = self.pipeline.top_k
            if top_k:
                self.pipeline.top_k = max(1, min(int(top_k), 50))
            try:
                result = self.pipeline.answer(question, source=source)
            finally:
                self.pipeline.top_k = previous
        payload = result.to_dict()
        payload["generator"] = getattr(self.pipeline.generator, "name", payload.get("generator", ""))
        payload["degraded"] = bool(getattr(self.pipeline.generator, "degraded", False))
        payload["confidence"] = confidence_of(result)
        return payload

    def close(self) -> None:
        self._ingest_pool.shutdown(wait=False)
        if self.pipeline is not None:
            self.pipeline.close()


def confidence_of(result) -> dict:
    """A confidence band derived from the retrieval score, not from the model.

    The model never gets a vote on how sure the system is: the number comes from
    cosine similarity between the query and the best evidence.
    """
    best = result.retrieval[0]["score"] if result.retrieval else 0.0
    if not result.grounded:
        band = "none"
    elif best >= 0.90:
        band = "high"
    elif best >= 0.85:
        band = "medium"
    else:
        band = "low"
    return {"band": band, "best_score": round(float(best), 4),
            "cited_passages": len(result.citations)}
