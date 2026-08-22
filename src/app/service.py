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
from src.languages import DEFAULT_CODE, Language, find as find_language, from_prefix
from src.rag.sources import CorpusSource, DocumentSource
from src.speech.providers import SpeechError, build_speech
from src.rag.store import PassageTextStore

DEFAULT_PREFIX = os.environ.get("HHGOARAG_PREFIX", "")
DEFAULT_LANGUAGE = os.environ.get("HHGOARAG_LANGUAGE", DEFAULT_CODE)
DEFAULT_EMBEDDING_MODEL = os.environ.get("HHGOARAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
PROCESSED = Path("data/processed")
MANIFESTS = Path("data/manifests")

# Which embedder implementation to load. "torch" (default) is the existing
# SentenceTransformerE5 path used for local development. "onnx" is the
# production path: no torch, no sentence-transformers, just onnxruntime and a
# standalone tokenizer loading the official ONNX export of the same model.
# Both produce the same 384-dim, L2-normalised vectors for the same unchanged
# FAISS index -- this is purely a runtime-backend choice, not a model change.
DEFAULT_EMBEDDING_BACKEND = os.environ.get("HHGOARAG_EMBEDDING_BACKEND", "torch")
DEFAULT_ONNX_MODEL_PATH = os.environ.get(
    "HHGOARAG_ONNX_MODEL_PATH", "data/models/multilingual-e5-small-onnx/model.onnx")
DEFAULT_ONNX_TOKENIZER_PATH = os.environ.get(
    "HHGOARAG_ONNX_TOKENIZER_PATH", "data/models/multilingual-e5-small-onnx/tokenizer.json")


def build_embedder(*, backend: str, model: str, device: str | None):
    """Resolve the configured embedder. Isolated so Service.load() stays readable."""
    if backend == "onnx":
        from src.retrieval.embedding import OnnxE5Embedder
        return OnnxE5Embedder(DEFAULT_ONNX_MODEL_PATH, DEFAULT_ONNX_TOKENIZER_PATH)
    from src.retrieval.embedding import SentenceTransformerE5
    return SentenceTransformerE5(model, device=device)


def resolve_device(requested: str | None = None) -> str:
    """Default to CPU for query-time embedding.

    Measured on this machine: a single query's embedding costs 18-20ms on CPU
    versus 240-350ms on MPS. Apple's Metal backend pays a fixed per-call
    dispatch/transfer cost that only amortises over large batches -- exactly
    what happens once at corpus-build time, not what happens here, where one
    query is embedded per request. GPU acceleration is still available on
    request, it is just no longer auto-selected for serving.
    """
    if requested:
        return requested
    override = os.environ.get("HHGOARAG_DEVICE")
    if override:
        return override
    return "cpu"


def built_corpora() -> dict[str, str]:
    """Language code -> artifact prefix, for every language with a usable index.

    Scans rather than being configured: a language becomes available the moment
    its corpus and index exist, with no code or settings change.
    """
    found: dict[str, tuple[int, str]] = {}
    for path in sorted(PROCESSED.glob("*-corpus.jsonl")):
        prefix = path.name[: -len("-corpus.jsonl")]
        if not (PROCESSED / f"{prefix}-index" / "index.faiss").exists():
            continue
        language = from_prefix(prefix)
        if language is None:
            continue
        size = path.stat().st_size
        # Largest corpus wins when a language has several builds.
        if language.code not in found or size > found[language.code][0]:
            found[language.code] = (size, prefix)
    return {code: prefix for code, (_, prefix) in sorted(found.items())}


def discover_prefix(preferred: str = DEFAULT_PREFIX) -> str | None:
    """Resolve an explicit prefix, else the default language, else the largest build."""
    if preferred and (PROCESSED / f"{preferred}-corpus.jsonl").exists():
        return preferred
    corpora = built_corpora()
    if not corpora:
        return None
    if DEFAULT_LANGUAGE in corpora:
        return corpora[DEFAULT_LANGUAGE]
    return max(corpora.values(), key=lambda p: (PROCESSED / f"{p}-corpus.jsonl").stat().st_size)


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
    speech: dict = field(default_factory=dict)
    load_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)
    error: str = ""
    metrics: dict = field(default_factory=dict)
    language: str = ""
    languages: list = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        payload["uptime_seconds"] = round(time.time() - self.started_at, 1)
        payload.pop("started_at")
        return payload


class Service:
    """Holds the loaded pipeline. Thread-safe for the single-process server."""

    def __init__(self, *, prefix: str | None = None, generator_kind: str = "extractive",
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
        # Corpus indexes are loaded on first use and kept for the process
        # lifetime. Eagerly loading every built language would cost roughly
        # 90 MB of RAM each for no benefit if only one is asked about.
        self._corpus_cache: dict[str, CorpusSource] = {}
        self._stores: list[PassageTextStore] = []
        self.language = DEFAULT_LANGUAGE
        # The speech provider is required by the task specification (Sarvam or
        # ElevenLabs). Built once; it reports honestly when unconfigured.
        self.speech = build_speech()

    # -- loading ---------------------------------------------------------
    def load(self) -> ServiceStatus:
        started = time.perf_counter()
        if self.pipeline is not None:
            # Reloading must not strand the previous corpus file handle.
            try:
                self.pipeline.close()
            except Exception:
                pass
            self.pipeline = None
        prefix = discover_prefix(self.requested_prefix)
        if prefix is None:
            self.status.error = (
                f"no built corpus found under {PROCESSED}/. Build one first:\n"
                f"  python3 scripts/run_pipeline.py --limit 5000")
            return self.status

        corpus = PROCESSED / f"{prefix}-corpus.jsonl"
        index_dir = PROCESSED / f"{prefix}-index"
        language = from_prefix(prefix)
        device = resolve_device(self.requested_device)
        generator = build_generator(self.generator_kind)

        try:
            if DEFAULT_EMBEDDING_BACKEND == "onnx":
                # No torch, no sentence-transformers: build the pieces
                # RagPipeline.load() would build, but with the ONNX embedder.
                from src.retrieval.index import FaissHNSWIndex
                embedder = build_embedder(backend="onnx", model=DEFAULT_EMBEDDING_MODEL, device=device)
                loaded_index = FaissHNSWIndex.load(index_dir)
                if loaded_index.dimension != embedder.dimension:
                    raise SystemExit(
                        f"index dimension {loaded_index.dimension} does not match the ONNX "
                        f"embedder ({embedder.dimension})")
                self.pipeline = RagPipeline(embedder=embedder, index=loaded_index,
                                           texts=PassageTextStore.build(corpus),
                                           generator=generator, top_k=self.top_k,
                                           min_score=self.min_score)
            else:
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
            speech=self.speech.status() if self.speech else {},
            load_seconds=round(time.perf_counter() - started, 2),
            metrics=self.read_metrics(prefix),
            language=language.code if language else "",
            languages=self.language_roster(),
        )
        self.language = language.code if language else DEFAULT_LANGUAGE
        if language:
            # The corpus loaded above IS this language's source; register it so
            # the first query does not load the same index a second time.
            source = CorpusSource(self.pipeline.index, self.pipeline.texts,
                                  key=f"corpus:{language.code}", label=language.label)
            self._corpus_cache[language.code] = source
            self.pipeline.add_source(source)
            self.pipeline.default_source = source.key
        self.restore_documents()
        if self.status.index_vectors != corpus_passages:
            self.status.ready = False
            self.status.error = (f"index/corpus mismatch: {self.status.index_vectors} vectors "
                                 f"vs {corpus_passages} passages")
        return self.status

    # -- speech ------------------------------------------------------------
    def transcribe(self, audio: bytes, *, filename: str, language: str | None) -> dict:
        """Transcribe speech through the configured provider."""
        if self.speech is None:
            raise SpeechError("no speech provider is configured")
        locale = None
        if language:
            from src.languages import find as find_language
            entry = find_language(language)
            locale = entry.speech_locale if entry else language
        transcript = self.speech.transcribe(audio, filename=filename, language=locale)
        return transcript.to_dict()

    # -- languages ---------------------------------------------------------
    def language_roster(self) -> list[dict]:
        """Every known language, marked with whether it is actually usable here."""
        from src.languages import LANGUAGES
        corpora = built_corpora()
        roster = []
        for language in LANGUAGES:
            entry = language.to_dict()
            prefix = corpora.get(language.code)
            entry["available"] = prefix is not None
            entry["prefix"] = prefix or ""
            entry["loaded"] = language.code in self._corpus_cache
            if prefix:
                entry["metrics"] = self.read_metrics(prefix)
            roster.append(entry)
        return roster

    def available_languages(self) -> list[str]:
        return list(built_corpora())

    def corpus_source(self, code: str) -> CorpusSource:
        """Return this language's corpus source, loading it once if needed."""
        language = find_language(code)
        if language is None:
            raise KeyError(f"unknown language {code!r}")
        if language.code in self._corpus_cache:
            return self._corpus_cache[language.code]
        prefix = built_corpora().get(language.code)
        if prefix is None:
            raise KeyError(f"{language.english} has no built corpus. "
                           f"Build one with: python3 scripts/run_pipeline.py --language {language.code}")
        from src.retrieval.index import FaissHNSWIndex
        index = FaissHNSWIndex.load(PROCESSED / f"{prefix}-index")
        if index.dimension != self.pipeline.embedder.dimension:
            raise RuntimeError(f"{language.english} index has dimension {index.dimension}, "
                               f"but the embedding model produces {self.pipeline.embedder.dimension}")
        store = PassageTextStore.build(PROCESSED / f"{prefix}-corpus.jsonl")
        self._stores.append(store)
        source = CorpusSource(index, store, key=f"corpus:{language.code}", label=language.label)
        self._corpus_cache[language.code] = source
        self.pipeline.add_source(source)
        return source

    def resolve_source_key(self, source: str | None, language: str | None) -> str | None:
        """Decide which knowledge source answers this question.

        Order matters: an explicit document wins, then an already-registered
        source for the requested language, then a lazy load from disk, then
        whatever the pipeline was constructed with. The last step is what lets a
        pipeline be injected with sources that never existed on disk.
        """
        if source and source.startswith("document:"):
            return source
        wanted = language or self.language
        if source and source.startswith("corpus:"):
            wanted = source.split(":", 1)[1]
            source = None
        key = f"corpus:{wanted}"
        if self.pipeline and key in self.pipeline.sources:
            return key
        try:
            return self.corpus_source(wanted).key
        except (KeyError, RuntimeError):
            if source and self.pipeline and source in self.pipeline.sources:
                return source
            if self.pipeline and self.pipeline.default_source in self.pipeline.sources:
                return self.pipeline.default_source
            raise

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
    def answer(self, question: str, top_k: int | None = None, source: str | None = None,
               language: str | None = None) -> dict:
        if self.pipeline is None:
            raise RuntimeError(self.status.error or "service is not loaded")
        source = self.resolve_source_key(source, language)
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
        for store in self._stores:
            try:
                store.close()
            except Exception:
                pass
        self._stores.clear()
        self._corpus_cache.clear()
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
