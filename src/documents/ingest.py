"""Upload -> validate -> extract -> chunk -> embed -> index, with visible status.

Runs entirely locally. The uploaded file is never transmitted anywhere.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from src.documents.chunk import chunk_pages
from src.documents.extract import content_id, extract_pdf
from src.documents.index import FlatIPIndex
from src.documents.store import (
    STATUS_CHUNKING, STATUS_EMBEDDING, STATUS_EXTRACTING, STATUS_FAILED,
    STATUS_INDEXING, STATUS_READY, DocumentRecord, DocumentStore,
)

MAX_UPLOAD_BYTES = 80 * 1024 * 1024
EMBED_BATCH = 32


def ingest_pdf(data: bytes, name: str, *, embedder, store: DocumentStore,
               on_status: Callable[[DocumentRecord], None] | None = None,
               batch_size: int = EMBED_BATCH, reuse: bool = True) -> DocumentRecord:
    """Ingest one PDF. Never raises for a bad document; the record carries why."""
    started = time.perf_counter()
    document_id = content_id(data)
    record = DocumentRecord(document_id=document_id, name=name, bytes=len(data))

    def announce(status: str) -> None:
        record.status = status
        store.save_record(record)
        if on_status:
            on_status(record)

    existing = store.get(document_id) if reuse else None
    if existing and existing.status == STATUS_READY and store.index_dir(document_id).exists():
        # Same bytes, already indexed: return the existing document rather than
        # spending minutes re-embedding an identical file.
        existing.name = existing.name or name
        return existing

    if len(data) > MAX_UPLOAD_BYTES:
        record.status, record.reason = STATUS_FAILED, "too_large"
        record.message = (f"The file is {len(data) / 1e6:.0f} MB; the limit is "
                          f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB.")
        store.save_record(record)
        return record

    announce(STATUS_EXTRACTING)
    extraction = extract_pdf(data)
    record.pages = extraction.page_count
    record.text_pages = extraction.text_pages
    record.characters = extraction.extracted_chars
    record.truncated = extraction.truncated
    if not extraction.ok:
        record.status, record.reason = STATUS_FAILED, extraction.reason
        record.message = extraction.message
        store.save_record(record)
        return record

    announce(STATUS_CHUNKING)
    chunks = chunk_pages(extraction.pages, document_id, name)
    record.chunks = len(chunks)
    if not chunks:
        record.status, record.reason = STATUS_FAILED, "no_chunks"
        record.message = "The PDF produced no usable text chunks."
        store.save_record(record)
        return record
    store.save_chunks(document_id, chunks)

    announce(STATUS_EMBEDDING)
    vectors: list[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors.append(embedder.embed_passages([chunk.text for chunk in batch]))
    matrix = np.vstack(vectors).astype(np.float32)

    announce(STATUS_INDEXING)
    index = FlatIPIndex(embedder.dimension)
    index.add([chunk.chunk_id for chunk in chunks], matrix)
    index.save(store.index_dir(document_id))

    record.embedding_model = getattr(embedder, "model_name", "")
    record.indexed_seconds = round(time.perf_counter() - started, 2)
    record.reason = "ok"
    record.message = ""
    announce(STATUS_READY)
    return record
