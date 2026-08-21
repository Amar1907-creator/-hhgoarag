"""Registry and on-disk home for uploaded documents.

Each document lives under its own content-addressed directory, entirely apart
from the Hindi corpus artifacts. Uploading a PDF can never touch, overwrite or
degrade the corpus.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path("data/documents")

STATUS_UPLOADING = "uploading"
STATUS_EXTRACTING = "extracting"
STATUS_CHUNKING = "chunking"
STATUS_EMBEDDING = "embedding"
STATUS_INDEXING = "indexing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

ORDER = (STATUS_UPLOADING, STATUS_EXTRACTING, STATUS_CHUNKING,
         STATUS_EMBEDDING, STATUS_INDEXING, STATUS_READY)


@dataclass
class DocumentRecord:
    document_id: str
    name: str
    status: str = STATUS_UPLOADING
    pages: int = 0
    text_pages: int = 0
    chunks: int = 0
    characters: int = 0
    bytes: int = 0
    truncated: bool = False
    reason: str = ""
    message: str = ""
    uploaded_at: float = field(default_factory=time.time)
    indexed_seconds: float = 0.0
    embedding_model: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["progress"] = (ORDER.index(self.status) + 1) / len(ORDER) if self.status in ORDER else 1.0
        payload["ready"] = self.status == STATUS_READY
        payload["failed"] = self.status == STATUS_FAILED
        return payload


class DocumentStore:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, document_id: str) -> Path:
        return self.root / document_id

    def exists(self, document_id: str) -> bool:
        return (self.directory(document_id) / "meta.json").exists()

    def save_record(self, record: DocumentRecord) -> None:
        """Publish the record atomically.

        The UI polls this file while the ingest thread rewrites it on every
        status change. A plain write lets a poll land on a truncated file, so
        the rename is what makes progress reporting safe to read.
        """
        directory = self.directory(record.document_id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "meta.json.tmp"
        temporary.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2))
        os.replace(temporary, directory / "meta.json")

    def get(self, document_id: str) -> DocumentRecord | None:
        path = self.directory(document_id) / "meta.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return None
        for key in ("progress", "ready", "failed"):
            payload.pop(key, None)
        return DocumentRecord(**payload)

    def list(self) -> list[DocumentRecord]:
        records = [self.get(path.name) for path in sorted(self.root.iterdir()) if path.is_dir()]
        return sorted([r for r in records if r], key=lambda r: r.uploaded_at, reverse=True)

    def save_chunks(self, document_id: str, chunks) -> Path:
        path = self.directory(document_id) / "chunks.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        return path

    def load_chunks(self, document_id: str) -> dict[str, dict]:
        path = self.directory(document_id) / "chunks.jsonl"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as handle:
            return {row["chunk_id"]: row for row in
                    (json.loads(line) for line in handle if line.strip())}

    def index_dir(self, document_id: str) -> Path:
        return self.directory(document_id) / "index"

    def delete(self, document_id: str) -> bool:
        directory = self.directory(document_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory)
        return True
