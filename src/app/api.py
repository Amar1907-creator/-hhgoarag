"""HHGOARAG HTTP API and web interface.

GET  /            served single-page interface
GET  /health      readiness, index/corpus alignment, generator status
GET  /api/stats   corpus, index and evaluation figures
GET  /api/demo    curated demonstration questions
POST /api/query   {"question": "..."} -> grounded answer with citations
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.documents.ingest import MAX_UPLOAD_BYTES

from src.app.service import Service

STATIC = Path(__file__).resolve().parents[2] / "static"
DEMO_FILE = Path("data/demo/questions.json")

FALLBACK_DEMO = [
    {"question": "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?", "expect": "evidence"},
    {"question": "मैग्नीशियम क्या है?", "expect": "evidence"},
    {"question": "एक सख्त उबला हुआ अंडा कितने समय तक पकाते हैं?", "expect": "evidence"},
    {"question": "क्या बृहस्पति ग्रह पर मानव बस्तियाँ हैं?", "expect": "abstention"},
]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    source: str | None = Field(default=None,
                               description="knowledge source key, e.g. 'corpus:hi' or 'document:doc_abc123'")
    language: str | None = Field(default=None, max_length=8,
                                 description="language code, e.g. 'hi' or 'ta'; ignored for documents")


def create_app(service: Service | None = None, *, load: bool = True) -> FastAPI:
    state = service or Service()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Model, index and passage store are loaded once, here, never per request.
        if load and not state.status.ready:
            state.load()
        yield
        state.close()

    application = FastAPI(title="HHGOARAG", version="1.0.0", lifespan=lifespan,
                          description="Hindi grounded retrieval-augmented answering. "
                                      "Runs entirely locally; no hosted API or API key.")

    @application.get("/health")
    def health() -> JSONResponse:
        payload = state.status.to_dict()
        payload["status"] = "ok" if state.status.ready else "unavailable"
        return JSONResponse(payload, status_code=200 if state.status.ready else 503)

    @application.get("/api/stats")
    def stats() -> dict:
        status = state.status
        return {"prefix": status.prefix, "language": status.language,
                "languages_available": [e["code"] for e in status.languages if e.get("available")],
                "corpus_passages": status.corpus_passages,
                "index_vectors": status.index_vectors,
                "aligned": status.corpus_passages == status.index_vectors and status.ready,
                "embedding_model": status.embedding_model,
                "embedding_dimension": status.embedding_dimension, "device": status.device,
                "generator": status.generator, "generator_available": status.generator_available,
                "requires_api_key": False, "load_seconds": status.load_seconds,
                "metrics": status.metrics}

    @application.get("/api/demo")
    def demo() -> dict:
        if DEMO_FILE.exists():
            try:
                return {"questions": json.loads(DEMO_FILE.read_text())}
            except Exception:
                pass
        return {"questions": FALLBACK_DEMO}

    @application.post("/api/query")
    def query(request: QueryRequest) -> dict:
        if not state.status.ready:
            raise HTTPException(status_code=503, detail=state.status.error or "service not ready")
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question must not be empty")
        started = time.perf_counter()
        try:
            payload = state.answer(question, top_k=request.top_k, source=request.source,
                                   language=request.language)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("\"'")) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        payload["served_in_ms"] = round((time.perf_counter() - started) * 1e3, 2)
        return payload

    @application.get("/api/languages")
    def languages() -> dict:
        roster = state.language_roster()
        return {"languages": roster, "current": state.language,
                "available": [entry["code"] for entry in roster if entry["available"]]}

    @application.get("/api/sources")
    def sources() -> dict:
        return {"sources": state.knowledge_sources(), "default": "corpus"}

    @application.get("/api/documents")
    def documents() -> dict:
        return {"documents": [record.to_dict() for record in state.documents.list()]}

    @application.get("/api/documents/{document_id}")
    def document(document_id: str) -> dict:
        record = state.documents.get(document_id)
        if record is None:
            raise HTTPException(status_code=404, detail="no such document")
        return record.to_dict()

    @application.delete("/api/documents/{document_id}")
    def forget_document(document_id: str) -> dict:
        if not state.delete_document(document_id):
            raise HTTPException(status_code=404, detail="no such document")
        return {"deleted": document_id}

    @application.post("/api/documents", status_code=202)
    async def upload(file: UploadFile = File(...)) -> dict:
        if not state.status.ready:
            raise HTTPException(status_code=503, detail=state.status.error or "service not ready")
        name = (file.filename or "document.pdf").strip()
        if not name.lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="only PDF files are supported")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="the uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413,
                                detail=f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
        # Ingestion continues in the background; the client polls the record.
        return state.ingest_document(data, name).to_dict()

    @application.get("/")
    def index():
        page = STATIC / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="interface not found")
        return FileResponse(page)

    application.state.service = state
    return application
