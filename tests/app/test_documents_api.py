"""PDF upload through the real HTTP API, end to end, with page-level citations."""

import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.app.api import create_app
from src.app.service import Service, ServiceStatus
from src.documents.store import DocumentStore
from src.rag.generator import ExtractiveGenerator
from src.rag.pipeline import RagPipeline
from tests.documents.test_pdf import FIXTURE, LexicalEmbedder

try:
    import faiss  # noqa: F401
    FAISS = True
except ImportError:
    FAISS = False


def build_client():
    directory = tempfile.TemporaryDirectory()
    store = DocumentStore(Path(directory.name))
    embedder = LexicalEmbedder()
    pipeline = RagPipeline(embedder=embedder, sources={}, generator=ExtractiveGenerator(),
                           top_k=5, min_score=0.10, default_source="corpus")
    status = ServiceStatus(ready=True, prefix="test", corpus_passages=0, index_vectors=0,
                           embedding_model="lexical-stub", embedding_dimension=256, device="cpu",
                           generator="extractive", generator_available=True)
    service = Service(pipeline=pipeline, status=status, documents=store)
    return TestClient(create_app(service, load=False)), service, directory


def upload(client, name="GOA Task-2.pdf", data=None):
    payload = data if data is not None else FIXTURE.read_bytes()
    return client.post("/api/documents", files={"file": (name, payload, "application/pdf")})


def wait_ready(client, document_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get(f"/api/documents/{document_id}").json()
        if record["ready"] or record["failed"]:
            return record
        time.sleep(0.05)
    raise AssertionError("document never reached a terminal status")


@unittest.skipUnless(FAISS, "faiss-cpu is required")
class UploadTests(unittest.TestCase):
    def setUp(self):
        self.client, self.service, self.directory = build_client()
        self.addCleanup(self.directory.cleanup)

    def test_upload_reports_progress_then_ready_with_real_counts(self):
        response = upload(self.client)
        self.assertEqual(response.status_code, 202)
        record = response.json()
        self.assertEqual(record["name"], "GOA Task-2.pdf")
        self.assertIn("progress", record)
        final = wait_ready(self.client, record["document_id"])
        self.assertTrue(final["ready"], final.get("message"))
        self.assertEqual(final["status"], "ready")
        self.assertEqual(final["pages"], 8)
        self.assertGreater(final["chunks"], 0)
        self.assertEqual(final["progress"], 1.0)
        self.assertGreater(final["characters"], 500)

    def test_uploaded_document_becomes_a_selectable_knowledge_source(self):
        record = wait_ready(self.client, upload(self.client).json()["document_id"])
        sources = self.client.get("/api/sources").json()["sources"]
        keys = [source["key"] for source in sources]
        self.assertIn(f"document:{record['document_id']}", keys)
        entry = next(s for s in sources if s["key"] == f"document:{record['document_id']}")
        self.assertEqual(entry["kind"], "document")
        self.assertEqual(entry["label"], "GOA Task-2.pdf")
        self.assertEqual(entry["size"], record["chunks"])

    def test_question_against_the_pdf_cites_the_right_page(self):
        record = wait_ready(self.client, upload(self.client).json()["document_id"])
        response = self.client.post("/api/query", json={
            "question": "गोवा का सबसे व्यस्त समुद्र तट कौन सा है?",
            "source": f"document:{record['document_id']}"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["grounded"], body["reason"])
        citation = body["citations"][0]
        self.assertEqual(citation["document"], "GOA Task-2.pdf")
        self.assertEqual(citation["page"], 7)
        self.assertIn("कलंगुट", citation["text"])
        self.assertEqual(body["sources_used"][0]["document"], "GOA Task-2.pdf")
        self.assertIn(7, body["sources_used"][0]["pages"])
        self.assertEqual(body["source"], f"document:{record['document_id']}")

    def test_unsupported_question_against_the_pdf_abstains(self):
        record = wait_ready(self.client, upload(self.client).json()["document_id"])
        self.service.pipeline.min_score = 0.95
        response = self.client.post("/api/query", json={
            "question": "मेरे बैंक खाते में कितना पैसा है?",
            "source": f"document:{record['document_id']}"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["grounded"])
        self.assertEqual(body["answer"], "")
        self.assertEqual(body["citations"], [])

    def test_duplicate_upload_returns_the_same_document(self):
        first = wait_ready(self.client, upload(self.client).json()["document_id"])
        second = upload(self.client).json()
        self.assertEqual(second["document_id"], first["document_id"])
        self.assertTrue(second["ready"], "an identical file should come back already indexed")
        self.assertEqual(len(self.client.get("/api/documents").json()["documents"]), 1)

    def test_scanned_pdf_upload_fails_with_a_readable_message(self):
        from tests.documents.test_pdf import image_only_pdf
        record = wait_ready(self.client, upload(self.client, "scan.pdf", image_only_pdf())["document_id"]
                            if isinstance(upload(self.client, "scan.pdf", image_only_pdf()), dict)
                            else upload(self.client, "scan.pdf", image_only_pdf()).json()["document_id"])
        self.assertTrue(record["failed"])
        self.assertEqual(record["reason"], "no_extractable_text")
        self.assertIn("scanned", record["message"])

    def test_non_pdf_upload_is_rejected_at_the_door(self):
        response = self.client.post("/api/documents",
                                    files={"file": ("notes.txt", b"hello", "text/plain")})
        self.assertEqual(response.status_code, 415)

    def test_empty_upload_is_rejected(self):
        response = upload(self.client, "empty.pdf", b"")
        self.assertEqual(response.status_code, 422)

    def test_unknown_source_is_a_404_not_a_crash(self):
        response = self.client.post("/api/query",
                                    json={"question": "q", "source": "document:missing"})
        self.assertEqual(response.status_code, 404)

    def test_delete_removes_the_document_and_its_source(self):
        record = wait_ready(self.client, upload(self.client).json()["document_id"])
        key = f"document:{record['document_id']}"
        self.assertEqual(self.client.delete(f"/api/documents/{record['document_id']}").status_code, 200)
        self.assertNotIn(key, [s["key"] for s in self.client.get("/api/sources").json()["sources"]])
        self.assertEqual(self.client.get(f"/api/documents/{record['document_id']}").status_code, 404)

    def test_documents_survive_a_service_restart(self):
        record = wait_ready(self.client, upload(self.client).json()["document_id"])
        restarted = Service(pipeline=self.service.pipeline, status=self.service.status,
                            documents=self.service.documents)
        restarted.pipeline.remove_source(f"document:{record['document_id']}")
        self.assertEqual(restarted.restore_documents(), 1)
        self.assertIn(f"document:{record['document_id']}",
                      [s["key"] for s in restarted.knowledge_sources()])


if __name__ == "__main__":
    unittest.main()
