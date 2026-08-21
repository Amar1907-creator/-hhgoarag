"""PDF ingestion end to end, against a real multi-page Hindi/English PDF.

tests/fixtures/goa-task-2-sample.pdf is a genuine PDF with an embedded font and
8 pages of mixed Hindi and English. It is committed so these tests never depend
on a Devanagari font being installed on the machine running them.
"""

import hashlib
import io
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.documents.chunk import chunk_pages, split_page
from src.documents.extract import (
    REASON_CORRUPT, REASON_EMPTY, REASON_ENCRYPTED, REASON_NOT_PDF, REASON_NO_TEXT,
    Page, content_id, extract_pdf,
)
from src.documents.ingest import ingest_pdf
from src.documents.store import STATUS_FAILED, STATUS_READY, DocumentStore
from src.rag.generator import ExtractiveGenerator
from src.rag.pipeline import RagPipeline
from src.rag.sources import DocumentSource

try:
    import faiss  # noqa: F401
    FAISS = True
except ImportError:
    FAISS = False

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "goa-task-2-sample.pdf"
DIMENSION = 256


class LexicalEmbedder:
    """Deterministic hashed bag-of-words vectors.

    Real vector semantics without downloading a model: tokens hash into buckets,
    so a query and a passage sharing words genuinely land near each other. Good
    enough to prove retrieval wiring, and it never touches the network.

    Tokenisation cannot use \\w: Python excludes Devanagari combining vowel signs
    from it, which shreds गोवा into ['ग', 'व'] and destroys the signal. Devanagari
    runs are matched explicitly, marks included.
    """

    TOKEN = re.compile(r"[\u0900-\u097F]+|[A-Za-z0-9]+")
    STOP = {"the", "is", "in", "a", "an", "of", "to", "and", "when", "what", "does",
            "for", "on", "at", "it", "this", "that", "are", "was", "be", "with"}

    dimension = DIMENSION
    model_name = "lexical-stub"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in self.TOKEN.findall(text.lower()):
            if token in self.STOP:
                continue
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_passages(self, texts):
        return np.vstack([self._vector(t) for t in texts]).astype(np.float32)

    def embed_queries(self, texts):
        return self.embed_passages(texts)


def make_store():
    directory = tempfile.TemporaryDirectory()
    return DocumentStore(Path(directory.name)), directory


def minimal_pdf(pages: int = 1, text: str = "hello world this is a page of text") -> bytes:
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    for _ in range(pages):
        c.drawString(70, 700, text)
        c.showPage()
    c.save()
    return buffer.getvalue()


def image_only_pdf() -> bytes:
    """A page with graphics but no text operators — a stand-in for a scan."""
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.rect(100, 100, 300, 300, fill=1)
    c.showPage()
    c.save()
    return buffer.getvalue()


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.result = extract_pdf(FIXTURE)

    def test_real_pdf_extracts(self):
        self.assertTrue(self.result.ok, self.result.reason)
        self.assertEqual(self.result.page_count, 8)
        self.assertEqual(self.result.text_pages, 8)
        self.assertGreater(self.result.extracted_chars, 500)

    def test_page_numbers_are_preserved_and_one_based(self):
        numbers = [page.number for page in self.result.pages]
        self.assertEqual(numbers, list(range(1, 9)))

    def test_content_lands_on_the_page_it_was_printed_on(self):
        by_page = {page.number: page.text for page in self.result.pages}
        self.assertIn("कलंगुट", by_page[7])          # beaches section, page 7
        self.assertIn("पुर्तगाली", by_page[2])        # history, page 2
        self.assertNotIn("कलंगुट", by_page[1])

    def test_hindi_english_and_mixed_text_all_survive(self):
        by_page = {page.number: page.text for page in self.result.pages}
        self.assertIn("गोवा", by_page[1])                       # Hindi
        self.assertIn("GOA Task-2", by_page[1])                 # English
        self.assertIn("November", by_page[7])                   # mixed page
        self.assertIn("समुद्र", by_page[7])

    def test_unicode_is_not_mangled(self):
        text = "".join(page.text for page in self.result.pages)
        self.assertEqual(text, text.encode("utf-8").decode("utf-8"))
        self.assertGreater(sum(1 for ch in text if "ऀ" <= ch <= "ॿ"), 200)


class RejectionTests(unittest.TestCase):
    """Refusing clearly matters as much as succeeding."""

    def test_empty_file(self):
        result = extract_pdf(b"")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, REASON_EMPTY)

    def test_not_a_pdf(self):
        result = extract_pdf(b"this is a text file, not a PDF at all")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, REASON_NOT_PDF)
        self.assertIn("not a PDF", result.message)

    def test_corrupt_pdf(self):
        broken = minimal_pdf()[:400] + b"\x00\x01\x02garbage"
        result = extract_pdf(broken)
        self.assertFalse(result.ok)
        self.assertIn(result.reason, (REASON_CORRUPT, REASON_NO_TEXT))

    def test_pdf_with_no_pages(self):
        result = extract_pdf(b"%PDF-1.4\n%%EOF\n")
        self.assertFalse(result.ok)
        self.assertIn(result.reason, (REASON_EMPTY, REASON_CORRUPT))

    def test_scanned_image_only_pdf_is_reported_not_pretended(self):
        result = extract_pdf(image_only_pdf())
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, REASON_NO_TEXT)
        self.assertIn("scanned", result.message)
        self.assertIn("OCR", result.message)

    def test_encrypted_pdf(self):
        import pypdf
        writer = pypdf.PdfWriter()
        writer.append(io.BytesIO(minimal_pdf()))
        writer.encrypt("secret")
        buffer = io.BytesIO()
        writer.write(buffer)
        result = extract_pdf(buffer.getvalue())
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, REASON_ENCRYPTED)

    def test_very_large_pdf_is_truncated_not_refused(self):
        result = extract_pdf(minimal_pdf(pages=12), max_pages=5)
        self.assertTrue(result.ok)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.pages), 5)


class ChunkingTests(unittest.TestCase):
    def test_chunks_never_span_pages(self):
        result = extract_pdf(FIXTURE)
        chunks = chunk_pages(result.pages, "doc_x", "GOA Task-2.pdf")
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertIn(chunk.page, range(1, 9))
            self.assertEqual(chunk.document_name, "GOA Task-2.pdf")
            self.assertEqual(chunk.document_id, "doc_x")
            self.assertTrue(chunk.chunk_id.startswith("doc_x_p"))
            self.assertIn(f"p{chunk.page:04d}", chunk.chunk_id)

    def test_every_text_page_produces_at_least_one_chunk(self):
        result = extract_pdf(FIXTURE)
        chunks = chunk_pages(result.pages, "d", "n")
        self.assertEqual(sorted({c.page for c in chunks}), list(range(1, 9)))

    def test_long_page_splits_into_bounded_chunks(self):
        parts = split_page("यह एक लंबा वाक्य है। " * 120, target=600)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 900 for part in parts))

    def test_short_page_is_one_chunk(self):
        self.assertEqual(len(split_page("गोवा भारत का राज्य है।")), 1)


class DeduplicationTests(unittest.TestCase):
    def test_identical_bytes_give_the_same_document_id(self):
        data = FIXTURE.read_bytes()
        self.assertEqual(content_id(data), content_id(bytes(data)))

    def test_different_bytes_give_different_ids(self):
        self.assertNotEqual(content_id(FIXTURE.read_bytes()), content_id(minimal_pdf()))


@unittest.skipUnless(FAISS, "faiss-cpu is required")
class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.store, self.directory = make_store()
        self.addCleanup(self.directory.cleanup)
        self.embedder = LexicalEmbedder()

    def test_full_ingest_reports_every_stage_and_ends_ready(self):
        seen = []
        record = ingest_pdf(FIXTURE.read_bytes(), "GOA Task-2.pdf", embedder=self.embedder,
                            store=self.store, on_status=lambda r: seen.append(r.status))
        self.assertEqual(record.status, STATUS_READY)
        self.assertEqual(seen, ["extracting", "chunking", "embedding", "indexing", "ready"])
        self.assertEqual(record.pages, 8)
        self.assertGreater(record.chunks, 0)
        self.assertTrue(record.to_dict()["ready"])
        self.assertEqual(record.to_dict()["progress"], 1.0)
        self.assertTrue(self.store.index_dir(record.document_id).joinpath("index.faiss").exists())

    def test_reupload_of_the_same_file_reuses_the_index(self):
        data = FIXTURE.read_bytes()
        first = ingest_pdf(data, "a.pdf", embedder=self.embedder, store=self.store)
        seen = []
        second = ingest_pdf(data, "a.pdf", embedder=self.embedder, store=self.store,
                            on_status=lambda r: seen.append(r.status))
        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(seen, [], "an identical upload must not be re-embedded")

    def test_scanned_pdf_fails_with_an_explanation_and_no_index(self):
        record = ingest_pdf(image_only_pdf(), "scan.pdf", embedder=self.embedder, store=self.store)
        self.assertEqual(record.status, STATUS_FAILED)
        self.assertEqual(record.reason, REASON_NO_TEXT)
        self.assertIn("scanned", record.message)
        self.assertFalse(self.store.index_dir(record.document_id).exists())

    def test_corrupt_upload_fails_cleanly(self):
        record = ingest_pdf(b"not a pdf", "bad.pdf", embedder=self.embedder, store=self.store)
        self.assertEqual(record.status, STATUS_FAILED)
        self.assertEqual(record.reason, REASON_NOT_PDF)

    def test_store_lists_and_deletes(self):
        record = ingest_pdf(FIXTURE.read_bytes(), "GOA Task-2.pdf",
                            embedder=self.embedder, store=self.store)
        self.assertEqual([r.document_id for r in self.store.list()], [record.document_id])
        self.assertTrue(self.store.delete(record.document_id))
        self.assertEqual(self.store.list(), [])


@unittest.skipUnless(FAISS, "faiss-cpu is required")
class EndToEndDocumentAnswerTests(unittest.TestCase):
    """The product promise: ask about the PDF, get the right page cited."""

    def setUp(self):
        self.store, self.directory = make_store()
        self.addCleanup(self.directory.cleanup)
        self.embedder = LexicalEmbedder()
        self.record = ingest_pdf(FIXTURE.read_bytes(), "GOA Task-2.pdf",
                                 embedder=self.embedder, store=self.store)
        source = DocumentSource.load(self.store, self.record.document_id)
        self.pipeline = RagPipeline(embedder=self.embedder, sources={source.key: source},
                                    default_source=source.key, generator=ExtractiveGenerator(),
                                    top_k=5, min_score=0.10)

    def test_question_about_page_seven_cites_page_seven(self):
        result = self.pipeline.answer("गोवा का सबसे व्यस्त समुद्र तट कौन सा है?")
        self.assertTrue(result.grounded, result.reason)
        citation = result.citations[0]
        self.assertEqual(citation["document"], "GOA Task-2.pdf")
        self.assertEqual(citation["page"], 7)
        self.assertIn("कलंगुट", citation["text"])
        self.assertIn("chunk_id", citation)
        self.assertEqual(citation["document_id"], self.record.document_id)

    def test_sources_section_lists_document_and_pages(self):
        result = self.pipeline.answer("गोवा का सबसे व्यस्त समुद्र तट कौन सा है?")
        self.assertEqual(result.sources_used[0]["document"], "GOA Task-2.pdf")
        self.assertIn(7, result.sources_used[0]["pages"])

    def test_english_question_retrieves_from_the_english_lines(self):
        result = self.pipeline.answer("When is the tourist season in Goa?")
        self.assertTrue(result.grounded, result.reason)
        self.assertEqual(result.citations[0]["page"], 7)

    def test_unsupported_question_abstains_instead_of_inventing(self):
        pipeline = RagPipeline(embedder=self.embedder,
                               sources=self.pipeline.sources, default_source=self.pipeline.default_source,
                               generator=ExtractiveGenerator(), top_k=5, min_score=0.95)
        result = pipeline.answer("मेरे बैंक खाते में कितना पैसा है?")
        self.assertFalse(result.grounded)
        self.assertTrue(result.abstained)
        self.assertEqual(result.answer, "")
        self.assertEqual(result.citations, [])

    def test_every_retrieved_chunk_carries_the_required_fields(self):
        source = self.pipeline.resolve_source(None)
        hydrated = source.hydrate([next(iter(source.chunks))])
        row = next(iter(hydrated.values()))
        for field in ("text", "document_id", "document", "page", "chunk_id"):
            self.assertIn(field, row)


if __name__ == "__main__":
    unittest.main()
