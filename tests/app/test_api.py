"""The API contract, exercised without a model or an index on disk."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.app.api import create_app
from src.app.service import Service, ServiceStatus, confidence_of
from src.rag.generator import ExtractiveGenerator
from src.rag.pipeline import RagPipeline
from src.rag.store import PassageTextStore

TEXTS = {"p_a": "मैनहट्टन परियोजना का प्रभाव बहुत बड़ा था।",
         "p_b": "परमाणु अनुसंधान में सहयोग महत्वपूर्ण था।",
         "p_c": "वैज्ञानिकों के बीच संचार आवश्यक था।"}
HITS = [("p_a", 0.94), ("p_b", 0.88), ("p_c", 0.83)]


class StubEmbedder:
    dimension = 4
    model_name = "stub"
    def embed_queries(self, texts): return np.zeros((len(texts), 4), dtype=np.float32)
    def embed_passages(self, texts): return np.zeros((len(texts), 4), dtype=np.float32)


class StubIndex:
    def __init__(self, hits=HITS): self.hits = hits
    class _Inner:
        ntotal = 3
    index = _Inner()
    def search(self, vectors, top_k): return [self.hits[:top_k]]


def build_service(hits=HITS, ready=True):
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for passage_id, text in TEXTS.items():
            handle.write(json.dumps({"passage_id": passage_id, "text": text,
                                     "language": "hin_Deva"}, ensure_ascii=False) + "\n")
    store = PassageTextStore.build(path)
    pipeline = RagPipeline(embedder=StubEmbedder(), index=StubIndex(hits),
                           texts=store, generator=ExtractiveGenerator())
    status = ServiceStatus(ready=ready, prefix="test", corpus_passages=3, index_vectors=3,
                           embedding_model="stub", embedding_dimension=4, device="cpu",
                           generator="extractive", generator_available=True, load_seconds=0.1,
                           error="" if ready else "no corpus built")
    return Service(pipeline=pipeline, status=status), directory


class HealthTests(unittest.TestCase):
    def test_health_reports_ready_and_never_requires_a_key(self):
        service, directory = build_service()
        self.addCleanup(directory.cleanup)
        client = TestClient(create_app(service, load=False))
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["corpus_passages"], body["index_vectors"])
        self.assertFalse(body["requires_api_key"])
        self.assertIn("uptime_seconds", body)

    def test_health_is_503_when_not_loaded(self):
        service, directory = build_service(ready=False)
        self.addCleanup(directory.cleanup)
        client = TestClient(create_app(service, load=False))
        response = client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "unavailable")


class QueryTests(unittest.TestCase):
    def setUp(self):
        service, directory = build_service()
        self.addCleanup(directory.cleanup)
        self.service = service
        self.client = TestClient(create_app(service, load=False))

    def test_grounded_answer_shape(self):
        response = self.client.post("/api/query", json={"question": "मैनहट्टन परियोजना क्या थी?"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in ("answer", "citations", "grounded", "abstained", "reason",
                    "confidence", "retrieval", "timings_ms", "served_in_ms", "generator"):
            self.assertIn(key, body)
        self.assertTrue(body["grounded"])
        self.assertEqual(body["citations"][0]["passage_id"], "p_a")
        self.assertIn(body["confidence"]["band"], ("high", "medium", "low"))
        self.assertFalse(body["degraded"])

    def test_weak_retrieval_returns_a_clean_abstention_not_an_error(self):
        service, directory = build_service(hits=[("p_a", 0.20), ("p_b", 0.10)])
        self.addCleanup(directory.cleanup)
        client = TestClient(create_app(service, load=False))
        response = client.post("/api/query", json={"question": "क्या बृहस्पति पर बस्तियाँ हैं?"})
        self.assertEqual(response.status_code, 200, "abstention is a valid answer, not a failure")
        body = response.json()
        self.assertFalse(body["grounded"])
        self.assertTrue(body["abstained"])
        self.assertEqual(body["answer"], "")
        self.assertEqual(body["confidence"]["band"], "none")

    def test_empty_and_oversized_questions_are_rejected(self):
        self.assertEqual(self.client.post("/api/query", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/query", json={"question": "   "}).status_code, 422)
        self.assertEqual(self.client.post("/api/query", json={"question": "x" * 1001}).status_code, 422)

    def test_top_k_is_bounded(self):
        self.assertEqual(self.client.post("/api/query", json={"question": "q", "top_k": 99}).status_code, 422)
        self.assertEqual(self.client.post("/api/query", json={"question": "q", "top_k": 2}).status_code, 200)

    def test_top_k_is_restored_after_the_request(self):
        before = self.service.pipeline.top_k
        self.client.post("/api/query", json={"question": "q", "top_k": 1})
        self.assertEqual(self.service.pipeline.top_k, before)

    def test_query_on_an_unready_service_is_503(self):
        service, directory = build_service(ready=False)
        self.addCleanup(directory.cleanup)
        client = TestClient(create_app(service, load=False))
        self.assertEqual(client.post("/api/query", json={"question": "q"}).status_code, 503)


class StatsAndDemoTests(unittest.TestCase):
    def setUp(self):
        service, directory = build_service()
        self.addCleanup(directory.cleanup)
        self.client = TestClient(create_app(service, load=False))

    def test_stats_reports_alignment(self):
        body = self.client.get("/api/stats").json()
        self.assertTrue(body["aligned"])
        self.assertFalse(body["requires_api_key"])

    def test_demo_questions_are_available(self):
        body = self.client.get("/api/demo").json()
        self.assertGreaterEqual(len(body["questions"]), 4)
        self.assertTrue(any(q.get("expect") == "abstention" for q in body["questions"]))

    def test_interface_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HHGOARAG", response.text)


class ConfidenceTests(unittest.TestCase):
    def test_bands_follow_retrieval_score_not_the_model(self):
        class R:
            def __init__(self, score, grounded):
                self.retrieval = [{"passage_id": "p", "score": score}]
                self.grounded = grounded
                self.citations = [{"passage_id": "p"}] if grounded else []
        self.assertEqual(confidence_of(R(0.95, True))["band"], "high")
        self.assertEqual(confidence_of(R(0.87, True))["band"], "medium")
        self.assertEqual(confidence_of(R(0.81, True))["band"], "low")
        self.assertEqual(confidence_of(R(0.99, False))["band"], "none")


if __name__ == "__main__":
    unittest.main()
