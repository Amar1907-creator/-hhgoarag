"""The grounding contract: never answer beyond the retrieved evidence."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.rag.evidence import (
    MIN_SCORE, REASON_LOW_SCORE, REASON_NO_HITS, REASON_OK,
    EvidenceSet, Evidence, select_evidence, validate_citations,
)
from src.rag.generator import (
    ExtractiveGenerator, GeneratedAnswer,
    build_prompt, format_evidence, parse_response,
)
from src.rag.pipeline import REASON_MODEL_ABSTAINED, REASON_UNGROUNDED, RagPipeline
from src.rag.store import PassageTextStore

try:
    import faiss  # noqa: F401
    FAISS = True
except ImportError:
    FAISS = False

HITS = [("p_a", 0.93), ("p_b", 0.88), ("p_c", 0.81)]
TEXTS = {"p_a": "मैनहट्टन परियोजना का प्रभाव", "p_b": "परमाणु अनुसंधान", "p_c": "वैज्ञानिक संचार"}


def evidence_set():
    return select_evidence(HITS, TEXTS)


class EvidenceSelectionTests(unittest.TestCase):
    def test_selects_in_rank_order(self):
        result = evidence_set()
        self.assertTrue(result.eligible)
        self.assertEqual(result.reason, REASON_OK)
        self.assertEqual(result.passage_ids, ["p_a", "p_b", "p_c"])
        self.assertEqual([item.rank for item in result.items], [1, 2, 3])

    def test_no_hits_is_not_eligible(self):
        self.assertEqual(select_evidence([], TEXTS).reason, REASON_NO_HITS)

    def test_hits_missing_from_the_corpus_are_ignored(self):
        result = select_evidence([("p_missing", 0.99)], TEXTS)
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, REASON_NO_HITS)

    def test_weak_retrieval_abstains(self):
        weak = [(pid, MIN_SCORE - 0.05) for pid, _ in HITS]
        result = select_evidence(weak, TEXTS)
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, REASON_LOW_SCORE)

    def test_items_below_threshold_are_dropped_not_the_whole_set(self):
        mixed = [("p_a", 0.95), ("p_b", 0.40)]
        result = select_evidence(mixed, TEXTS)
        self.assertTrue(result.eligible)
        self.assertEqual(result.passage_ids, ["p_a"])

    def test_character_budget_is_respected(self):
        long_texts = {"p_a": "क" * 500, "p_b": "ख" * 500, "p_c": "ग" * 500}
        result = select_evidence(HITS, long_texts, max_chars=1100)
        self.assertEqual(len(result.items), 2)

    def test_max_items_is_respected(self):
        result = select_evidence(HITS, TEXTS, max_items=2)
        self.assertEqual(len(result.items), 2)


class CitationValidationTests(unittest.TestCase):
    def test_invented_citations_are_separated(self):
        kept, invented = validate_citations(["p_a", "p_zzz"], evidence_set())
        self.assertEqual(kept, ["p_a"])
        self.assertEqual(invented, ["p_zzz"])


class ResponseParsingTests(unittest.TestCase):
    def test_plain_json(self):
        answer, citations, insufficient = parse_response(
            '{"answer": "उत्तर", "citations": [1, 3], "insufficient": false}', evidence_set())
        self.assertEqual(answer, "उत्तर")
        self.assertEqual(citations, ["p_a", "p_c"])
        self.assertFalse(insufficient)

    def test_json_wrapped_in_prose(self):
        answer, citations, _ = parse_response(
            'Here you go:\n{"answer": "उत्तर", "citations": [2]}\nhope that helps', evidence_set())
        self.assertEqual(answer, "उत्तर")
        self.assertEqual(citations, ["p_b"])

    def test_out_of_range_citation_numbers_are_dropped(self):
        _, citations, _ = parse_response('{"answer": "x", "citations": [1, 9, 0, -2]}', evidence_set())
        self.assertEqual(citations, ["p_a"])

    def test_duplicate_citations_collapse(self):
        _, citations, _ = parse_response('{"answer": "x", "citations": [2, 2, 2]}', evidence_set())
        self.assertEqual(citations, ["p_b"])

    def test_empty_answer_counts_as_insufficient(self):
        _, _, insufficient = parse_response('{"answer": "", "citations": []}', evidence_set())
        self.assertTrue(insufficient)

    def test_unparseable_reply_yields_no_citations(self):
        answer, citations, _ = parse_response("I think it is probably about physics.", evidence_set())
        self.assertEqual(citations, [])
        self.assertIn("physics", answer)

    def test_prompt_contains_every_evidence_item_and_the_question(self):
        prompt = build_prompt("सवाल?", evidence_set())
        self.assertIn("सवाल?", prompt)
        for text in TEXTS.values():
            self.assertIn(text, prompt)
        self.assertIn("[1]", format_evidence(evidence_set()))


class StubEmbedder:
    dimension = 4
    model_name = "stub"

    def __init__(self, vector=(1.0, 0.0, 0.0, 0.0)):
        self.vector = vector

    def embed_queries(self, texts):
        return np.asarray([self.vector] * len(texts), dtype=np.float32)

    def embed_passages(self, texts):
        return np.asarray([self.vector] * len(texts), dtype=np.float32)


class StubIndex:
    def __init__(self, hits): self.hits = hits
    def search(self, vectors, top_k): return [self.hits[:top_k]]


class ScriptedGenerator:
    name = "scripted"
    def __init__(self, answer): self.answer = answer
    def generate(self, question, evidence): return self.answer


def text_store(rows):
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for passage_id, text in rows.items():
            handle.write(json.dumps({"passage_id": passage_id, "text": text,
                                     "language": "hin_Deva"}, ensure_ascii=False) + "\n")
    store = PassageTextStore.build(path)
    return store, directory


class PipelineTests(unittest.TestCase):
    def pipeline(self, hits, generator):
        store, directory = text_store(TEXTS)
        self.addCleanup(directory.cleanup)
        self.addCleanup(store.close)
        return RagPipeline(embedder=StubEmbedder(), index=StubIndex(hits),
                           texts=store, generator=generator)

    def test_grounded_answer_carries_citations_with_text(self):
        pipeline = self.pipeline(HITS, ScriptedGenerator(
            GeneratedAnswer(answer="उत्तर", citations=["p_a"], model="scripted")))
        result = pipeline.answer("सवाल?")
        self.assertTrue(result.grounded)
        self.assertFalse(result.abstained)
        self.assertEqual(result.reason, REASON_OK)
        self.assertEqual(result.citations[0]["passage_id"], "p_a")
        self.assertEqual(result.citations[0]["text"], TEXTS["p_a"])
        self.assertIn("total", result.timings_ms)
        self.assertEqual(len(result.retrieval), 3)

    def test_weak_retrieval_abstains_before_calling_the_model(self):
        called = []
        class Recorder:
            name = "recorder"
            def generate(self, question, evidence):
                called.append(question)
                return GeneratedAnswer(answer="should never run", citations=["p_a"])
        pipeline = self.pipeline([(pid, 0.30) for pid, _ in HITS], Recorder())
        result = pipeline.answer("सवाल?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, REASON_LOW_SCORE)
        self.assertEqual(called, [], "the model must not be called without eligible evidence")
        self.assertEqual(result.answer, "")

    def test_answer_with_only_invented_citations_is_refused(self):
        pipeline = self.pipeline(HITS, ScriptedGenerator(
            GeneratedAnswer(answer="प्लॉज़िबल लेकिन बिना स्रोत", citations=["p_hallucinated"])))
        result = pipeline.answer("सवाल?")
        self.assertFalse(result.grounded)
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, REASON_UNGROUNDED)
        self.assertEqual(result.answer, "")
        self.assertEqual(result.invented_citations, ["p_hallucinated"])

    def test_partly_invented_citations_keep_the_valid_ones(self):
        pipeline = self.pipeline(HITS, ScriptedGenerator(
            GeneratedAnswer(answer="उत्तर", citations=["p_b", "p_nope"])))
        result = pipeline.answer("सवाल?")
        self.assertTrue(result.grounded)
        self.assertEqual([c["passage_id"] for c in result.citations], ["p_b"])
        self.assertEqual(result.invented_citations, ["p_nope"])

    def test_model_abstention_is_honoured(self):
        pipeline = self.pipeline(HITS, ScriptedGenerator(
            GeneratedAnswer(answer="", citations=[], insufficient=True)))
        result = pipeline.answer("सवाल?")
        self.assertTrue(result.abstained)
        self.assertEqual(result.reason, REASON_MODEL_ABSTAINED)

    def test_empty_question_is_rejected(self):
        pipeline = self.pipeline(HITS, ScriptedGenerator(GeneratedAnswer(answer="x", citations=["p_a"])))
        self.assertEqual(pipeline.answer("   ").reason, "empty_question")

    def test_extractive_fallback_needs_no_api_key(self):
        pipeline = self.pipeline(HITS, ExtractiveGenerator())
        result = pipeline.answer("सवाल?")
        self.assertTrue(result.grounded)
        self.assertEqual(result.answer, TEXTS["p_a"])
        self.assertEqual(result.citations[0]["passage_id"], "p_a")


class TextStoreTests(unittest.TestCase):
    def test_round_trip_and_unknown_ids(self):
        store, directory = text_store(TEXTS)
        self.addCleanup(directory.cleanup)
        self.addCleanup(store.close)
        self.assertEqual(store.texts(["p_b"]), {"p_b": TEXTS["p_b"]})
        self.assertEqual(store.texts(["p_missing"]), {})
        self.assertEqual(len(store), 3)


@unittest.skipUnless(FAISS, "faiss-cpu is required")
class IndexIntegrationTests(unittest.TestCase):
    """The real index type the pipeline loads, end to end."""

    def test_pipeline_over_a_real_faiss_index(self):
        from src.retrieval.index import FaissHNSWIndex
        store, directory = text_store(TEXTS)
        self.addCleanup(directory.cleanup)
        self.addCleanup(store.close)
        index = FaissHNSWIndex(4, m=4)
        vectors = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        index.add(["p_a", "p_b", "p_c"], vectors)
        pipeline = RagPipeline(embedder=StubEmbedder(), index=index, texts=store,
                               generator=ExtractiveGenerator(), top_k=3)
        result = pipeline.answer("सवाल?")
        self.assertTrue(result.grounded)
        self.assertEqual(result.citations[0]["passage_id"], "p_a")


if __name__ == "__main__":
    unittest.main()
