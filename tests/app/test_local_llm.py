"""The runtime generator must be local and must never need an API key."""

import json
import unittest
from pathlib import Path

from src.rag.evidence import select_evidence
from src.rag.generator import (
    MODEL_PREFERENCE, ExtractiveGenerator, FallbackGenerator, OllamaError,
    OllamaGenerator, build_generator, choose_model, parse_response,
)

TEXTS = {"p_a": "मैनहट्टन परियोजना", "p_b": "परमाणु अनुसंधान"}
HITS = [("p_a", 0.94), ("p_b", 0.88)]


def evidence():
    return select_evidence(HITS, TEXTS)


class NoHostedDependencyTests(unittest.TestCase):
    def test_no_anthropic_or_api_key_anywhere_in_the_runtime(self):
        root = Path(__file__).resolve().parents[2]
        offenders = []
        for path in list((root / "src").rglob("*.py")) + [root / "static/index.html"]:
            text = path.read_text()
            for needle in ("ANTHROPIC_API_KEY", "anthropic", "api.anthropic.com", "OPENAI_API_KEY"):
                if needle in text:
                    offenders.append(f"{path.relative_to(root)}: {needle}")
        self.assertEqual(offenders, [], "the shipped product must not reference a hosted LLM API")

    def test_generator_reports_that_no_key_is_required(self):
        generator = OllamaGenerator(model="qwen2.5:3b-instruct", transport=lambda payload: {})
        self.assertFalse(generator.status()["requires_api_key"])


class ModelSelectionTests(unittest.TestCase):
    def test_prefers_the_strongest_installed_model(self):
        self.assertEqual(choose_model(["qwen2.5:3b-instruct", "qwen2.5:7b-instruct"]),
                         "qwen2.5:7b-instruct")

    def test_tolerates_quantisation_suffixes(self):
        self.assertEqual(choose_model(["qwen2.5:7b-instruct-q4_K_M"]), "qwen2.5:7b-instruct-q4_K_M")

    def test_falls_back_to_any_instruct_model(self):
        self.assertEqual(choose_model(["mistral:7b-instruct"]), "mistral:7b-instruct")

    def test_no_models_means_none(self):
        self.assertIsNone(choose_model([]))

    def test_preference_list_is_ordered_largest_first(self):
        self.assertTrue(MODEL_PREFERENCE[0].startswith("qwen2.5:7b"))


class OllamaTransportTests(unittest.TestCase):
    def transport(self, content):
        captured = {}
        def send(payload):
            captured.update(payload)
            return {"message": {"content": content}, "eval_count": 42,
                    "prompt_eval_count": 300, "eval_duration": 1_500_000_000}
        return send, captured

    def test_grounded_generation(self):
        send, captured = self.transport('{"answer": "यह उत्तर है", "citations": [1], "insufficient": false}')
        generator = OllamaGenerator(model="qwen2.5:3b-instruct", transport=send)
        result = generator.generate("सवाल?", evidence())
        self.assertEqual(result.answer, "यह उत्तर है")
        self.assertEqual(result.citations, ["p_a"])
        self.assertEqual(result.usage["output_tokens"], 42)
        self.assertEqual(captured["options"]["temperature"], 0)
        self.assertFalse(captured["stream"])
        self.assertEqual(captured["format"], "json")
        self.assertIn("ONLY the numbered evidence", captured["messages"][0]["content"])

    def test_model_abstention_passes_through(self):
        send, _ = self.transport('{"answer": "", "citations": [], "insufficient": true}')
        result = OllamaGenerator(model="m", transport=send).generate("q", evidence())
        self.assertTrue(result.insufficient)

    def test_string_citations_are_coerced(self):
        answer, citations, _ = parse_response('{"answer":"a","citations":["1","2"]}', evidence())
        self.assertEqual(citations, ["p_a", "p_b"])

    def test_unreachable_ollama_raises_a_clear_error(self):
        def send(payload): raise ConnectionRefusedError("connection refused")
        generator = OllamaGenerator(model="m", transport=send)
        with self.assertRaises(OllamaError) as caught:
            generator.generate("q", evidence())
        self.assertIn("Ollama request failed", str(caught.exception))

    def test_missing_model_refuses_before_calling(self):
        generator = OllamaGenerator(model="m", transport=lambda p: {})
        generator.available = False
        with self.assertRaises(OllamaError) as caught:
            generator.generate("q", evidence())
        self.assertIn("ollama pull", str(caught.exception))


class FallbackTests(unittest.TestCase):
    def test_falls_back_to_extraction_when_the_local_model_fails(self):
        class Broken:
            name = "broken"
            def generate(self, question, evidence): raise OllamaError("down")
        generator = FallbackGenerator(Broken())
        result = generator.generate("सवाल?", evidence())
        self.assertTrue(generator.degraded)
        self.assertEqual(result.citations, ["p_a"])
        self.assertIn("unavailable", result.model)

    def test_healthy_primary_is_not_marked_degraded(self):
        send = lambda payload: {"message": {"content": '{"answer":"ठीक","citations":[1]}'}}
        generator = FallbackGenerator(OllamaGenerator(model="m", transport=send))
        result = generator.generate("q", evidence())
        self.assertFalse(generator.degraded)
        self.assertEqual(result.answer, "ठीक")

    def test_build_generator_extractive_needs_nothing(self):
        generator = build_generator("extractive")
        self.assertIsInstance(generator, ExtractiveGenerator)
        self.assertEqual(generator.generate("q", evidence()).citations, ["p_a"])


if __name__ == "__main__":
    unittest.main()
