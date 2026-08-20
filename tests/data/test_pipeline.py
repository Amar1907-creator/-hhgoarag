import tempfile
import unittest
from pathlib import Path

from src.data.chunking import sentence_windows, whole_passage
from src.data.deduplicate import ExactDeduplicator, passage_id
from src.data.normalize import normalize_text
from src.data.schema import validate_record


def record():
    return {"source_lang": "eng_Latn", "target_lang": "hin_Deva", "meta": {"model_name": "x"}, "query": "q", "Answer": "a", "query_id": 7, "query_type": "DESCRIPTION", "passages": {"is_selected": [1], "English_passages": ["English"], "Translated_passages": [" नमस्ते  दुनिया "]}, "Eng_Query": "q", "Eng_Answer": "a"}


class PipelineTests(unittest.TestCase):
    def test_schema_validation(self): self.assertTrue(validate_record(record()).valid)

    def test_malformed_record_is_reported(self):
        value = record(); value["passages"]["is_selected"] = []
        result = validate_record(value); self.assertFalse(result.valid); self.assertIn("passage list lengths differ", result.errors[0])

    def test_normalization(self): self.assertEqual(normalize_text(" A\u00a0  B "), "A B")

    def test_stable_passage_id(self): self.assertEqual(passage_id("hi", "text"), passage_id("hi", "text"))

    def test_exact_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            deduper = ExactDeduplicator(Path(directory) / "seen.sqlite")
            self.assertTrue(deduper.add("p_one")); self.assertFalse(deduper.add("p_one")); deduper.close()

    def test_chunking_utilities(self):
        self.assertEqual(whole_passage("a"), ["a"])
        self.assertEqual(sentence_windows("One. Two. Three.", 2, 1), ["One. Two.", "Two. Three.", "Three."])


if __name__ == "__main__": unittest.main()
