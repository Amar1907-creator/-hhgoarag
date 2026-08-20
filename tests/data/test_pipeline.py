import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.data.chunking import sentence_windows, whole_passage
from src.data.deduplicate import ExactDeduplicator, passage_id
from src.data.normalize import normalize_text
from src.data.schema import validate_record

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_corpus.py"
_spec = importlib.util.spec_from_file_location("build_corpus_module", _SCRIPT)
build_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_corpus)


def record():
    return {"source_lang": "eng_Latn", "target_lang": "hin_Deva", "meta": {"model_name": "x"}, "query": "q", "Answer": "a", "query_id": 7, "query_type": "DESCRIPTION", "passages": {"is_selected": [1], "English_passages": ["English"], "Translated_passages": [" नमस्ते  दुनिया "]}, "Eng_Query": "q", "Eng_Answer": "a"}


class PipelineTests(unittest.TestCase):
    def test_schema_validation(self): self.assertTrue(validate_record(record()).valid)

    def test_malformed_record_is_reported(self):
        value = record(); value["passages"]["is_selected"] = []
        result = validate_record(value); self.assertFalse(result.valid); self.assertIn("passage list lengths differ", result.errors[0])

    def test_normalization(self): self.assertEqual(normalize_text(" A   B "), "A B")

    def test_stable_passage_id(self): self.assertEqual(passage_id("hi", "text"), passage_id("hi", "text"))

    def test_exact_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            deduper = ExactDeduplicator(Path(directory) / "seen.sqlite")
            self.assertTrue(deduper.add("p_one")); self.assertFalse(deduper.add("p_one")); deduper.close()

    def test_chunking_utilities(self):
        self.assertEqual(whole_passage("a"), ["a"])
        self.assertEqual(sentence_windows("One. Two. Three.", 2, 1), ["One. Two.", "Two. Three.", "Three."])


class DeduplicationStateTests(unittest.TestCase):
    """A fresh build must not inherit a previous run's dedup state."""

    def test_state_persists_across_processes_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "seen.sqlite"
            first = ExactDeduplicator(database)
            self.assertTrue(first.add("p_one")); first.close()
            second = ExactDeduplicator(database)
            self.assertFalse(second.add("p_one"), "stale state should still suppress a repeat ID")
            second.close()

    def test_reset_makes_a_rerun_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "seen.sqlite"
            first = ExactDeduplicator(database, reset=True)
            emitted_first = [first.add(f"p_{n}") for n in range(5)]
            self.assertEqual(first.count(), 5); first.close()
            second = ExactDeduplicator(database, reset=True)
            emitted_second = [second.add(f"p_{n}") for n in range(5)]
            self.assertEqual(second.count(), 5); second.close()
            self.assertEqual(emitted_first, emitted_second, "identical input must emit an identical corpus")
            self.assertTrue(all(emitted_second))

    def test_count_reports_retained_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            deduper = ExactDeduplicator(Path(directory) / "seen.sqlite", reset=True)
            self.assertEqual(deduper.count(), 0)
            deduper.add("p_a"); deduper.add("p_a"); deduper.add("p_b")
            self.assertEqual(deduper.count(), 2); deduper.close()


class OutputGuardTests(unittest.TestCase):
    """Existing artifacts must never be silently replaced."""

    def paths(self, directory, prefix="hi-train"):
        return build_corpus.output_paths(Path(directory), prefix)

    def test_prefix_controls_every_artifact_name(self):
        paths = self.paths("/tmp/x", "hi-train-100k")
        self.assertEqual(paths["corpus"].name, "hi-train-100k-corpus.jsonl")
        self.assertEqual(paths["provenance"].name, "hi-train-100k-provenance.jsonl")
        self.assertEqual(paths["errors"].name, "hi-train-100k-validation-errors.jsonl")
        self.assertEqual(paths["seen"].name, "hi-train-100k-seen.sqlite3")

    def test_clean_directory_is_writable(self):
        with tempfile.TemporaryDirectory() as directory:
            build_corpus.assert_safe_to_write(self.paths(directory), overwrite=False, append=False)

    def test_existing_artifact_blocks_a_fresh_build(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            paths["corpus"].write_text("{}\n")
            with self.assertRaises(SystemExit) as caught:
                build_corpus.assert_safe_to_write(paths, overwrite=False, append=False)
            self.assertIn("refusing to overwrite", str(caught.exception))

    def test_overwrite_and_append_bypass_the_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(directory)
            paths["corpus"].write_text("{}\n")
            build_corpus.assert_safe_to_write(paths, overwrite=True, append=False)
            build_corpus.assert_safe_to_write(paths, overwrite=False, append=True)


if __name__ == "__main__": unittest.main()
