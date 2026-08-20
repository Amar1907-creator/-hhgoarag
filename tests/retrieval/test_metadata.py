import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.metadata import MemoryMetadataStore, OffsetMetadataStore, load_metadata, resolve


def write_corpus(directory: Path, rows: list[dict]) -> Path:
    path = directory / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def sample_rows() -> list[dict]:
    return [
        {"passage_id": "p_a", "text": "alpha", "language": "hin_Deva", "source_id": "1"},
        {"passage_id": "p_b", "text": "नमस्ते दुनिया", "language": "hin_Deva", "source_id": "2"},
        {"passage_id": "p_c", "text": "gamma", "language": "hin_Deva", "source_id": "3"},
    ]


class MetadataTests(unittest.TestCase):
    def test_mapping_and_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"; path.write_text(json.dumps({"passage_id": "p", "text": "x", "language": "hi"}) + "\n")
            metadata = load_metadata(path)
            self.assertEqual(resolve(metadata, ["p"]), [{"passage_id": "p", "language": "hi"}])


class OffsetStoreTests(unittest.TestCase):
    """The offset store must be indistinguishable from the in-memory one."""

    def test_matches_memory_store_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_corpus(Path(directory), sample_rows())
            memory = MemoryMetadataStore(path)
            offsets = OffsetMetadataStore.build(path)
            for ranking in (["p_a"], ["p_c", "p_a"], ["p_b"], ["p_a", "p_b", "p_c"]):
                self.assertEqual(offsets.resolve(ranking), memory.resolve(ranking), ranking)
            self.assertEqual(len(offsets), len(memory))
            offsets.close(); memory.close()

    def test_text_is_excluded_and_unicode_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_corpus(Path(directory), sample_rows())
            store = OffsetMetadataStore.build(path)
            row = store.resolve(["p_b"])[0]
            self.assertNotIn("text", row)
            self.assertEqual(row, {"passage_id": "p_b", "language": "hin_Deva", "source_id": "2"})
            store.close()

    def test_unknown_ids_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_corpus(Path(directory), sample_rows())
            store = OffsetMetadataStore.build(path)
            self.assertEqual(store.resolve(["p_missing"]), [])
            self.assertEqual([row["passage_id"] for row in store.resolve(["p_missing", "p_c"])], ["p_c"])
            store.close()

    def test_duplicate_passage_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_corpus(Path(directory), sample_rows() + [{"passage_id": "p_a", "text": "again"}])
            with self.assertRaises(ValueError):
                OffsetMetadataStore.build(path)
