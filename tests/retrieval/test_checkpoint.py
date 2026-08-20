"""Interrupting an index build must never lose ordering or duplicate a passage."""

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    import faiss  # noqa: F401
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_retrieval.py"


def load_script():
    spec = importlib.util.spec_from_file_location("benchmark_retrieval_module", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubEmbedder:
    """Deterministic unit vectors; no model download, no network."""

    model_name = "stub-embedder"
    dimension = 8

    def __init__(self, fail_after_batches: int | None = None) -> None:
        self.fail_after_batches = fail_after_batches
        self.batches = 0

    def _vector(self, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).digest()[: self.dimension]
        vector = np.asarray([byte / 255.0 for byte in digest], dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def embed_passages(self, texts):
        self.batches += 1
        if self.fail_after_batches is not None and self.batches > self.fail_after_batches:
            raise KeyboardInterrupt
        return np.vstack([self._vector(text) for text in texts]).astype(np.float32)

    def embed_queries(self, texts):
        return np.vstack([self._vector(text) for text in texts]).astype(np.float32)


def write_corpus(directory: Path, count: int) -> Path:
    path = directory / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for number in range(count):
            handle.write(json.dumps({"passage_id": f"p_{number:04d}", "text": f"passage number {number}",
                                     "language": "hin_Deva", "source_id": str(number)}) + "\n")
    return path


@unittest.skipUnless(FAISS_AVAILABLE, "faiss-cpu and numpy are required")
class CheckpointResumeTests(unittest.TestCase):
    PASSAGES = 40
    BATCH = 4

    def build(self, corpus, index_dir, embedder, resume=False):
        return load_script().build_index(
            corpus_path=corpus, embedder=embedder, index_dir=index_dir, batch_size=self.BATCH,
            checkpoint_every=1, progress_every=0, resume=resume, log=io.StringIO())

    def test_uninterrupted_build_preserves_corpus_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = write_corpus(root, self.PASSAGES)
            report = self.build(corpus, root / "index", StubEmbedder())
            index = report["index"]
            self.assertEqual(index.index.ntotal, self.PASSAGES)
            self.assertEqual(index.ids, [f"p_{n:04d}" for n in range(self.PASSAGES)])

    def test_resume_reproduces_an_uninterrupted_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = write_corpus(root, self.PASSAGES)

            reference = self.build(corpus, root / "reference", StubEmbedder())["index"]

            interrupted_dir = root / "interrupted"
            with self.assertRaises(KeyboardInterrupt):
                self.build(corpus, interrupted_dir, StubEmbedder(fail_after_batches=3))

            # The checkpoint holds only completed batches, and it is on disk.
            partial = load_script().FaissHNSWIndex.load(interrupted_dir)
            self.assertEqual(partial.index.ntotal, 3 * self.BATCH)
            state = json.loads((interrupted_dir / "build_state.json").read_text())
            self.assertEqual(state["passages_done"], 3 * self.BATCH)

            resumed = self.build(corpus, interrupted_dir, StubEmbedder(), resume=True)
            self.assertEqual(resumed["resumed_from"], 3 * self.BATCH)
            self.assertEqual(resumed["index"].ids, reference.ids)
            self.assertEqual(resumed["index"].index.ntotal, self.PASSAGES)

    def test_resume_is_idempotent_when_already_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = write_corpus(root, self.PASSAGES)
            index_dir = root / "index"
            self.build(corpus, index_dir, StubEmbedder())
            again = self.build(corpus, index_dir, StubEmbedder(), resume=True)
            self.assertEqual(again["index"].index.ntotal, self.PASSAGES)
            self.assertEqual(len(again["index"].ids), self.PASSAGES)

    def test_resume_rejects_a_different_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = write_corpus(root, self.PASSAGES)
            index_dir = root / "index"
            with self.assertRaises(KeyboardInterrupt):
                self.build(corpus, index_dir, StubEmbedder(fail_after_batches=2))
            other = StubEmbedder(); other.model_name = "some-other-model"
            with self.assertRaises(SystemExit):
                self.build(corpus, index_dir, other, resume=True)

    def test_resume_rejects_a_changed_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = write_corpus(root, self.PASSAGES)
            index_dir = root / "index"
            with self.assertRaises(KeyboardInterrupt):
                self.build(corpus, index_dir, StubEmbedder(fail_after_batches=2))
            write_corpus(root, self.PASSAGES + 5)
            with self.assertRaises(SystemExit):
                self.build(corpus, index_dir, StubEmbedder(), resume=True)

    def test_hard_kill_between_renames_is_repaired(self):
        """save() publishes ids.json before index.faiss, so a kill between the
        two leaves trailing IDs whose vectors never landed. Resume must repair
        that rather than refusing, and still reproduce the reference build."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = write_corpus(root, self.PASSAGES)
            reference = self.build(corpus, root / "reference", StubEmbedder())["index"]

            index_dir = root / "index"
            with self.assertRaises(KeyboardInterrupt):
                self.build(corpus, index_dir, StubEmbedder(fail_after_batches=2))

            ids_path = index_dir / "ids.json"
            ids = json.loads(ids_path.read_text())
            orphaned = [f"p_{n:04d}" for n in range(len(ids), len(ids) + self.BATCH)]
            ids_path.write_text(json.dumps(ids + orphaned))

            module = load_script()
            with self.assertRaises(ValueError):
                module.FaissHNSWIndex.load(index_dir)  # strict load still refuses

            resumed = self.build(corpus, index_dir, StubEmbedder(), resume=True)
            self.assertEqual(resumed["resumed_from"], len(ids))
            self.assertEqual(resumed["index"].ids, reference.ids)

    def test_vectors_without_ids_are_not_silently_accepted(self):
        """The opposite inconsistency is unrecoverable and must fail loudly."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = write_corpus(root, self.PASSAGES)
            index_dir = root / "index"
            self.build(corpus, index_dir, StubEmbedder())
            ids_path = index_dir / "ids.json"
            truncated = json.loads(ids_path.read_text())[:-5]
            ids_path.write_text(json.dumps(truncated))
            module = load_script()
            with self.assertRaises(ValueError):
                module.FaissHNSWIndex.load(index_dir, repair=True)


if __name__ == "__main__":
    unittest.main()
