import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.metadata import load_metadata, resolve


class MetadataTests(unittest.TestCase):
    def test_mapping_and_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"; path.write_text(json.dumps({"passage_id": "p", "text": "x", "language": "hi"}) + "\n")
            metadata = load_metadata(path)
            self.assertEqual(resolve(metadata, ["p"]), [{"passage_id": "p", "language": "hi"}])
