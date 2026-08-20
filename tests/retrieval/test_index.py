import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.retrieval.index import FaissHNSWIndex


class IndexTests(unittest.TestCase):
    def test_create_search_and_load(self):
        index = FaissHNSWIndex(2, m=4)
        index.add(["p1", "p2"], np.asarray([[1, 0], [0, 1]], dtype=np.float32))
        self.assertEqual(index.search(np.asarray([[.9, .1]], dtype=np.float32), 1)[0][0][0], "p1")
        with tempfile.TemporaryDirectory() as directory:
            index.save(Path(directory)); restored = FaissHNSWIndex.load(Path(directory))
            self.assertEqual(restored.search(np.asarray([[.1, .9]], dtype=np.float32), 1)[0][0][0], "p2")

    def test_rejects_bad_mapping(self):
        index = FaissHNSWIndex(2, m=4)
        with self.assertRaises(ValueError): index.add(["p1"], np.asarray([[1, 0], [0, 1]], dtype=np.float32))
