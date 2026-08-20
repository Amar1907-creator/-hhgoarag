import unittest

import numpy as np

from src.retrieval.embedding import Embedder


class TinyEmbedder:
    dimension = 2
    def embed_queries(self, texts): return np.ones((len(texts), 2), dtype=np.float32)
    def embed_passages(self, texts): return np.ones((len(texts), 2), dtype=np.float32)


class EmbeddingInterfaceTests(unittest.TestCase):
    def test_interface_contract(self):
        embedder: Embedder = TinyEmbedder()
        self.assertEqual(embedder.embed_queries(["q"]).shape, (1, 2))
        self.assertEqual(embedder.embed_passages(["p"]).dtype, np.float32)
