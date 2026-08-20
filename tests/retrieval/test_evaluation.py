import unittest

from src.retrieval.evaluation import evaluate


class EvaluationTests(unittest.TestCase):
    def test_metrics(self):
        scores = evaluate({"q1": ["p1", "p2"], "q2": ["p3", "p4"]}, {"q1": {"p2"}, "q2": {"p3"}})
        self.assertEqual(scores["recall_at_1"], .5); self.assertEqual(scores["recall_at_5"], 1.0); self.assertEqual(scores["mrr"], .75)

    def test_no_labels_is_invalid(self):
        with self.assertRaises(ValueError): evaluate({"q": ["p"]}, {"q": set()})
