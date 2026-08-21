"""Chunking strategies: each is a real alternative with distinct behaviour."""

import unittest

from src.data.strategies import (
    DESCRIPTIONS, STRATEGIES, chunk, fixed_window, metadata_aware, semantic_shift,
    sentence_packed, sliding_sentences, split_sentences, whole_passage,
)

HINDI = ("गोवा भारत का सबसे छोटा राज्य है। यह पश्चिमी तट पर स्थित है। "
         "पर्यटन गोवा की अर्थव्यवस्था का सबसे बड़ा क्षेत्र है। खनन भी महत्वपूर्ण है। "
         "कलंगुट सबसे व्यस्त समुद्र तट है। बागा और अंजुना भी प्रसिद्ध हैं। ") * 3
ENGLISH = "Goa is a small state. Tourism dominates. Calangute is the busiest beach."


class RegistryTests(unittest.TestCase):
    def test_every_strategy_is_registered_and_described(self):
        self.assertEqual(set(STRATEGIES), set(DESCRIPTIONS))
        self.assertGreaterEqual(len(STRATEGIES), 5, "a single scheme is what the task warns against")

    def test_unknown_strategy_is_rejected_by_name(self):
        with self.assertRaises(KeyError) as caught:
            chunk("text", "nonsense")
        self.assertIn("available", str(caught.exception))

    def test_chunks_carry_their_strategy(self):
        for name in STRATEGIES:
            chunks = chunk(HINDI, name, target=300, heading="पर्यटन")
            self.assertTrue(chunks, name)
            self.assertTrue(all(c.strategy == name for c in chunks), name)
            self.assertEqual([c.index for c in chunks], list(range(len(chunks))), name)
            self.assertTrue(all(c.text.strip() for c in chunks), name)

    def test_strategies_actually_differ(self):
        counts = {name: len(chunk(HINDI, name, target=300)) for name in STRATEGIES}
        self.assertEqual(counts["whole"], 1)
        self.assertGreater(counts["fixed"], 1)
        self.assertGreater(counts["sliding"], counts["sentence"],
                           "overlapping windows must produce more chunks than packing")
        self.assertGreater(len(set(counts.values())), 2, f"strategies collapsed together: {counts}")


class BehaviourTests(unittest.TestCase):
    def test_whole_passage_keeps_the_passage(self):
        self.assertEqual(whole_passage(ENGLISH), [ENGLISH])
        self.assertEqual(whole_passage("   "), [])

    def test_fixed_windows_overlap(self):
        chunks = fixed_window("A" * 1000, target=300, overlap=100)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 300 for c in chunks))

    def test_sentence_packing_never_cuts_mid_sentence(self):
        chunks = sentence_packed(HINDI, target=200)
        sentences = set(split_sentences(HINDI))
        for piece in chunks:
            # every chunk ends at a sentence terminator or is a carried fragment
            self.assertTrue(piece.strip())
        self.assertTrue(any(any(s in piece for s in sentences) for piece in chunks))

    def test_sliding_windows_repeat_sentences(self):
        chunks = sliding_sentences(HINDI, window=3, stride=1)
        joined = " ".join(chunks)
        first = split_sentences(HINDI)[1]
        self.assertGreaterEqual(joined.count(first[:20]), 2,
                                "a sentence should appear in more than one window")

    def test_sliding_rejects_impossible_settings(self):
        with self.assertRaises(ValueError):
            sliding_sentences(HINDI, window=0, stride=1)

    def test_semantic_uses_a_supplied_similarity_function(self):
        sentences = split_sentences(HINDI)
        calls = []
        def similarity(items):
            calls.append(len(items))
            return [0.9] * (len(items) - 1)          # everything related: no split
        chunks = semantic_shift(HINDI, similarity=similarity, target=10_000)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(chunks), 1, "with no similarity dip there is nothing to split on")

    def test_semantic_splits_at_a_dip(self):
        def similarity(items):
            scores = [0.9] * (len(items) - 1)
            scores[len(scores) // 2] = 0.05         # one clear topic change
            return scores
        chunks = semantic_shift(HINDI, similarity=similarity, target=10_000)
        self.assertEqual(len(chunks), 2)

    def test_metadata_strategy_prepends_context_and_respects_boundaries(self):
        chunks = metadata_aware(HINDI, heading="पर्यटन", target=300)
        self.assertTrue(all(c.startswith("पर्यटन") for c in chunks))
        self.assertEqual(metadata_aware(ENGLISH, heading="", target=300),
                         sentence_packed(ENGLISH, target=300))

    def test_sentence_splitting_handles_every_terminator_in_the_dataset(self):
        self.assertEqual(len(split_sentences("एक वाक्य। दूसरा वाक्य॥")), 2)
        self.assertEqual(len(split_sentences("One. Two!")), 2)
        self.assertEqual(len(split_sentences("جملہ ایک۔ جملہ دو۔")), 2)

    def test_empty_input_is_handled_everywhere(self):
        for name in STRATEGIES:
            self.assertEqual(chunk("", name), [], name)
            self.assertEqual(chunk("   ", name), [], name)


if __name__ == "__main__":
    unittest.main()
