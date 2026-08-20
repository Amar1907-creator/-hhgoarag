import unittest

from src.data.loader import (
    LANGUAGE_ALIASES,
    coerce_record,
    hub_parquet_path,
    parquet_relative_path,
    resolve_language_code,
)


class LoaderTests(unittest.TestCase):
    def test_resolve_language_code(self):
        self.assertEqual(resolve_language_code("hi"), "hin")
        self.assertEqual(resolve_language_code("hin"), "hin")

    def test_parquet_relative_path(self):
        self.assertEqual(parquet_relative_path(split="train", config="hi"), "train/hintrain.parquet")
        self.assertEqual(parquet_relative_path(split="validation", config="hi"), "validation/hinval.parquet")

    def test_telugu_train_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no train file"):
            parquet_relative_path(split="train", config="te")

    def test_hub_parquet_path(self):
        revision = "bf5cdc1f26e581e519018e434db14edd1b77602b"
        expected = f"datasets/ai4bharat/MSMARCO-XI@{revision}/train/hintrain.parquet"
        self.assertEqual(hub_parquet_path(split="train", config="hi", revision=revision), expected)

    def test_coerce_record(self):
        raw = {
            "source_lang": "eng_Latn",
            "target_lang": "hin_Deva",
            "meta": {"model_name": "x"},
            "query": "q",
            "Answer": "a",
            "query_id": 1185869,
            "query_type": "DESCRIPTION",
            "passages": {
                "is_selected": [1, 0],
                "English_passages": ["a", "b"],
                "Translated_passages": [" alpha ", " beta "],
            },
            "Eng_Query": "q",
            "Eng_Answer": "a",
        }
        record = coerce_record(raw)
        self.assertEqual(record["query_id"], 1185869)
        self.assertIsInstance(record["query_id"], int)
        self.assertEqual(record["passages"]["is_selected"], [1, 0])

    def test_aliases_cover_inventory(self):
        self.assertIn("hin", LANGUAGE_ALIASES.values())
        self.assertIn("tel", LANGUAGE_ALIASES.values())


if __name__ == "__main__":
    unittest.main()
