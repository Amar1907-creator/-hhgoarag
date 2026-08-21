"""One pipeline, every language.

These tests put real text in each required script through the shared pipeline
stages -- normalisation, passage identity, chunking -- because a multilingual
claim is only worth what the scripts actually survive.
"""

import unittest

from src.data.chunking import sentences
from src.data.deduplicate import passage_id
from src.data.loader import LANGUAGE_ALIASES, parquet_relative_path
from src.data.normalize import normalize_text
from src.documents.chunk import split_page
from src.languages import (
    BY_CODE, DEFAULT_CODE, LANGUAGES, SPEECH_NONE, SPEECH_UNTESTED, SPEECH_VERIFIED,
    TRAINABLE, find, from_prefix, get,
)

# One real sentence per required script.
SAMPLES = {
    "as": "অসম ভাৰতৰ এখন ৰাজ্য।",
    "bn": "বাংলা একটি সমৃদ্ধ ভাষা।",
    "gu": "ગુજરાત ભારતનું એક રાજ્ય છે।",
    "hi": "गोवा भारत का सबसे छोटा राज्य है।",
    "kn": "ಕರ್ನಾಟಕ ಭಾರತದ ಒಂದು ರಾಜ್ಯವಾಗಿದೆ.",
    "ml": "കേരളം ഇന്ത്യയിലെ ഒരു സംസ്ഥാനമാണ്.",
    "mr": "महाराष्ट्र हे भारतातील एक राज्य आहे.",
    "ne": "नेपाल एउटा सुन्दर देश हो।",
    "or": "ଓଡ଼ିଶା ଭାରତର ଏକ ରାଜ୍ୟ।",
    "pa": "ਪੰਜਾਬ ਭਾਰਤ ਦਾ ਇੱਕ ਰਾਜ ਹੈ।",
    "sa": "संस्कृतम् एका प्राचीना भाषा अस्ति।",
    "ta": "தமிழ்நாடு இந்தியாவின் ஒரு மாநிலம் ஆகும்.",
    "te": "తెలంగాణ భారతదేశంలోని ఒక రాష్ట్రం.",
    "ur": "اردو جنوبی ایشیا کی ایک زبان ہے۔",
}


class RegistryTests(unittest.TestCase):
    def test_every_dataset_language_is_registered(self):
        self.assertEqual(len(LANGUAGES), 14)
        self.assertEqual(len(TRAINABLE), 13)
        self.assertEqual(sorted(l.code for l in TRAINABLE),
                         ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "ur"])

    def test_telugu_is_present_but_not_trainable(self):
        telugu = get("te")
        self.assertFalse(telugu.has_train)
        self.assertIn("validation", telugu.note)

    def test_every_language_is_presentable(self):
        for language in LANGUAGES:
            self.assertTrue(language.native.strip(), language.code)
            self.assertTrue(language.english.strip(), language.code)
            self.assertTrue(language.script.strip(), language.code)
            self.assertIn(language.direction, ("ltr", "rtl"), language.code)
            self.assertTrue(language.placeholder().strip(), language.code)
            self.assertNotIn(language.code, language.placeholder().lower().split(),
                             f"{language.code} placeholder leaks a config code")

    def test_urdu_is_right_to_left_and_others_are_not(self):
        self.assertEqual(get("ur").direction, "rtl")
        self.assertTrue(all(l.direction == "ltr" for l in LANGUAGES if l.code != "ur"))

    def test_registry_codes_match_the_loader(self):
        for language in LANGUAGES:
            self.assertIn(language.code, LANGUAGE_ALIASES, language.code)
            self.assertEqual(LANGUAGE_ALIASES[language.code], language.parquet, language.code)

    def test_every_trainable_language_resolves_to_a_train_parquet(self):
        for language in TRAINABLE:
            path = parquet_relative_path(split="train", config=language.code)
            self.assertEqual(path, f"train/{language.parquet}train.parquet")

    def test_telugu_train_is_rejected_by_the_loader(self):
        with self.assertRaises(ValueError):
            parquet_relative_path(split="train", config="te")

    def test_lookup_by_code_stem_and_prefix(self):
        self.assertEqual(get("ta").english, "Tamil")
        self.assertEqual(get("tam").english, "Tamil")
        self.assertEqual(from_prefix("ta-train-5k").code, "ta")
        self.assertIsNone(from_prefix("nonsense-train-5k"))
        self.assertIsNone(find("zz"))
        self.assertEqual(DEFAULT_CODE, "hi")


class SpeechHonestyTests(unittest.TestCase):
    """Never advertise dictation that has not been shown to work."""

    def test_nothing_claims_verified_without_evidence(self):
        claimed = [l.code for l in LANGUAGES if l.speech_status == SPEECH_VERIFIED]
        self.assertEqual(claimed, [], "a language may only be marked verified after a human confirms it")

    def test_languages_without_an_engine_offer_no_locale(self):
        for language in LANGUAGES:
            if language.speech_status == SPEECH_NONE:
                self.assertIsNone(language.speech_locale, language.code)
                self.assertTrue(language.note, f"{language.code} must say why voice is unavailable")
            else:
                self.assertEqual(language.speech_status, SPEECH_UNTESTED)
                self.assertTrue(language.speech_locale, language.code)
                self.assertRegex(language.speech_locale, r"^[a-z]{2}(-[A-Za-z]+)+$")


class ScriptHandlingTests(unittest.TestCase):
    """Normalisation, identity and chunking must hold for every script."""

    def test_normalisation_preserves_every_script(self):
        for code, text in SAMPLES.items():
            normalized = normalize_text(text)
            self.assertTrue(normalized, code)
            self.assertEqual(normalized, normalize_text(normalized), f"{code} is not idempotent")
            # No character class may be silently dropped.
            self.assertEqual(len(normalized.replace(" ", "")), len(text.replace(" ", "")), code)

    def test_normalisation_collapses_whitespace_in_every_script(self):
        for code, text in SAMPLES.items():
            noisy = f"  {text}   \n {text} "
            self.assertEqual(normalize_text(noisy), f"{text} {text}", code)

    def test_passage_ids_are_stable_per_language(self):
        for code, text in SAMPLES.items():
            normalized = normalize_text(text)
            self.assertEqual(passage_id(code, normalized), passage_id(code, normalized), code)
            self.assertTrue(passage_id(code, normalized).startswith("p_"))

    def test_identical_text_in_different_languages_gets_different_ids(self):
        """Language is part of the identity, so one language cannot collide with another."""
        shared = "गोवा"
        ids = {code: passage_id(code, shared) for code in ("hi", "mr", "ne", "sa")}
        self.assertEqual(len(set(ids.values())), len(ids), ids)

    def test_ids_are_unique_across_all_sample_scripts(self):
        ids = {passage_id(code, normalize_text(text)) for code, text in SAMPLES.items()}
        self.assertEqual(len(ids), len(SAMPLES))

    def test_document_chunking_works_in_every_script(self):
        for code, text in SAMPLES.items():
            page = " ".join([text] * 40)
            chunks = split_page(page, target=300)
            self.assertTrue(chunks, code)
            self.assertTrue(all(chunk.strip() for chunk in chunks), code)
            # Nothing may vanish: every chunk's content comes from the page.
            for chunk in chunks:
                self.assertIn(chunk[:20], page, code)

    def test_sentence_splitting_handles_danda_and_full_stop(self):
        self.assertEqual(len(sentences("गोवा राज्य है। यह छोटा है।")), 2)
        self.assertEqual(len(sentences("கேரளம் ஒரு மாநிலம். இது அழகானது.")), 2)


class IsolationTests(unittest.TestCase):
    """One language's index must never serve another's."""

    def test_corpus_source_keys_are_per_language(self):
        from src.rag.sources import CorpusSource

        class FakeIndex:
            class index:
                ntotal = 3
            def search(self, vector, top_k): return [[("p_x", 0.9)]]

        class FakeTexts:
            def texts(self, ids): return {"p_x": "text"}

        hindi = CorpusSource(FakeIndex(), FakeTexts(), key="corpus:hi", label="हिन्दी")
        tamil = CorpusSource(FakeIndex(), FakeTexts(), key="corpus:ta", label="தமிழ்")
        self.assertNotEqual(hindi.key, tamil.key)
        self.assertEqual({hindi.key, tamil.key}, {"corpus:hi", "corpus:ta"})

    def test_artifact_prefixes_do_not_collide_between_languages(self):
        prefixes = {f"{l.code}-train-5k" for l in TRAINABLE}
        self.assertEqual(len(prefixes), len(TRAINABLE))
        for prefix in prefixes:
            self.assertEqual(from_prefix(prefix).code, prefix.split("-")[0])


if __name__ == "__main__":
    unittest.main()
