"""The language registry: one place that knows what a language is.

Everything language-specific in HHGOARAG resolves through this table -- artifact
prefixes, UI labels, text direction, speech locale. Adding a language is a row
here plus a corpus build; there is no per-language code path anywhere.

The set below is exactly what the pinned MSMARCO-XI revision provides: 13
languages with train data, plus Telugu which has validation data but no train
file and therefore cannot have a corpus built from this revision.

Speech locales are the BCP-47 tags a browser would be asked to use. A tag here
is an ATTEMPT, not a claim: the browser is the only authority on whether it can
recognise a language, so the interface probes at runtime and reports failure
honestly rather than advertising support that may not exist.
"""

from __future__ import annotations

from dataclasses import dataclass

SPEECH_UNTESTED = "untested"      # a locale exists to try; no human has confirmed it
SPEECH_NONE = "unavailable"       # no browser speech engine is known for this language
SPEECH_VERIFIED = "verified"      # a human confirmed dictation works


@dataclass(frozen=True)
class Language:
    code: str                  # loader config and artifact prefix, e.g. "hi"
    parquet: str               # Parquet stem in the pinned revision, e.g. "hin"
    english: str
    native: str
    script: str
    direction: str = "ltr"
    speech_locale: str | None = None
    speech_status: str = SPEECH_UNTESTED
    has_train: bool = True
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.native} · {self.english}"

    def placeholder(self) -> str:
        return PLACEHOLDERS.get(self.code, f"Ask your question in {self.english}…")

    def to_dict(self) -> dict:
        return {"code": self.code, "english": self.english, "native": self.native,
                "script": self.script, "direction": self.direction,
                "speech_locale": self.speech_locale, "speech_status": self.speech_status,
                "has_train": self.has_train, "placeholder": self.placeholder(),
                "label": self.label, "note": self.note}


# Native-script prompts. Written per language rather than machine-translated so
# the interface reads as one product to a speaker of each.
PLACEHOLDERS = {
    "as": "আপোনাৰ প্ৰশ্ন লিখক…",
    "bn": "আপনার প্রশ্ন লিখুন…",
    "gu": "તમારો પ્રશ્ન લખો…",
    "hi": "अपना प्रश्न हिंदी में लिखें…",
    "kn": "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಬರೆಯಿರಿ…",
    "ml": "നിങ്ങളുടെ ചോദ്യം എഴുതുക…",
    "mr": "तुमचा प्रश्न लिहा…",
    "ne": "आफ्नो प्रश्न लेख्नुहोस्…",
    "or": "ଆପଣଙ୍କ ପ୍ରଶ୍ନ ଲେଖନ୍ତୁ…",
    "pa": "ਆਪਣਾ ਸਵਾਲ ਲਿਖੋ…",
    "sa": "भवतः प्रश्नं लिखतु…",
    "ta": "உங்கள் கேள்வியை எழுதுங்கள்…",
    "te": "మీ ప్రశ్న రాయండి…",
    "ur": "اپنا سوال لکھیں…",
}

LANGUAGES: tuple[Language, ...] = (
    Language("as", "asm", "Assamese", "অসমীয়া", "Bengali",
             speech_locale=None, speech_status=SPEECH_NONE,
             note="no browser speech engine known for Assamese"),
    Language("bn", "ben", "Bengali", "বাংলা", "Bengali", speech_locale="bn-IN"),
    Language("gu", "guj", "Gujarati", "ગુજરાતી", "Gujarati", speech_locale="gu-IN"),
    Language("hi", "hin", "Hindi", "हिन्दी", "Devanagari", speech_locale="hi-IN"),
    Language("kn", "kan", "Kannada", "ಕನ್ನಡ", "Kannada", speech_locale="kn-IN"),
    Language("ml", "mal", "Malayalam", "മലയാളം", "Malayalam", speech_locale="ml-IN"),
    Language("mr", "mar", "Marathi", "मराठी", "Devanagari", speech_locale="mr-IN"),
    Language("ne", "nep", "Nepali", "नेपाली", "Devanagari", speech_locale="ne-NP"),
    Language("or", "ori", "Odia", "ଓଡ଼ିଆ", "Odia",
             speech_locale=None, speech_status=SPEECH_NONE,
             note="no browser speech engine known for Odia"),
    Language("pa", "pan", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi", speech_locale="pa-Guru-IN"),
    Language("sa", "san", "Sanskrit", "संस्कृतम्", "Devanagari",
             speech_locale=None, speech_status=SPEECH_NONE,
             note="no browser speech engine known for Sanskrit"),
    Language("ta", "tam", "Tamil", "தமிழ்", "Tamil", speech_locale="ta-IN"),
    Language("ur", "urd", "Urdu", "اردو", "Arabic", direction="rtl", speech_locale="ur-IN"),
    Language("te", "tel", "Telugu", "తెలుగు", "Telugu", speech_locale="te-IN",
             has_train=False,
             note="validation data only in the pinned revision; no train corpus can be built"),
)

BY_CODE = {language.code: language for language in LANGUAGES}
BY_PARQUET = {language.parquet: language for language in LANGUAGES}
TRAINABLE = tuple(language for language in LANGUAGES if language.has_train)
DEFAULT_CODE = "hi"


def get(code: str) -> Language:
    """Resolve a language by config code or Parquet stem."""
    key = (code or "").strip().lower()
    if key in BY_CODE:
        return BY_CODE[key]
    if key in BY_PARQUET:
        return BY_PARQUET[key]
    raise KeyError(f"unknown language {code!r}; known: {', '.join(sorted(BY_CODE))}")


def find(code: str) -> Language | None:
    try:
        return get(code)
    except KeyError:
        return None


def from_prefix(prefix: str) -> Language | None:
    """Recover the language from an artifact prefix such as 'hi-train-5k'."""
    return find(prefix.split("-", 1)[0]) if prefix else None
