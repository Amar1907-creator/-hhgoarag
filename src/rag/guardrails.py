"""Guardrails: deciding what not to answer, before and after retrieval.

Three layers, all deterministic and all cheap enough to sit on the hot path:

1. Input screening rejects unsafe requests, prompt-injection attempts and
   degenerate input before any embedding is computed.
2. Retrieval screening (in evidence.py) abstains when nothing similar enough was
   found -- which is what "off-topic" means against an open-domain web corpus.
   A topic classifier would be the wrong tool: MSMARCO covers almost everything,
   so the honest test is whether the corpus actually contains support.
3. Output screening rejects answers whose content is not carried by the cited
   evidence, catching the case where a model paraphrases beyond its source.

No model is consulted for any of this. A guardrail that needs an LLM to run adds
the failure mode it exists to prevent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

MAX_QUESTION_CHARS = 1000
MIN_QUESTION_CHARS = 2
# Fraction of an answer's content words that must appear in the cited evidence.
# Measured on Hindi and English answers against real evidence: faithful answers
# (verbatim, paraphrase, multi-fact) scored 0.83-1.00, answers that drifted into
# outside knowledge scored 0.17-0.40. 0.60 sits in the gap with margin either
# side, so a terse paraphrase is not punished and a plausible invention is.
MIN_GROUNDING_OVERLAP = 0.60

BLOCK_UNSAFE = "unsafe_request"
BLOCK_INJECTION = "prompt_injection"
BLOCK_EMPTY = "empty_question"
BLOCK_TOO_LONG = "question_too_long"
BLOCK_NO_CONTENT = "no_recognisable_question"
BLOCK_UNGROUNDED = "answer_not_supported_by_evidence"
BLOCK_ENTITY_MISMATCH = "evidence_entity_mismatch"
ALLOWED = "allowed"

# First-line lexical screen in English and Devanagari. Deliberately narrow: it
# catches explicit requests for harm, and anything subtler is left to the
# grounding checks rather than pretending this is a safety classifier.
# Order matters: self-harm is checked before the general violence pattern so a
# person asking about themselves gets support rather than a flat refusal.
#
# Each pattern requires an ACTION, not merely a subject. "how to make a bomb" is
# refused; "what did the atomic bomb do to the war?" is a history question and
# must pass. A guardrail that bans topics instead of requests is a broken
# guardrail, and there is a test holding this line.
UNSAFE_PATTERNS = (
    (r"\b(how (to|do i|can i)|best way to|steps to)\b.{0,30}"
     r"\b(kill myself|killing myself|end my life|ending my life|commit suicide|self.?harm)\b",
     "self_harm"),
    (r"\b(i want to|i'm going to|im going to)\b.{0,20}\b(kill myself|end my life)\b", "self_harm"),
    (r"(आत्महत्या|खुदकुशी)\s*\S{0,12}\s*(कैसे|करूं|करने का तरीका)", "self_harm"),
    (r"\b(how to|how do i|steps to|way to|guide to)\b.{0,40}"
     r"\b(make|build|synthesi[sz]e|construct|manufacture)\b.{0,30}"
     r"\b(bomb|explosive|ied|nerve agent|sarin|ricin|meth|methamphetamine)\b", "weapons"),
    (r"\b(how to|best way to|how do i)\b.{0,25}\b(kill|murder|poison)\b.{0,20}"
     r"\b(someone|somebody|him|her|them|people|a person)\b", "violence"),
    (r"\b(child|minor|underage)\b.{0,25}\b(sexual|porn|nude|explicit)\b", "child_safety"),
    # Requires a making verb: "बम कैसे बनाएं" is refused, "परमाणु बम कैसे बदला" is not.
    (r"(बम|विस्फोटक|आईईडी)\s*\S{0,10}\s*(बनाने|बनाऊं|बनाना|बनाएं|बनाऊँ)", "weapons"),
)

INJECTION_PATTERNS = (
    r"ignore (all |your |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (your|the|all) (instructions|rules|system prompt)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (a|an|in) ",
    r"\bDAN\b mode",
    r"पिछले निर्देश.{0,10}(भूल|अनदेखा)",
)

SELF_HARM_MESSAGE = (
    "I can't help with that. If you are struggling, talking to someone helps — in India, "
    "Tele-MANAS is available free on 14416, any time."
)
UNSAFE_MESSAGE = "This request asks for information that could cause harm, so it was not searched."
INJECTION_MESSAGE = ("This looks like an attempt to change the system's instructions rather than a "
                     "question about the corpus, so it was not searched.")

# The Devanagari block contains the danda (U+0964), and the Arabic block its own
# punctuation, so a naive range match keeps "था।" as one token and stopword
# filtering silently fails on every Hindi sentence ending. Tokens are therefore
# stripped of Unicode punctuation by category, which works for every script.
_WORD = re.compile(r"[\u0900-\u097F\u0980-\u0DFF\u0600-\u06FF]+|[A-Za-z0-9]+")


def _strip_punctuation(token: str) -> str:
    return "".join(char for char in token if not unicodedata.category(char).startswith("P"))

# Words too common to prove anything about grounding.
_STOP = {
    "the", "is", "a", "an", "of", "to", "and", "in", "for", "on", "at", "it", "this", "that",
    "are", "was", "be", "with", "as", "by", "or", "from", "has", "have", "not",
    "है", "हैं", "का", "के", "की", "को", "में", "से", "और", "पर", "यह", "वह", "एक", "था", "थे",
}


@dataclass
class Verdict:
    allowed: bool = True
    code: str = ALLOWED
    message: str = ""
    category: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "code": self.code, "message": self.message,
                "category": self.category, **({"detail": self.detail} if self.detail else {})}


def content_words(text: str) -> set[str]:
    words = (_strip_punctuation(token).lower() for token in _WORD.findall(text or ""))
    return {word for word in words if word and word not in _STOP and len(word) > 1}


def screen_input(question: str) -> Verdict:
    """Decide whether a question should reach the retriever at all."""
    raw = (question or "").strip()
    if not raw:
        return Verdict(False, BLOCK_EMPTY, "Please enter a question.")
    if len(raw) > MAX_QUESTION_CHARS:
        return Verdict(False, BLOCK_TOO_LONG,
                       f"Questions are limited to {MAX_QUESTION_CHARS} characters.")

    normalized = unicodedata.normalize("NFKC", raw).lower()
    for pattern, category in UNSAFE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE | re.UNICODE):
            message = SELF_HARM_MESSAGE if category == "self_harm" else UNSAFE_MESSAGE
            return Verdict(False, BLOCK_UNSAFE, message, category=category)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE | re.UNICODE):
            return Verdict(False, BLOCK_INJECTION, INJECTION_MESSAGE, category="injection")

    if len(raw) < MIN_QUESTION_CHARS or not _WORD.search(raw):
        return Verdict(False, BLOCK_NO_CONTENT,
                       "That does not look like a question. Please rephrase it.")
    return Verdict()


def grounding_overlap(answer: str, evidence_texts: list[str]) -> float:
    """Share of the answer's content words that appear in the cited evidence."""
    answer_words = content_words(answer)
    if not answer_words:
        return 0.0
    evidence_words: set[str] = set()
    for text in evidence_texts:
        evidence_words |= content_words(text)
    return len(answer_words & evidence_words) / len(answer_words)


# A small, deliberately partial set of country names/demonyms in Hindi and
# English. Not a gazetteer -- it exists to catch the loudest, cheapest-to-catch
# failure mode seen in practice: retrieval returning evidence about a
# different country than the one named in the question (e.g. "भारत के पहले
# राष्ट्रपति" answered from a passage about American presidents, where "पहले
# राष्ट्रपति" matches lexically but the country does not). Absence from this
# list blocks nothing; presence only blocks when the question and the evidence
# name *different*, non-overlapping countries, so ordinary questions that name
# no country (the large majority of this corpus) are never touched by it.
COUNTRY_TERMS: dict[str, tuple[str, ...]] = {
    "india": ("भारत", "भारतीय", "hindustan", "india", "indian"),
    "usa": ("अमेरिका", "अमेरिकी", "संयुक्त राज्य", "यूएसए",
            "usa", "america", "american", "united states"),
    "uk": ("ब्रिटेन", "इंग्लैंड", "यूनाइटेड किंगडम", "britain", "england", "united kingdom"),
    "china": ("चीन", "चीनी", "china", "chinese"),
    "pakistan": ("पाकिस्तान", "पाकिस्तानी", "pakistan"),
    "russia": ("रूस", "रूसी", "सोवियत", "russia", "russian", "soviet"),
    "japan": ("जापान", "जापानी", "japan", "japanese"),
    "france": ("फ्रांस", "फ़्रांस", "france", "french"),
    "germany": ("जर्मनी", "germany", "german"),
    "canada": ("कनाडा", "canada", "canadian"),
    "australia": ("ऑस्ट्रेलिया", "australia", "australian"),
    "brazil": ("ब्राज़ील", "ब्राजील", "brazil", "brazilian"),
    "italy": ("इटली", "italy", "italian"),
    "spain": ("स्पेन", "spain", "spanish"),
    "nepal": ("नेपाल", "नेपाली", "nepal", "nepali"),
    "bangladesh": ("बांग्लादेश", "बांग्लादेशी", "bangladesh", "bangladeshi"),
}

ENTITY_MISMATCH_MESSAGE = (
    "The retrieved evidence appears to be about a different country than the "
    "one asked about, so the answer was withheld rather than guessed.")


def mentioned_countries(text: str) -> set[str]:
    """Which of a small set of country/demonym names appear in this text."""
    normalized = unicodedata.normalize("NFKC", (text or "")).lower()
    return {country for country, terms in COUNTRY_TERMS.items()
            if any(term in normalized for term in terms)}


def screen_relevance(question: str, evidence_texts: list[str]) -> Verdict:
    """Retrieval screening: refuse evidence that names a different country.

    Deliberately narrow, like the other guardrails here: it only fires when the
    question names a country/demonym AND the retrieved evidence names a
    *different* one with no overlap. It cannot catch every entity mismatch
    (person names, cities, organisations all pass through untouched), but it
    catches the clearest and costliest one -- for free, and before the model is
    even called, since generation is the expensive step this saves.
    """
    wanted = mentioned_countries(question)
    if not wanted:
        return Verdict()
    found = mentioned_countries(" ".join(evidence_texts))
    if not found or wanted & found:
        return Verdict()
    return Verdict(False, BLOCK_ENTITY_MISMATCH, ENTITY_MISMATCH_MESSAGE,
                   category="entity_mismatch",
                   detail={"question_countries": sorted(wanted), "evidence_countries": sorted(found)})


def screen_output(answer: str, cited_texts: list[str], *,
                  minimum: float = MIN_GROUNDING_OVERLAP) -> Verdict:
    """Reject an answer whose wording is not carried by the passages it cites.

    A model that paraphrases within its evidence keeps most of the content words;
    one that drifts into its own knowledge does not. This is a blunt instrument,
    and it is meant to be: it needs no model and cannot itself hallucinate.
    """
    if not answer.strip():
        return Verdict(False, BLOCK_UNGROUNDED, "The answer was empty.")
    if not cited_texts:
        return Verdict(False, BLOCK_UNGROUNDED, "The answer cited no evidence.")
    overlap = grounding_overlap(answer, cited_texts)
    if overlap < minimum:
        return Verdict(False, BLOCK_UNGROUNDED,
                       "The answer went beyond what the retrieved passages support, so it was "
                       "withheld.", category="low_overlap",
                       detail={"overlap": round(overlap, 3), "required": minimum})
    return Verdict(detail={"overlap": round(overlap, 3)})
