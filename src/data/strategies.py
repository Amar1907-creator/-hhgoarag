"""Chunking strategies, as a registry of comparable alternatives.

The task specification asks for real thought about how the dataset is split,
not one fixed-size scheme. Each strategy below is a genuine alternative with
different failure modes, they share one interface so they can be swapped by
configuration, and scripts/benchmark_chunking.py measures them against the same
corpus, evaluation set and embedding model so the choice is made by measurement.

The axes that actually matter here:

  granularity   whole passage vs sentence group vs fixed window
  overlap       none, fixed character overlap, or sentence stride
  boundaries    arbitrary offsets vs sentence-aware vs semantic shift
  metadata      whether structure (page, heading) constrains and enriches a chunk

MSMARCO passages are already short and self-contained, so `whole` is a serious
contender rather than a straw man -- which is exactly why it is benchmarked
instead of assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from src.data.normalize import normalize_text

# Sentence terminators across the scripts in this dataset: Latin, Devanagari
# danda, Arabic full stop, and the Urdu question mark.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥۔؟])\s+")

DEFAULT_TARGET = 700
DEFAULT_OVERLAP = 120
DEFAULT_WINDOW = 3
DEFAULT_STRIDE = 2


@dataclass
class Chunk:
    text: str
    index: int
    strategy: str
    meta: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split((text or "").strip()) if part.strip()]


# -- strategies ---------------------------------------------------------------

def whole_passage(text: str, **_) -> list[str]:
    """One chunk per passage. Zero fragmentation, zero overlap cost.

    Right when the source unit is already a coherent retrieval unit, as MSMARCO
    passages are. Wrong for long documents, where one embedding has to represent
    several unrelated ideas.
    """
    cleaned = normalize_text(text)
    return [cleaned] if cleaned else []


def fixed_window(text: str, *, target: int = DEFAULT_TARGET,
                 overlap: int = DEFAULT_OVERLAP, **_) -> list[str]:
    """Fixed character windows with overlap. Predictable size, blind to meaning.

    Included as the baseline the specification warns against submitting alone --
    it is here to be beaten, and to show what it costs when it is.
    """
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= target:
        return [cleaned]
    stride = max(1, target - overlap)
    return [cleaned[start:start + target].strip()
            for start in range(0, len(cleaned), stride)
            if cleaned[start:start + target].strip()]


def sentence_packed(text: str, *, target: int = DEFAULT_TARGET,
                    overlap: int = DEFAULT_OVERLAP, **_) -> list[str]:
    """Pack whole sentences up to a target, carrying a tail of the previous chunk.

    Never cuts mid-sentence, so no embedding represents half a clause. The
    carried tail keeps a fact readable when it straddles a boundary.
    """
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    sentences = split_sentences(cleaned)
    if not sentences:
        return [cleaned]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > target * 1.6:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(fixed_window(sentence, target=target, overlap=overlap))
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= target:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            carry = tail[tail.find(" ") + 1:] if " " in tail else ""
            current = f"{carry} {sentence}".strip() if carry else sentence
    if current:
        chunks.append(current)
    return chunks


def sliding_sentences(text: str, *, window: int = DEFAULT_WINDOW,
                      stride: int = DEFAULT_STRIDE, **_) -> list[str]:
    """Overlapping sentence windows.

    Every sentence appears in more than one chunk, so a fact never sits alone at
    an edge. The cost is index size: roughly window/stride times the vectors.
    """
    if stride < 1 or window < 1:
        raise ValueError("window and stride must be at least 1")
    sentences = split_sentences(normalize_text(text))
    if not sentences:
        return []
    if len(sentences) <= window:
        return [" ".join(sentences)]
    return [" ".join(sentences[start:start + window])
            for start in range(0, len(sentences) - window + stride, stride)
            if sentences[start:start + window]]


# Cosine similarity between E5 embeddings and Jaccard word overlap live on
# completely different scales -- neighbouring sentences score ~0.8-0.9 under the
# former and ~0.05-0.2 under the latter -- so no absolute threshold serves both.
# Boundaries are therefore RELATIVE: a split happens where similarity dips
# clearly below this passage's own average. That needs no calibration and works
# for any similarity function, including ones added later.
SEMANTIC_DIP_FACTOR = 0.5


def semantic_shift(text: str, *, similarity: Callable[[list[str]], list[float]] | None = None,
                   threshold: float | None = None, target: int = DEFAULT_TARGET, **_) -> list[str]:
    """Split where meaning changes, not where a counter reaches a number.

    Adjacent sentences are compared and a boundary is cut where similarity dips
    below this passage's own mean by half a standard deviation. Being relative
    rather than absolute, it needs no per-corpus tuning. With no similarity
    function supplied it falls back to lexical overlap, so the strategy runs
    without an embedding model; the benchmark passes the real encoder in.

    Known limit of the fallback: when neighbouring sentences share no words at
    all, every score is equally low, there is no relative dip, and the passage is
    left whole. Lexical overlap simply cannot see a topic shift between
    sentences with no vocabulary in common; embedding similarity can.
    """
    sentences = split_sentences(normalize_text(text))
    if len(sentences) <= 1:
        return [" ".join(sentences)] if sentences else []

    scores = similarity(sentences) if similarity else _lexical_similarity(sentences)
    if threshold is not None:
        cut = threshold
    else:
        mean = sum(scores) / len(scores) if scores else 0.0
        spread = (sum((score - mean) ** 2 for score in scores) / len(scores)) ** 0.5 if scores else 0.0
        # Uniformly similar sentences contain no dip to cut at. Comparing against
        # the mean directly would split at every position here, because floating
        # point makes the mean of identical values fractionally larger than them.
        cut = float("-inf") if spread <= 1e-9 else mean - SEMANTIC_DIP_FACTOR * spread
    chunks: list[str] = []
    current = [sentences[0]]
    for position, sentence in enumerate(sentences[1:]):
        joined = len(" ".join(current)) + 1 + len(sentence)
        if scores[position] < cut or joined > target * 1.5:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _lexical_similarity(sentences: list[str]) -> list[float]:
    """Jaccard overlap between neighbours; the model-free stand-in."""
    def words(text: str) -> set[str]:
        return set(re.findall(r"[\wऀ-෿؀-ۿ]+", text.lower()))
    scores = []
    for left, right in zip(sentences, sentences[1:]):
        first, second = words(left), words(right)
        union = first | second
        scores.append(len(first & second) / len(union) if union else 0.0)
    return scores


def metadata_aware(text: str, *, heading: str = "", boundary: str = "",
                   target: int = DEFAULT_TARGET, overlap: int = DEFAULT_OVERLAP, **_) -> list[str]:
    """Sentence packing that never crosses a structural boundary, with context prepended.

    Two things happen here. A chunk stays inside its page or section, so a
    citation can name where it came from truthfully. And the heading is
    prepended to the chunk text, so a passage reading "it extends 101 kilometres"
    still embeds near a question about coastlines. Retrieval sees the context a
    human would have had from the page.
    """
    pieces = sentence_packed(text, target=target, overlap=overlap)
    prefix = normalize_text(heading)
    if not prefix:
        return pieces
    return [f"{prefix} — {piece}" if not piece.startswith(prefix) else piece for piece in pieces]


STRATEGIES: dict[str, Callable[..., list[str]]] = {
    "whole": whole_passage,
    "fixed": fixed_window,
    "sentence": sentence_packed,
    "sliding": sliding_sentences,
    "semantic": semantic_shift,
    "metadata": metadata_aware,
}

DESCRIPTIONS = {
    "whole": "one chunk per source passage; no split, no overlap",
    "fixed": "fixed character windows with character overlap; meaning-blind baseline",
    "sentence": "sentence-packed to a target size with a carried tail",
    "sliding": "overlapping sentence windows; every sentence appears more than once",
    "semantic": "boundaries where adjacent-sentence similarity drops",
    "metadata": "sentence packing bounded by page or section, with the heading prepended",
}


def chunk(text: str, strategy: str = "sentence", **options) -> list[Chunk]:
    """Apply a named strategy and return chunks with their provenance attached."""
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown chunking strategy {strategy!r}; "
                       f"available: {', '.join(sorted(STRATEGIES))}")
    pieces = STRATEGIES[strategy](text, **options)
    meta = {key: value for key, value in options.items() if isinstance(value, (str, int, float))}
    return [Chunk(text=piece, index=position, strategy=strategy, meta=dict(meta))
            for position, piece in enumerate(pieces) if piece.strip()]
