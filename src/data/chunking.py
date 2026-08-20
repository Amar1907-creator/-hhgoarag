"""Deterministic chunking candidates; policy selection happens after benchmarking."""

from __future__ import annotations

import re
from collections.abc import Callable

_SENTENCES = re.compile(r"(?<=[.!?।])\s+", flags=re.UNICODE)


def whole_passage(text: str) -> list[str]:
    return [text] if text else []


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCES.split(text.strip()) if part.strip()]


def sentence_windows(text: str, window_size: int = 3, overlap: int = 1) -> list[str]:
    """Return overlapping sentence windows; validates settings to avoid infinite loops."""
    if window_size < 1 or overlap < 0 or overlap >= window_size:
        raise ValueError("require window_size >= 1 and 0 <= overlap < window_size")
    parts = sentences(text)
    step = window_size - overlap
    return [" ".join(parts[start : start + window_size]) for start in range(0, len(parts), step)]


def semantic_chunks(text: str, boundary_fn: Callable[[str], list[int]]) -> list[str]:
    """Optional controlled semantic splitting; caller supplies a benchmarked boundary model."""
    points = sorted(set(boundary_fn(text)))
    if any(point <= 0 or point >= len(text) for point in points):
        raise ValueError("semantic boundaries must be inside the text")
    starts_and_ends = [0, *points, len(text)]
    return [text[start:end].strip() for start, end in zip(starts_and_ends, starts_and_ends[1:]) if text[start:end].strip()]
