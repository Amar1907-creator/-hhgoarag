"""Page-aware chunking.

A chunk never spans two pages. That costs a little packing efficiency and buys
the thing the product actually promises: every retrieved chunk can name the page
it came from, so a citation points at a page a human can turn to.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.data.normalize import normalize_text

TARGET_CHARS = 700
OVERLAP_CHARS = 120
MIN_CHUNK_CHARS = 40

_BOUNDARY = re.compile(r"(?<=[।.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    text: str
    index_on_page: int

    def to_dict(self) -> dict:
        return asdict(self)


def split_page(text: str, *, target: int = TARGET_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split one page on sentence boundaries, falling back to hard slices."""
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= target:
        return [text]

    sentences = [part.strip() for part in _BOUNDARY.split(text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > target * 1.6:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), target - overlap):
                piece = sentence[start:start + target].strip()
                if piece:
                    chunks.append(piece)
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
    return [chunk for chunk in chunks if len(chunk) >= MIN_CHUNK_CHARS] or [text[:target]]


def chunk_pages(pages, document_id: str, document_name: str, **kwargs) -> list[Chunk]:
    """Turn extracted pages into page-attributed chunks, in reading order."""
    chunks: list[Chunk] = []
    for page in pages:
        for position, piece in enumerate(split_page(page.text, **kwargs)):
            chunks.append(Chunk(
                chunk_id=f"{document_id}_p{page.number:04d}_c{position:03d}",
                document_id=document_id, document_name=document_name,
                page=page.number, text=piece, index_on_page=position))
    return chunks
