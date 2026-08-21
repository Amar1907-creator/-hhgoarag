"""PDF text extraction with page numbers preserved.

Local only: pypdf is a pure-Python reader, nothing leaves the machine and no
service is contacted. The extractor's job is as much to REFUSE clearly as to
succeed: a scanned PDF that yields no text must say so rather than produce an
empty document that silently answers nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# A page with fewer than this many characters has no usable text on it.
MIN_CHARS_PER_PAGE = 12
# A document whose average page yields less than this is image-only in practice.
MIN_AVERAGE_CHARS = 20
MAX_PAGES = 2000
MAX_CHARS = 8_000_000
PDF_MAGIC = b"%PDF-"

REASON_OK = "ok"
REASON_NOT_PDF = "not_a_pdf"
REASON_CORRUPT = "corrupt_pdf"
REASON_ENCRYPTED = "encrypted_pdf"
REASON_EMPTY = "empty_pdf"
REASON_NO_TEXT = "no_extractable_text"

MESSAGES = {
    REASON_NOT_PDF: "That file is not a PDF.",
    REASON_CORRUPT: "The PDF could not be read; it looks damaged or incomplete.",
    REASON_ENCRYPTED: "The PDF is password protected, so its text cannot be read.",
    REASON_EMPTY: "The PDF has no pages.",
    REASON_NO_TEXT: ("No text could be extracted. This PDF appears to be scanned images "
                     "rather than text, and this build does not run OCR."),
}


@dataclass
class Page:
    number: int          # 1-based, as printed in a PDF viewer
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass
class ExtractionResult:
    ok: bool = False
    reason: str = REASON_CORRUPT
    pages: list[Page] = field(default_factory=list)
    page_count: int = 0
    text_pages: int = 0
    extracted_chars: int = 0
    truncated: bool = False

    @property
    def message(self) -> str:
        return MESSAGES.get(self.reason, "") if not self.ok else ""


def content_id(data: bytes) -> str:
    """Stable document ID from file content, so the same PDF is never indexed twice."""
    return "doc_" + hashlib.sha256(data).hexdigest()[:16]


def _clean(text: str) -> str:
    """Repair the line breaking that PDF extraction produces.

    Extracted PDF text arrives broken at layout line ends, which cuts sentences
    and words in half. Joining wrapped lines matters for retrieval: an embedding
    of a half-sentence matches poorly.
    """
    lines = [line.strip() for line in (text or "").replace("\r", "\n").split("\n")]
    out: list[str] = []
    for line in lines:
        if not line:
            out.append("")
            continue
        if out and out[-1] and not out[-1].endswith(("।", ".", "?", "!", ":", ";")):
            if out[-1].endswith("-"):
                out[-1] = out[-1][:-1] + line          # de-hyphenate across the break
            else:
                out[-1] = out[-1] + " " + line
        else:
            out.append(line)
    return "\n".join(part for part in out if part).strip()


def extract_pdf(source: bytes | Path | str, *, max_pages: int = MAX_PAGES,
                max_chars: int = MAX_CHARS) -> ExtractionResult:
    """Extract per-page text. Never raises for a bad file; returns a reason."""
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    if not data:
        return ExtractionResult(reason=REASON_EMPTY)
    if not data.lstrip()[:1024].startswith(PDF_MAGIC):
        return ExtractionResult(reason=REASON_NOT_PDF)

    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("install pypdf to read PDF documents") from exc

    import io
    try:
        reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:      # empty-password documents are common
                    return ExtractionResult(reason=REASON_ENCRYPTED)
            except Exception:
                return ExtractionResult(reason=REASON_ENCRYPTED)
        page_objects = reader.pages
        total = len(page_objects)
    except Exception:
        return ExtractionResult(reason=REASON_CORRUPT)

    if total == 0:
        return ExtractionResult(reason=REASON_EMPTY, page_count=0)

    pages: list[Page] = []
    extracted = 0
    truncated = False
    for number in range(1, min(total, max_pages) + 1):
        try:
            raw = page_objects[number - 1].extract_text() or ""
        except Exception:
            raw = ""                              # one broken page must not lose the rest
        text = _clean(raw)
        if len(text) >= MIN_CHARS_PER_PAGE:
            pages.append(Page(number=number, text=text))
            extracted += len(text)
        if extracted >= max_chars:
            truncated = True
            break
    if total > max_pages:
        truncated = True

    if not pages or extracted < max(MIN_AVERAGE_CHARS * min(total, max_pages), 50) // 2:
        return ExtractionResult(reason=REASON_NO_TEXT, page_count=total,
                                text_pages=len(pages), extracted_chars=extracted)

    return ExtractionResult(ok=True, reason=REASON_OK, pages=pages, page_count=total,
                            text_pages=len(pages), extracted_chars=extracted, truncated=truncated)
