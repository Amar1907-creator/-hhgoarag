"""Conservative text normalization used only for retrieval keys."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Return a deterministic, Unicode-safe form without changing case or script."""
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()
