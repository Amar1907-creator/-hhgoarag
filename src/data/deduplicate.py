"""Disk-backed exact passage deduplication."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def passage_id(language: str, normalized_text: str) -> str:
    """Stable ID for a language/text pair; it does not depend on ingestion order."""
    digest = hashlib.sha256(f"{language}\0{normalized_text}".encode("utf-8")).hexdigest()
    return f"p_{digest}"


class ExactDeduplicator:
    """SQLite-backed set avoids retaining the complete unique corpus in RAM.

    The `seen` table outlives the process, so a fresh build MUST pass reset=True.
    Reusing stale state against a truncated corpus file silently produces an
    empty or partial corpus: every passage is already "seen" and nothing is
    written. Only an explicitly appending, multi-part build should keep state.
    """

    def __init__(self, database: Path, *, reset: bool = False) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA journal_mode=WAL")
        if reset:
            # DROP rather than deleting the file: the table is recreated empty and
            # the artifact path stays stable for callers and manifests.
            self.connection.execute("DROP TABLE IF EXISTS seen")
        self.connection.execute("CREATE TABLE IF NOT EXISTS seen (passage_id TEXT PRIMARY KEY)")
        self.connection.commit()

    def add(self, stable_id: str) -> bool:
        cursor = self.connection.execute("INSERT OR IGNORE INTO seen(passage_id) VALUES (?)", (stable_id,))
        return cursor.rowcount == 1

    def count(self) -> int:
        """Number of retained unique passage IDs; reported in build manifests."""
        return int(self.connection.execute("SELECT COUNT(*) FROM seen").fetchone()[0])

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()
