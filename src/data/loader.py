"""Pinned remote Parquet loader for MSMARCO-XI language files."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from src.data.remote_parquet import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_MAX_ATTEMPTS,
    CountingReader,
    stream_records,
)

DATASET_ID = "ai4bharat/MSMARCO-XI"
PINNED_REVISION = "bf5cdc1f26e581e519018e434db14edd1b77602b"

# Three-letter Parquet stem codes in the pinned revision inventory.
TRAIN_LANGUAGE_CODES = frozenset(
    {"asm", "ben", "guj", "hin", "kan", "mal", "mar", "nep", "ori", "pan", "san", "tam", "urd"}
)
VALIDATION_LANGUAGE_CODES = TRAIN_LANGUAGE_CODES | {"tel"}
VALIDATION_ONLY_LANGUAGE_CODES = frozenset({"tel"})

# Common two-letter aliases accepted by build/inspect scripts.
LANGUAGE_ALIASES: dict[str, str] = {
    "as": "asm",
    "asm": "asm",
    "bn": "ben",
    "ben": "ben",
    "gu": "guj",
    "guj": "guj",
    "hi": "hin",
    "hin": "hin",
    "kn": "kan",
    "kan": "kan",
    "ml": "mal",
    "mal": "mal",
    "mr": "mar",
    "mar": "mar",
    "ne": "nep",
    "nep": "nep",
    "or": "ori",
    "ori": "ori",
    "pa": "pan",
    "pan": "pan",
    "sa": "san",
    "san": "san",
    "ta": "tam",
    "tam": "tam",
    "te": "tel",
    "tel": "tel",
    "ur": "urd",
    "urd": "urd",
}


def _validate_revision(revision: str) -> None:
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
        raise ValueError("revision must be a 40-character immutable commit SHA")


def resolve_language_code(config: str) -> str:
    """Map a declared loader config to the Parquet language stem code."""
    if config == "default":
        raise ValueError(
            "loader config 'default' is not a language file; pass an explicit language such as 'hi' or 'hin'"
        )
    normalized = config.strip().lower()
    if normalized not in LANGUAGE_ALIASES:
        supported = ", ".join(sorted(LANGUAGE_ALIASES))
        raise ValueError(f"unsupported loader config {config!r}; supported values include: {supported}")
    return LANGUAGE_ALIASES[normalized]


def parquet_relative_path(*, split: str, config: str) -> str:
    """Return the repository-relative Parquet path for a split/language config."""
    language_code = resolve_language_code(config)
    if split == "train":
        if language_code in VALIDATION_ONLY_LANGUAGE_CODES:
            raise ValueError(
                "Telugu has validation data but no train file in the pinned revision; "
                "it cannot be used for a default train corpus"
            )
        if language_code not in TRAIN_LANGUAGE_CODES:
            raise ValueError(f"no train Parquet file exists for language code {language_code!r}")
        return f"train/{language_code}train.parquet"
    if split == "validation":
        if language_code not in VALIDATION_LANGUAGE_CODES:
            raise ValueError(f"no validation Parquet file exists for language code {language_code!r}")
        return f"validation/{language_code}val.parquet"
    raise ValueError(f"unsupported split {split!r}; expected 'train' or 'validation'")


def hub_parquet_path(*, split: str, config: str, revision: str) -> str:
    _validate_revision(revision)
    relative = parquet_relative_path(split=split, config=config)
    return f"datasets/{DATASET_ID}@{revision}/{relative}"


def _coerce_scalar(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list)):
        return value.item()
    return value


def _coerce_passages(value: Mapping[str, Any]) -> dict[str, list[Any]]:
    passages = {key: list(value[key]) for key in ("English_passages", "Translated_passages", "is_selected")}
    passages["is_selected"] = [int(_coerce_scalar(flag)) for flag in passages["is_selected"]]
    return passages


def coerce_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a PyArrow row mapping into schema-validated Python types."""
    record = dict(raw)
    record["query_id"] = int(_coerce_scalar(record["query_id"]))
    record["passages"] = _coerce_passages(record["passages"])
    return record


def records_from_batch(batch: Any) -> list[dict[str, Any]]:
    columns = batch.to_pydict()
    if not columns:
        return []
    row_count = len(next(iter(columns.values())))
    records: list[dict[str, Any]] = []
    for index in range(row_count):
        raw = {name: values[index] for name, values in columns.items()}
        records.append(coerce_record(raw))
    return records


def remote_handle_factory(*, split: str, config: str, revision: str,
                          block_size: int = DEFAULT_BLOCK_SIZE,
                          counting: bool = False) -> Callable[[], Any]:
    """Return a zero-argument opener for the pinned remote Parquet file.

    A factory rather than an open handle: the streaming reader reopens the
    source after a transient network fault, and needs to be able to do that
    without knowing anything about the Hub.
    """
    _validate_revision(revision)
    filesystem_path = hub_parquet_path(split=split, config=config, revision=revision)

    def opener() -> Any:
        try:
            from huggingface_hub import HfFileSystem
        except ImportError as exc:
            raise RuntimeError("install huggingface_hub and pyarrow to load MSMARCO-XI Parquet files") from exc
        filesystem = HfFileSystem()
        try:
            # readahead keeps at most one block plus lookahead in memory, unlike
            # the byte-range caches that grow toward whole-file retention.
            handle = filesystem.open(filesystem_path, "rb", block_size=block_size, cache_type="readahead")
        except TypeError:
            handle = filesystem.open(filesystem_path, "rb", block_size=block_size)
        return CountingReader(handle) if counting else handle

    return opener


def load_split(*, split: str, config: str, revision: str, batch_size: int = DEFAULT_BATCH_SIZE,
               start_offset: int = 0, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
               block_size: int = DEFAULT_BLOCK_SIZE, buffer_size: int = DEFAULT_BUFFER_SIZE,
               log: Callable[[str], None] | None = None) -> Iterable[dict]:
    """Stream records from a pinned remote language Parquet file.

    Bounded memory: batches are converted and released one at a time, and the
    Parquet reader is configured to read pages incrementally instead of
    materialising the file's single 3.7 GB row group. See remote_parquet.py.
    """
    return stream_records(
        open_handle=remote_handle_factory(split=split, config=config, revision=revision, block_size=block_size),
        coerce=coerce_record,
        batch_size=batch_size,
        start_offset=start_offset,
        max_attempts=max_attempts,
        buffer_size=buffer_size,
        log=log,
    )


def load_sample(*, split: str, config: str, revision: str, limit: int = 1) -> list[dict]:
    """Load a bounded prefix of records for loader validation."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    sample: list[dict] = []
    for record in load_split(split=split, config=config, revision=revision):
        sample.append(record)
        if len(sample) >= limit:
            break
    return sample
