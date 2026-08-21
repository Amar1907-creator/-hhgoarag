"""Bounded-memory, retrying remote Parquet access for pinned MSMARCO-XI files.

Why this module exists
----------------------
Every train language file in the pinned revision is a SINGLE row group
(hintrain.parquet: 3.72 GB, 778,638 rows, 1 row group). PyArrow's defaults
materialise a whole column chunk per read, so `ParquetFile(handle)` followed by
`iter_batches(batch_size=256)` fetches and decodes the ENTIRE row group before
yielding the first record, no matter how small the batch size or the --limit.

Measured on a 307 MB / 20,000-row / 1-row-group file with the identical nested
schema, reading only the first 5 records:

    defaults (what the old loader did)      307.5 MB read (100%)   +359 MB RSS
    pre_buffer=False                        307.5 MB read (100%)   +18 MB RSS
    pre_buffer=False, buffer_size=1 MiB      10.7 MB read (3.5%)   +0 MB RSS

Scaled to the real file that is the difference between pulling 3.72 GB into
~4 GB of RSS over one long-lived HTTPS connection -- which is what produced the
~3 GB RSS failure and the us.aws.cdn.hf.co ReadTimeouts -- and pulling a few
tens of MB in small range requests at flat memory.

So: pre_buffer=False plus an explicit non-zero buffer_size is mandatory here,
and neither may be dropped as an "optimisation".
"""

from __future__ import annotations

import os
import resource
import ssl
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

# PyArrow reads pages through a buffered stream of this size. buffer_size=0 (the
# default) means "read the whole column chunk", which is the bug above.
DEFAULT_BUFFER_SIZE = 1 << 20
# fsspec range-request granularity for the remote handle.
DEFAULT_BLOCK_SIZE = 8 << 20
# Records converted to Python per batch. Bounded memory, not bounded I/O.
DEFAULT_BATCH_SIZE = 64

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 2.0
DEFAULT_MAX_DELAY = 60.0

# Substrings identifying a retryable network fault.
_TRANSIENT_MARKERS = (
    "readtimeout", "connecttimeout", "connectionerror", "connectionreset",
    "chunkedencodingerror", "incompleteread", "protocolerror", "remotedisconnected",
    "timeout", "temporarily unavailable", "connection aborted", "connection reset",
    "502", "503", "504", "429", "too many requests", "bad gateway", "service unavailable",
)
# Substrings identifying a fault that retrying cannot fix.
_FATAL_MARKERS = (
    "certificate verify failed", "certificate_verify_failed", "sslcertverificationerror",
    "self signed certificate", "unable to get local issuer",
    "401", "403", "404", "unauthorized", "forbidden", "not found",
    "arrownotimplemented", "arrowinvalid",
)


def peak_rss_mb() -> float:
    """Peak resident set size in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1_048_576 if os.uname().sysname == "Darwin" else value / 1024


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def is_certificate_error(exc: BaseException) -> bool:
    """Certificate-chain failures are configuration, not weather."""
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    text = _describe(exc)
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        text += " " + _describe(cause)
    return any(marker in text for marker in
               ("certificate verify failed", "certificate_verify_failed", "unable to get local issuer"))


def is_transient(exc: BaseException) -> bool:
    """True only for faults where waiting and retrying is the right response."""
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        return False
    if is_certificate_error(exc):
        return False
    text = _describe(exc)
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        text += " " + _describe(cause)
    if any(marker in text for marker in _FATAL_MARKERS):
        return False
    if isinstance(exc, TimeoutError):
        return True
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def backoff_delay(attempt: int, *, base: float = DEFAULT_BASE_DELAY, maximum: float = DEFAULT_MAX_DELAY) -> float:
    """Deterministic exponential backoff; attempt is 1-based."""
    return min(maximum, base * (2 ** (attempt - 1)))


class CountingReader:
    """File wrapper that counts bytes actually pulled.

    Used by preflight to prove the reader performs bounded range reads rather
    than dragging the whole object down, and by builds to report transfer.
    """

    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.bytes_read = 0
        self.read_calls = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        # PyArrow's Python-file adapter probes this before every open.
        return self._closed or bool(getattr(self.handle, "closed", False))

    def writable(self) -> bool:
        return False

    def flush(self) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        block = self.handle.read(size)
        self.bytes_read += len(block)
        self.read_calls += 1
        return block

    def readinto(self, buffer) -> int:
        if hasattr(self.handle, "readinto"):
            count = self.handle.readinto(buffer) or 0
        else:
            block = self.handle.read(len(buffer))
            count = len(block)
            buffer[:count] = block
        self.bytes_read += count
        self.read_calls += 1
        return count

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.handle.seek(offset, whence)

    def tell(self) -> int:
        return self.handle.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._closed = True
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def open_parquet(handle: Any, *, buffer_size: int = DEFAULT_BUFFER_SIZE):
    """Open a ParquetFile in the only configuration that reads incrementally."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("install pyarrow to read MSMARCO-XI Parquet files") from exc
    if buffer_size <= 0:
        raise ValueError("buffer_size must be > 0; 0 makes PyArrow read whole column chunks")
    return pq.ParquetFile(handle, pre_buffer=False, buffer_size=buffer_size)


def iter_batches(parquet_file, *, batch_size: int = DEFAULT_BATCH_SIZE,
                 columns: Sequence[str] | None = None, skip_records: int = 0) -> Iterator[tuple[int, Any]]:
    """Yield (absolute_start_offset, batch), skipping a prefix without decoding it.

    Skipped batches are dropped before any Python conversion, so resuming costs
    bytes off the wire but almost no CPU or memory.
    """
    if skip_records < 0:
        raise ValueError("skip_records must not be negative")
    offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None):
        rows = batch.num_rows
        if offset + rows <= skip_records:
            offset += rows
            del batch
            continue
        if offset < skip_records:
            # Partial overlap: keep only the tail of this batch.
            keep = offset + rows - skip_records
            trimmed = batch.slice(rows - keep)
            del batch
            yield skip_records, trimmed
            offset += rows
            continue
        yield offset, batch
        offset += rows


def stream_records(
    *,
    open_handle: Callable[[], Any],
    coerce: Callable[[dict], dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
    columns: Sequence[str] | None = None,
    start_offset: int = 0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] | None = None,
) -> Iterator[dict]:
    """Stream records with bounded memory, reconnecting on transient faults.

    On a transient fault the source is reopened and the already-delivered prefix
    is skipped, so a dropped connection costs a re-read of the prefix rather
    than the whole job. Certificate and permission errors are raised straight
    away: retrying them only wastes time.
    """
    offset = start_offset
    attempt = 0
    while True:
        try:
            handle = open_handle()
            try:
                parquet_file = open_parquet(handle, buffer_size=buffer_size)
                for _, batch in iter_batches(parquet_file, batch_size=batch_size,
                                             columns=columns, skip_records=offset):
                    columns_dict = batch.to_pydict()
                    rows = batch.num_rows
                    del batch
                    for index in range(rows):
                        yield coerce({name: values[index] for name, values in columns_dict.items()})
                        offset += 1
                    del columns_dict
                return
            finally:
                try:
                    handle.close()
                except Exception:  # pragma: no cover - close is best effort
                    pass
        except (KeyboardInterrupt, GeneratorExit):
            raise
        except Exception as exc:
            if is_certificate_error(exc):
                raise RuntimeError(
                    "TLS certificate verification failed against the Hugging Face CDN. This is a "
                    "local trust-store problem, not a transient fault, and is not retried. On macOS "
                    "run the 'Install Certificates.command' bundled with your Python install, or "
                    "point SSL_CERT_FILE at a valid CA bundle (e.g. `python3 -m pip install certifi` "
                    "then `export SSL_CERT_FILE=$(python3 -m certifi)`). SSL verification is never "
                    "disabled by this tool."
                ) from exc
            attempt += 1
            if not is_transient(exc) or attempt >= max_attempts:
                raise
            delay = backoff_delay(attempt)
            if log:
                log(f"transient read failure at record {offset:,} "
                    f"({type(exc).__name__}); retry {attempt}/{max_attempts - 1} in {delay:.0f}s")
            sleep(delay)
