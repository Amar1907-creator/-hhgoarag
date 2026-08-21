"""Tests for the bounded-memory, retrying Parquet ingestion path.

The fixture is a real Parquet file written with the EXACT nested schema of
train/hintrain.parquet in the pinned revision, in a single row group, so the
decoding and bounded-read behaviour under test is the behaviour the real file
will exercise. It is a schema fixture for mechanics, never a stand-in for
MSMARCO-XI content: no quality claim is derived from it.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW = True
except ImportError:
    PYARROW = False

from src.data.remote_parquet import (
    CountingReader,
    backoff_delay,
    is_certificate_error,
    is_transient,
    iter_batches,
    open_parquet,
    stream_records,
)
from src.data.schema import validate_record
from src.data.normalize import normalize_text
from src.data.deduplicate import passage_id

REVISION = "bf5cdc1f26e581e519018e434db14edd1b77602b"


def msmarco_schema():
    return pa.schema([
        ("source_lang", pa.string()), ("target_lang", pa.string()),
        ("meta", pa.struct([("frequency_penalty", pa.int64()), ("max_tokens", pa.int64()),
                            ("model_name", pa.string()), ("presence_penalty", pa.int64()),
                            ("temperature", pa.int64()), ("top_p", pa.int64())])),
        ("Answer", pa.string()), ("query_id", pa.int64()), ("query_type", pa.string()),
        ("passages", pa.struct([("English_passages", pa.list_(pa.string())),
                                ("Translated_passages", pa.list_(pa.string())),
                                ("is_selected", pa.list_(pa.int64()))])),
        ("Eng_Query", pa.string()), ("Eng_Answer", pa.string()), ("query", pa.string()),
    ])


def write_fixture(path: Path, rows: int = 400, passages: int = 10, passage_chars: int = 400,
                  malformed_at: int | None = None) -> Path:
    """One row group, matching the pinned file's geometry."""
    import random
    devanagari = "अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह "
    def text(n, seed):
        rng = random.Random(seed)
        return "".join(rng.choice(devanagari) for _ in range(n))
    data = {"source_lang": [], "target_lang": [], "meta": [], "Answer": [], "query_id": [],
            "query_type": [], "passages": [], "Eng_Query": [], "Eng_Answer": [], "query": []}
    for n in range(rows):
        count = passages if n != malformed_at else passages - 3  # misaligned passage lists
        data["source_lang"].append("eng_Latn")
        data["target_lang"].append("hin_Deva")
        data["meta"].append({"frequency_penalty": 0, "max_tokens": 2048, "model_name": "gpt",
                             "presence_penalty": 0, "temperature": 0, "top_p": 1})
        data["Answer"].append(text(60, n))
        data["query_id"].append(1_000_000 + n)
        data["query_type"].append("DESCRIPTION")
        data["passages"].append({
            "English_passages": [f"english passage {n}-{i}" for i in range(passages)],
            "Translated_passages": [text(passage_chars, n * 31 + i) for i in range(passages)],
            "is_selected": [1 if i == 0 else 0 for i in range(count)]})
        data["Eng_Query"].append(f"query {n}")
        data["Eng_Answer"].append(f"answer {n}")
        data["query"].append(text(40, n))
    pq.write_table(pa.Table.from_pydict(data, schema=msmarco_schema()), path,
                   compression="snappy", row_group_size=rows)
    return path


class TransientClassificationTests(unittest.TestCase):
    """Retrying a certificate failure only wastes the user's time."""

    class ReadTimeout(Exception): pass
    class SSLCertVerificationError(Exception): pass
    class ArrowNotImplementedError(Exception): pass

    def test_network_faults_are_transient(self):
        for exc in (self.ReadTimeout("timed out"), TimeoutError("timeout"),
                    OSError("Connection reset by peer"), Exception("503 Service Unavailable"),
                    Exception("429 Too Many Requests")):
            self.assertTrue(is_transient(exc), exc)

    def test_certificate_failures_are_fatal(self):
        exc = self.SSLCertVerificationError("certificate verify failed: unable to get local issuer certificate")
        self.assertTrue(is_certificate_error(exc))
        self.assertFalse(is_transient(exc))

    def test_structural_failures_are_fatal(self):
        for exc in (self.ArrowNotImplementedError("nested data conversions not implemented"),
                    Exception("401 Unauthorized"), Exception("404 Not Found")):
            self.assertFalse(is_transient(exc), exc)

    def test_interrupts_are_never_retried(self):
        self.assertFalse(is_transient(KeyboardInterrupt()))

    def test_backoff_is_exponential_and_capped(self):
        self.assertEqual([backoff_delay(n, base=2, maximum=60) for n in range(1, 7)],
                         [2, 4, 8, 16, 32, 60])


@unittest.skipUnless(PYARROW, "pyarrow is required")
class NestedDecodingTests(unittest.TestCase):
    """The datasets streaming path fails on this nested structure; this one must not."""

    def test_nested_passages_decode_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=40)
            records = list(stream_records(open_handle=lambda: open(path, "rb"),
                                          coerce=lambda row: row, batch_size=8))
            self.assertEqual(len(records), 40)
            first = records[0]
            self.assertEqual(first["query_id"], 1_000_000)
            self.assertEqual(first["target_lang"], "hin_Deva")
            self.assertEqual(len(first["passages"]["Translated_passages"]), 10)
            self.assertEqual(len(first["passages"]["English_passages"]), 10)
            self.assertEqual(first["passages"]["is_selected"][0], 1)
            self.assertEqual(first["meta"]["model_name"], "gpt")

    def test_decoded_record_survives_the_whole_pipeline(self):
        """Nested decode -> schema validation -> normalization -> stable passage ID."""
        from src.data.loader import coerce_record
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=5)
            records = list(stream_records(open_handle=lambda: open(path, "rb"),
                                          coerce=coerce_record, batch_size=2))
            for record in records:
                result = validate_record(record)
                self.assertTrue(result.valid, result.errors)
                self.assertIsInstance(record["query_id"], int)
                for text in record["passages"]["Translated_passages"]:
                    normalized = normalize_text(text)
                    self.assertTrue(normalized)
                    self.assertTrue(passage_id(record["target_lang"], normalized).startswith("p_"))

    def test_malformed_record_is_detected_not_crashed_on(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=10, malformed_at=4)
            records = list(stream_records(open_handle=lambda: open(path, "rb"),
                                          coerce=lambda row: row, batch_size=4))
            results = [validate_record(record) for record in records]
            self.assertFalse(results[4].valid)
            self.assertIn("passage list lengths differ", results[4].errors[0])
            self.assertTrue(all(results[n].valid for n in range(10) if n != 4))


@unittest.skipUnless(PYARROW, "pyarrow is required")
class BoundedReadTests(unittest.TestCase):
    """Reading a prefix must not drag down the whole object."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        # Large enough that one row group spans many pages, like the real file.
        cls.big = write_fixture(Path(cls._directory.name) / "big.parquet", rows=1500, passage_chars=700)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def read_bytes_for(self, path, limit, **kwargs):
        reader = CountingReader(open(path, "rb"))
        parquet_file = open_parquet(reader, **kwargs)
        taken = 0
        for _, batch in iter_batches(parquet_file, batch_size=8):
            taken += batch.num_rows
            del batch
            if taken >= limit:
                break
        reader.close()
        return reader.bytes_read

    def test_prefix_read_is_far_smaller_than_the_whole_file(self):
        """The regression that cost the 3 GB RSS failure: PyArrow's defaults read
        the entire single row group before yielding record one."""
        total = self.big.stat().st_size
        prefix = self.read_bytes_for(self.big, 8)
        whole = self.read_bytes_for(self.big, 10 ** 9)
        self.assertLess(prefix, total * 0.5,
                        f"reading 8 records pulled {prefix:,} of {total:,} bytes")
        self.assertLess(prefix, whole * 0.5,
                        f"prefix read {prefix:,} vs full read {whole:,}")

    def test_pyarrow_defaults_would_read_everything(self):
        """Documents why buffer_size is not optional."""
        import pyarrow.parquet as pq
        reader = CountingReader(open(self.big, "rb"))
        parquet_file = pq.ParquetFile(reader)  # PyArrow defaults
        for batch in parquet_file.iter_batches(batch_size=8):
            del batch
            break
        default_bytes = reader.bytes_read
        reader.close()
        bounded = self.read_bytes_for(self.big, 8)
        self.assertGreater(default_bytes, bounded * 2,
                           f"defaults {default_bytes:,} vs bounded {bounded:,}")

    def test_zero_buffer_size_is_rejected(self):
        """buffer_size=0 is PyArrow's default and reads whole column chunks."""
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=10)
            with open(path, "rb") as handle:
                with self.assertRaises(ValueError):
                    open_parquet(handle, buffer_size=0)
            del directory

    def test_batches_are_released_between_iterations(self):
        import gc, sys
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=120)
            with open(path, "rb") as handle:
                parquet_file = open_parquet(handle)
                live = []
                for _, batch in iter_batches(parquet_file, batch_size=16):
                    live.append(sys.getrefcount(batch))
                    del batch
                gc.collect()
            self.assertTrue(all(count <= 3 for count in live), live)


@unittest.skipUnless(PYARROW, "pyarrow is required")
class SkipAndResumeTests(unittest.TestCase):

    def test_skip_records_matches_the_full_stream_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=50)
            full = [r["query_id"] for r in stream_records(
                open_handle=lambda: open(path, "rb"), coerce=lambda r: r, batch_size=7)]
            tail = [r["query_id"] for r in stream_records(
                open_handle=lambda: open(path, "rb"), coerce=lambda r: r, batch_size=7, start_offset=23)]
            self.assertEqual(tail, full[23:])

    def test_skip_inside_a_batch_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=40)
            with open(path, "rb") as handle:
                parquet_file = open_parquet(handle)
                offsets = [(offset, batch.num_rows) for offset, batch in
                           iter_batches(parquet_file, batch_size=16, skip_records=20)]
            self.assertEqual(offsets[0][0], 20)
            self.assertEqual(sum(rows for _, rows in offsets), 20)


@unittest.skipUnless(PYARROW, "pyarrow is required")
class RetryTests(unittest.TestCase):
    """A dropped connection must cost a reconnect, not the whole job."""

    class ReadTimeout(Exception): pass

    def flaky_opener(self, path, fail_after_reads, failures):
        state = {"opens": 0, "failed": 0}

        class FlakyHandle:
            def __init__(self, inner): self.inner = inner; self.reads = 0
            @property
            def closed(self): return self.inner.closed
            def writable(self): return False
            def flush(self): pass
            def read(self, size=-1):
                self.reads += 1
                if self.reads > fail_after_reads and state["failed"] < failures:
                    state["failed"] += 1
                    raise RetryTests.ReadTimeout("read timed out against us.aws.cdn.hf.co")
                return self.inner.read(size)
            def readinto(self, buffer):
                block = self.read(len(buffer))
                buffer[: len(block)] = block
                return len(block)
            def seek(self, o, w=0): return self.inner.seek(o, w)
            def tell(self): return self.inner.tell()
            def seekable(self): return True
            def readable(self): return True
            def close(self): self.inner.close()

        def opener():
            state["opens"] += 1
            return FlakyHandle(open(path, "rb"))
        return opener, state

    def test_records_are_delivered_exactly_once_across_a_reconnect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=120)
            expected = [r["query_id"] for r in stream_records(
                open_handle=lambda: open(path, "rb"), coerce=lambda r: r, batch_size=8)]
            opener, state = self.flaky_opener(path, fail_after_reads=6, failures=1)
            slept, messages = [], []
            got = [r["query_id"] for r in stream_records(
                open_handle=opener, coerce=lambda r: r, batch_size=8,
                sleep=slept.append, log=messages.append)]
            self.assertEqual(got, expected)
            self.assertEqual(len(got), len(set(got)), "a reconnect must not duplicate records")
            self.assertGreaterEqual(state["opens"], 2, "the source should have been reopened")
            self.assertTrue(slept and slept[0] > 0)
            self.assertTrue(any("transient read failure" in m for m in messages))

    def test_retries_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_fixture(Path(directory) / "f.parquet", rows=60)
            opener, _ = self.flaky_opener(path, fail_after_reads=3, failures=99)
            with self.assertRaises(RetryTests.ReadTimeout):
                list(stream_records(open_handle=opener, coerce=lambda r: r,
                                    batch_size=8, max_attempts=3, sleep=lambda _: None))

    def test_certificate_failure_is_not_retried_and_explains_itself(self):
        """TLS verification fails at connect time. Retrying cannot fix a trust
        store, so it must surface immediately with a remedy."""
        class SSLCertVerificationError(Exception):
            pass

        def opener():
            raise SSLCertVerificationError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                "unable to get local issuer certificate (_ssl.c:1006)")

        slept = []
        with self.assertRaises(RuntimeError) as caught:
            list(stream_records(open_handle=opener, coerce=lambda r: r, sleep=slept.append))
        message = str(caught.exception)
        self.assertIn("certificate", message.lower())
        self.assertIn("SSL_CERT_FILE", message)
        self.assertIn("Install Certificates.command", message)
        self.assertEqual(slept, [], "a certificate failure must not sleep and retry")
        self.assertIsInstance(caught.exception.__cause__, SSLCertVerificationError)

    def test_ssl_verification_is_never_disabled(self):
        """Guard against a future 'fix' that turns verification off."""
        source = Path(__file__).resolve().parents[2] / "src" / "data" / "remote_parquet.py"
        text = source.read_text()
        for forbidden in ("verify=False", "CERT_NONE", "_create_unverified_context",
                          "check_hostname = False", "HF_HUB_DISABLE_SSL"):
            self.assertNotIn(forbidden, text)


class PinnedRevisionTests(unittest.TestCase):

    def test_branch_names_are_rejected_before_any_network_access(self):
        from src.data.loader import remote_handle_factory
        with self.assertRaises(ValueError):
            remote_handle_factory(split="train", config="hi", revision="main")

    def test_pinned_revision_builds_the_expected_path(self):
        from src.data.loader import hub_parquet_path
        self.assertEqual(hub_parquet_path(split="train", config="hi", revision=REVISION),
                         f"datasets/ai4bharat/MSMARCO-XI@{REVISION}/train/hintrain.parquet")


if __name__ == "__main__":
    unittest.main()
