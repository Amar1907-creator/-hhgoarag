"""Preflight checks must be exercisable without touching the network."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

try:
    import pyarrow  # noqa: F401
    PYARROW = True
except ImportError:
    PYARROW = False

from src.data.loader import coerce_record
from src.data.remote_parquet import stream_records

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "preflight.py"


def load_preflight():
    spec = importlib.util.spec_from_file_location("preflight_module", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records_from_fixture(rows=5, **kwargs):
    from tests.data.test_remote_parquet import write_fixture
    directory = tempfile.TemporaryDirectory()
    path = write_fixture(Path(directory.name) / "f.parquet", rows=rows, **kwargs)
    records = list(stream_records(open_handle=lambda: open(path, "rb"),
                                  coerce=coerce_record, batch_size=rows))
    return records, directory


class ReportTests(unittest.TestCase):
    def test_status_accounting(self):
        preflight = load_preflight()
        report = preflight.Report()
        with contextlib.redirect_stdout(io.StringIO()):
            report.ok("a", "fine"); report.warn("b", "hmm"); report.fail("c", "broken")
        self.assertEqual((report.failed, report.warned, len(report.checks)), (1, 1, 3))
        self.assertEqual([c["status"] for c in report.checks], ["PASS", "WARN", "FAIL"])


class LocalCheckTests(unittest.TestCase):
    def test_environment_and_outputs(self):
        preflight = load_preflight()
        report = preflight.Report()
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                preflight.check_environment(report)
                preflight.check_outputs(report, Path(directory) / "processed", Path(directory) / "manifests")
                preflight.check_resources(report, Path(directory))
        statuses = {c["check"]: c["status"] for c in report.checks}
        self.assertEqual(statuses["python_version"], "PASS")
        self.assertEqual(statuses["writable:processed"], "PASS")
        self.assertEqual(statuses["writable:manifests"], "PASS")

    def test_unwritable_output_directory_fails(self):
        preflight = load_preflight()
        report = preflight.Report()
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "blocked"
            blocked.write_text("i am a file, not a directory")
            with contextlib.redirect_stdout(io.StringIO()):
                preflight.check_outputs(report, blocked / "processed", Path(directory) / "manifests")
        self.assertGreaterEqual(report.failed, 1)


@unittest.skipUnless(PYARROW, "pyarrow is required")
class RecordCheckTests(unittest.TestCase):
    """The --limit 5 diagnostic, run against real Parquet with the pinned schema."""

    def test_healthy_records_pass_every_stage(self):
        preflight = load_preflight()
        records, directory = records_from_fixture(rows=5)
        try:
            report = preflight.Report()
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rows = preflight.check_records(report, records, "xx", "train")
            statuses = {c["check"]: c["status"] for c in report.checks}
            self.assertEqual(statuses["nested_passages"], "PASS")
            self.assertEqual(statuses["schema_validation"], "PASS")
            self.assertEqual(statuses["normalization_and_ids"], "PASS")
            self.assertEqual(report.failed, 0)
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(row["first_passage_id"].startswith("p_") for row in rows))
            self.assertIn("query_id", out.getvalue())
        finally:
            directory.cleanup()

    def test_misaligned_passage_lists_fail_loudly(self):
        preflight = load_preflight()
        records, directory = records_from_fixture(rows=5, malformed_at=2)
        try:
            report = preflight.Report()
            with contextlib.redirect_stdout(io.StringIO()):
                preflight.check_records(report, records, "xx", "train")
            statuses = {c["check"]: c["status"] for c in report.checks}
            self.assertEqual(statuses["nested_passages"], "FAIL")
            self.assertGreaterEqual(report.failed, 1)
        finally:
            directory.cleanup()

    def test_duplicate_passages_are_counted_not_dropped_silently(self):
        preflight = load_preflight()
        records, directory = records_from_fixture(rows=3)
        try:
            duplicated = records + [dict(records[0])]
            report = preflight.Report()
            with contextlib.redirect_stdout(io.StringIO()):
                preflight.check_records(report, duplicated, "xx", "train")
            detail = next(c["detail"] for c in report.checks if c["check"] == "normalization_and_ids")
            import re
            unique, duplicates = (int(n) for n in re.search(r"(\d+) unique, (\d+) duplicate", detail).groups())
            self.assertEqual(unique, 30, detail)   # 3 records x 10 passages
            self.assertEqual(duplicates, 10, detail)  # the repeated record
        finally:
            directory.cleanup()

    def test_no_records_is_a_failure(self):
        preflight = load_preflight()
        report = preflight.Report()
        with contextlib.redirect_stdout(io.StringIO()):
            preflight.check_records(report, [], "hi", "train")
        self.assertEqual(report.failed, 1)


class KnownFactsTests(unittest.TestCase):
    def test_recorded_facts_match_the_pinned_inventory(self):
        preflight = load_preflight()
        known = preflight.KNOWN[("hi", "train")]
        self.assertEqual(known["rows"], 778_638)
        self.assertEqual(known["row_groups"], 1)
        self.assertEqual(known["bytes"], 3_719_813_179)
        self.assertEqual(known["first_query_id"], 1_185_869)
        self.assertEqual(known["relative_path"], "train/hintrain.parquet")

    def test_prefix_read_ceiling_is_enforced(self):
        preflight = load_preflight()
        self.assertLessEqual(preflight.MAX_PREFIX_FRACTION, 0.10)


if __name__ == "__main__":
    unittest.main()
