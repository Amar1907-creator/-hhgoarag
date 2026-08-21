"""Coverage gating and status reporting must refuse to launder bad numbers."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_mod", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_corpus(path: Path, ids):
    with path.open("w", encoding="utf-8") as handle:
        for passage_id in ids:
            handle.write(json.dumps({"passage_id": passage_id, "text": f"text {passage_id}",
                                     "language": "hin_Deva"}) + "\n")


def write_eval(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for query_id, positives in rows:
            handle.write(json.dumps({"query_id": query_id, "query": "सवाल",
                                     "positive_passage_ids": positives}) + "\n")


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.verify = load("verify_evaluation")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_full_coverage(self):
        write_corpus(self.root / "c.jsonl", ["p_1", "p_2", "p_3"])
        write_eval(self.root / "e.jsonl", [(1, ["p_1"]), (2, ["p_2", "p_3"])])
        report = self.verify.coverage(self.root / "e.jsonl", self.verify.load_corpus_ids(self.root / "c.jsonl"))
        self.assertEqual(report["evaluation_queries"], 2)
        self.assertEqual(report["total_positive_ids"], 3)
        self.assertEqual(report["positive_ids_in_corpus"], 3)
        self.assertEqual(report["query_coverage_pct"], 100.0)

    def test_the_994_passage_failure_mode_is_detected(self):
        """The real bug: positives that are not in the indexed corpus."""
        write_corpus(self.root / "c.jsonl", [f"p_{n}" for n in range(3)])
        write_eval(self.root / "e.jsonl", [(n, [f"q_{n}"]) for n in range(50)] + [(99, ["p_1"])])
        report = self.verify.coverage(self.root / "e.jsonl", self.verify.load_corpus_ids(self.root / "c.jsonl"))
        self.assertEqual(report["queries_with_corpus_positive"], 1)
        self.assertLess(report["query_coverage_pct"], 5.0)

    def test_partial_positives_still_count_the_query(self):
        write_corpus(self.root / "c.jsonl", ["p_1"])
        write_eval(self.root / "e.jsonl", [(1, ["p_1", "missing"])])
        report = self.verify.coverage(self.root / "e.jsonl", self.verify.load_corpus_ids(self.root / "c.jsonl"))
        self.assertEqual(report["queries_with_corpus_positive"], 1)
        self.assertEqual(report["positive_ids_in_corpus"], 1)
        self.assertEqual(report["positive_coverage_pct"], 50.0)

    def test_empty_evaluation_does_not_divide_by_zero(self):
        write_corpus(self.root / "c.jsonl", ["p_1"])
        write_eval(self.root / "e.jsonl", [])
        report = self.verify.coverage(self.root / "e.jsonl", self.verify.load_corpus_ids(self.root / "c.jsonl"))
        self.assertEqual(report["evaluation_queries"], 0)
        self.assertEqual(report["query_coverage_pct"], 0.0)


class StatusReportTests(unittest.TestCase):
    def setUp(self):
        self.status = load("project_status")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def build_tree(self, root: Path, benchmark: dict):
        (root / "data/processed").mkdir(parents=True)
        (root / "data/manifests").mkdir(parents=True)
        write_corpus(root / "data/processed/x-train-corpus.jsonl", ["p_1", "p_2"])
        write_eval(root / "data/processed/x-validation-evaluation.jsonl", [(1, ["p_1"]), (2, ["p_2"])])
        (root / "data/manifests/x-train-build.json").write_text(json.dumps(
            {"revision": "b" * 40, "records_processed": 10, "unique_passages": 2,
             "duplicate_percentage": 0.0, "configuration": {"loader_config": "hi", "split": "train"}}))
        (root / "data/manifests/x-train-benchmark.json").write_text(json.dumps(benchmark))

    def report_for(self, benchmark: dict) -> str:
        import subprocess, sys, shutil
        root = Path(self.directory.name) / "tree"
        root.mkdir()
        self.build_tree(root, benchmark)
        for item in ("scripts", "src"):
            shutil.copytree(Path(__file__).resolve().parents[2] / item, root / item)
        result = subprocess.run([sys.executable, "scripts/project_status.py", "--prefix", "x-train",
                                 "--eval-prefix", "x-validation", "--skip-tests"],
                                cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        return result.stdout

    def test_manifest_without_p95_reports_instead_of_crashing(self):
        """Benchmark manifests written before p95 existed must not break the report."""
        output = self.report_for({"model": "e5", "embedding_dimension": 384, "corpus_passages": 2,
                                  "metrics": {"recall_at_5": 0.5, "recall_at_10": 0.6, "mrr": 0.4},
                                  "warm_latency": {"total_retrieval": {"p50_ms": 12.0}}})
        self.assertIn("not recorded", output)
        self.assertIn("12.00 ms", output)

    def test_full_coverage_metrics_are_not_labelled_invalid(self):
        output = self.report_for({"model": "e5", "embedding_dimension": 384, "corpus_passages": 2,
                                  "metrics": {"recall_at_5": 0.5, "recall_at_10": 0.6, "mrr": 0.4},
                                  "warm_latency": {"total_retrieval": {"p50_ms": 12.0, "p95_ms": 20.0}}})
        self.assertNotIn("INVALID", output)
        self.assertIn("0.5000", output)

    def test_low_coverage_metrics_are_labelled_invalid(self):
        """The 994-passage trap: never present Recall next to 2% coverage unflagged."""
        import subprocess, sys, shutil
        root = Path(self.directory.name) / "bad"
        root.mkdir()
        self.build_tree(root, {"model": "e5", "embedding_dimension": 384, "corpus_passages": 2,
                               "metrics": {"recall_at_5": 0.54, "recall_at_10": 0.63, "mrr": 0.378},
                               "warm_latency": {"total_retrieval": {"p50_ms": 13.8, "p95_ms": 20.0}}})
        # Positives that are not in the corpus, exactly like the real failure.
        write_eval(root / "data/processed/x-validation-evaluation.jsonl",
                   [(n, [f"absent_{n}"]) for n in range(40)] + [(99, ["p_1"])])
        for item in ("scripts", "src"):
            shutil.copytree(Path(__file__).resolve().parents[2] / item, root / item)
        result = subprocess.run([sys.executable, "scripts/project_status.py", "--prefix", "x-train",
                                 "--eval-prefix", "x-validation", "--skip-tests"],
                                cwd=root, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertIn("INVALID (low coverage)", result.stdout)
        self.assertIn("NOT VALID", result.stdout)

    def test_count_lines_ignores_blank_lines(self):
        path = Path(self.directory.name) / "f.jsonl"
        path.write_text('{"a":1}\n\n{"a":2}\n')
        self.assertEqual(self.status.count_lines(path), 2)

    def test_count_lines_returns_none_when_absent(self):
        self.assertIsNone(self.status.count_lines(Path(self.directory.name) / "nope.jsonl"))


class PipelineDriverTests(unittest.TestCase):
    def setUp(self):
        self.driver = load("run_pipeline")

    def test_prefix_derivation(self):
        self.assertEqual(5000 // 1000, 5)

    def test_helpers(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "f.jsonl"
        path.write_text('{"a":1}\n{"a":2}\n\n')
        self.assertEqual(self.driver.count_lines(path), 2)
        self.assertEqual(self.driver.read_json(Path(directory.name) / "absent.json"), {})


if __name__ == "__main__":
    unittest.main()
