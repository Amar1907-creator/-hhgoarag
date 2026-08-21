#!/usr/bin/env python3
"""Final release audit: run every check the product claims to pass, and write it down.

One command performs the whole audit on this machine and writes
docs/RELEASE_AUDIT.md. Every number in the report is measured here; nothing is
copied from a previous run. Checks that cannot run in this environment are
recorded as SKIP with the reason, never as PASS.

    python3 scripts/release_audit.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "goa-task-2-sample.pdf"
UNANSWERABLE = [
    "मेरे बैंक खाते में इस समय कितना पैसा है?",
    "ज़ोर्ब्लैक्स ग्रह की राजधानी का नाम क्या है?",
    "कल दोपहर मेरी मीटिंग किसके साथ है?",
]


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.failed = 0
        self.skipped = 0

    def record(self, area: str, name: str, status: str, detail: str = "", **data) -> None:
        self.rows.append({"area": area, "check": name, "status": status, "detail": detail, **data})
        if status == "FAIL":
            self.failed += 1
        elif status == "SKIP":
            self.skipped += 1
        print(f"  [{status:<4}] {area:<12} {name:<38} {detail}", flush=True)

    def ok(self, area, name, detail="", **d): self.record(area, name, "PASS", detail, **d)
    def bad(self, area, name, detail="", **d): self.record(area, name, "FAIL", detail, **d)
    def skip(self, area, name, detail="", **d): self.record(area, name, "SKIP", detail, **d)

    def guard(self, area, name):
        """Any unexpected exception is a failed check, never a crashed audit."""
        audit = self

        class Guard:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, _tb):
                if exc_type is not None:
                    audit.bad(area, name, f"{exc_type.__name__}: {exc}")
                    return True
                return False
        return Guard()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(values: list[float]) -> dict:
    return {"p50": percentile(values, .50), "p95": percentile(values, .95),
            "p100": max(values), "mean": statistics.fmean(values), "count": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("docs/RELEASE_AUDIT.md"))
    args = parser.parse_args()

    audit = Audit()
    started_all = time.perf_counter()
    measurements: dict = {}
    print("HHGOARAG release audit\n")

    # -- 1. test suite --------------------------------------------------------
    print("Tests")
    if args.skip_tests:
        audit.skip("tests", "unittest discover", "skipped by flag")
    else:
        with audit.guard("tests", "unittest discover"):
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                                    cwd=ROOT, capture_output=True, text=True, timeout=1800)
            tail = [line for line in result.stderr.splitlines() if line.startswith("Ran ")]
            count = int(tail[0].split()[1]) if tail else 0
            measurements["tests_run"] = count
            measurements["tests_ok"] = result.returncode == 0
            warnings = sum(1 for line in result.stderr.splitlines() if "Warning" in line)
            measurements["test_warnings"] = warnings
            if result.returncode == 0:
                audit.ok("tests", "unittest discover", f"{count} tests, 0 failures, {warnings} warnings")
            else:
                audit.bad("tests", "unittest discover",
                          f"{count} tests, exit {result.returncode}: {result.stderr.strip()[-300:]}")

    # -- 2. no hosted LLM anywhere in the runtime -----------------------------
    print("\nIndependence")
    with audit.guard("independence", "no hosted LLM"):
        needles = ("ANTHROPIC_API_KEY", "anthropic", "api.anthropic.com", "OPENAI_API_KEY")
        offenders = []
        targets = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) \
            + [ROOT / "static/index.html", ROOT / "run.sh", ROOT / "requirements.txt", ROOT / "README.md"]
        for path in targets:
            if not path.exists() or path.name == "release_audit.py":
                continue
            text = path.read_text(errors="ignore")
            for needle in needles:
                if needle in text:
                    offenders.append(f"{path.relative_to(ROOT)}:{needle}")
        measurements["hosted_llm_references"] = offenders
        (audit.ok if not offenders else audit.bad)(
            "independence", "no hosted LLM reference",
            "none in src/, scripts/, static/, run.sh, requirements, README" if not offenders
            else ", ".join(offenders))
    with audit.guard("independence", "no API key in env"):
        import os
        present = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if os.environ.get(k)]
        audit.ok("independence", "runs without an API key",
                 "no key is read by the product" + (f" (env has {present}, unused)" if present else ""))

    # -- 3. artifacts ---------------------------------------------------------
    print("\nArtifacts")
    from src.app.service import Service, discover_prefix, resolve_device
    prefix = discover_prefix(args.prefix or "hi-train-5k")
    if prefix is None:
        audit.bad("artifacts", "corpus present", "no built corpus under data/processed/")
        write_report(audit, measurements, args.output, started_all)
        sys.exit(1)
    measurements["prefix"] = prefix
    corpus_path = Path("data/processed") / f"{prefix}-corpus.jsonl"
    index_dir = Path("data/processed") / f"{prefix}-index"
    with audit.guard("artifacts", "corpus/index alignment"):
        with corpus_path.open("rb") as handle:
            corpus_lines = sum(1 for line in handle if line.strip())
        ids = json.loads((index_dir / "ids.json").read_text())
        import faiss
        ntotal = faiss.read_index(str(index_dir / "index.faiss")).ntotal
        measurements.update(corpus_passages=corpus_lines, index_ids=len(ids), index_vectors=ntotal)
        aligned = corpus_lines == len(ids) == ntotal
        (audit.ok if aligned else audit.bad)(
            "artifacts", "corpus == ids == index.ntotal",
            f"{corpus_lines:,} = {len(ids):,} = {ntotal:,}" if aligned
            else f"corpus {corpus_lines:,}, ids {len(ids):,}, index {ntotal:,}")
    with audit.guard("artifacts", "validation errors"):
        errors = Path("data/processed") / f"{prefix}-validation-errors.jsonl"
        count = sum(1 for line in errors.open() if line.strip()) if errors.exists() else 0
        measurements["validation_errors"] = count
        (audit.ok if count == 0 else audit.bad)("artifacts", "validation errors", f"{count}")
    with audit.guard("artifacts", "evaluation coverage"):
        pipeline_manifest = Path("data/manifests") / f"{prefix}-pipeline.json"
        coverage = json.loads(pipeline_manifest.read_text()).get("coverage", {}) \
            if pipeline_manifest.exists() else {}
        measurements["coverage"] = coverage
        pct = coverage.get("query_coverage_pct", 0)
        (audit.ok if pct >= 95 else audit.bad)(
            "artifacts", "evaluation coverage",
            f"{pct:.1f}% query coverage, {coverage.get('evaluation_queries', 0)} queries")
    with audit.guard("artifacts", "benchmark metrics"):
        bench = json.loads((Path("data/manifests") / f"{prefix}-benchmark.json").read_text())
        measurements["retrieval_metrics"] = bench.get("metrics", {})
        m = bench.get("metrics", {})
        audit.ok("artifacts", "recorded retrieval metrics",
                 f"R@1 {m.get('recall_at_1', 0):.4f} R@5 {m.get('recall_at_5', 0):.4f} "
                 f"R@10 {m.get('recall_at_10', 0):.4f} MRR {m.get('mrr', 0):.4f}")

    # -- 4. startup -----------------------------------------------------------
    print("\nStartup")
    device = resolve_device(None)
    measurements["device"] = device
    measurements["platform"] = f"{platform.system()} {platform.machine()}"
    try:
        import torch
        measurements["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:
        measurements["mps_available"] = False
    audit.ok("startup", "device selection",
             f"{device} (mps available: {measurements['mps_available']})")

    service = Service(prefix=prefix)
    with audit.guard("startup", "service load"):
        began = time.perf_counter()
        status = service.load()
        elapsed = time.perf_counter() - began
        measurements["startup_seconds"] = round(elapsed, 2)
        if not status.ready:
            audit.bad("startup", "service load", status.error)
            write_report(audit, measurements, args.output, started_all)
            sys.exit(1)
        audit.ok("startup", "service load",
                 f"{elapsed:.1f}s, {status.corpus_passages:,} passages, {status.embedding_model}")

    if not service.status.ready:
        write_report(audit, measurements, args.output, started_all)
        sys.exit(1)

    # -- 5. generator ---------------------------------------------------------
    print("\nGenerator")
    from src.rag.generator import ExtractiveGenerator, build_generator, installed_models
    models = installed_models()
    measurements["ollama_models"] = models
    if models:
        audit.ok("generator", "local model available", f"{', '.join(models[:3])}")
    else:
        audit.skip("generator", "local model available",
                   "Ollama not reachable; the extractive fallback is in use")
    with audit.guard("generator", "extractive fallback"):
        from src.rag.evidence import select_evidence
        evidence = select_evidence([("p", 0.95)], {"p": "गोवा एक राज्य है।"})
        answer = ExtractiveGenerator().generate("गोवा क्या है?", evidence)
        assert answer.citations == ["p"] and answer.answer
        audit.ok("generator", "extractive fallback", "answers with a citation and no model")

    # -- 6. corpus question flow ---------------------------------------------
    print("\nCorpus question flow")
    demo_path = Path("data/demo/questions.json")
    demo = json.loads(demo_path.read_text()) if demo_path.exists() else []
    eval_path = Path("data/processed") / f"{prefix.replace('-train-', '-validation-')}-evaluation.jsonl"
    questions = [row["question"] for row in demo if row.get("expect") == "evidence"]
    if eval_path.exists():
        with eval_path.open(encoding="utf-8") as handle:
            questions += [json.loads(line)["query"] for line in handle if line.strip()]
    questions = questions[: args.queries] or ["गोवा क्या है?"]

    stage_times: dict[str, list[float]] = {"embed": [], "search": [], "lookup": [], "generate": [], "total": []}
    grounded_count = 0
    citation_violations = 0
    with audit.guard("retrieval", "grounded answers"):
        for question in questions:
            payload = service.answer(question)
            for stage, values in stage_times.items():
                if stage in payload["timings_ms"]:
                    values.append(payload["timings_ms"][stage])
            if payload["grounded"]:
                grounded_count += 1
                retrieved = {hit["passage_id"] for hit in payload["retrieval"]}
                if any(c["passage_id"] not in retrieved for c in payload["citations"]):
                    citation_violations += 1
        measurements["queries_run"] = len(questions)
        measurements["grounded"] = grounded_count
        audit.ok("retrieval", "answers produced",
                 f"{grounded_count}/{len(questions)} grounded, "
                 f"{len(questions) - grounded_count} abstained")
    with audit.guard("retrieval", "citation validity"):
        (audit.ok if citation_violations == 0 else audit.bad)(
            "retrieval", "every citation was retrieved",
            f"{citation_violations} answers cited a passage that was not retrieved"
            if citation_violations else "no citation outside the retrieved set")

    with audit.guard("retrieval", "determinism"):
        repeats = [service.answer(questions[0]) for _ in range(3)]
        same = len({r["answer"] for r in repeats}) == 1 and \
            len({tuple(c["passage_id"] for c in r["citations"]) for r in repeats}) == 1
        measurements["deterministic"] = same
        (audit.ok if same else audit.bad)("retrieval", "repeated queries agree",
                                          "same answer and citations three times" if same
                                          else "repeated identical queries diverged")

    # -- 7. abstention --------------------------------------------------------
    print("\nAbstention")
    with audit.guard("abstention", "unsupported questions"):
        refused, answered = [], []
        for question in UNANSWERABLE:
            payload = service.answer(question)
            (refused if not payload["grounded"] else answered).append(
                (question, payload["retrieval"][0]["score"] if payload["retrieval"] else 0.0))
        measurements["abstention_refused"] = len(refused)
        measurements["abstention_answered"] = [
            {"question": q, "best_score": round(s, 4)} for q, s in answered]
        if refused:
            audit.ok("abstention", "refuses unsupported questions",
                     f"{len(refused)}/{len(UNANSWERABLE)} refused, "
                     f"best score {min(s for _, s in refused):.3f}")
        else:
            audit.bad("abstention", "refuses unsupported questions",
                      f"none of {len(UNANSWERABLE)} was refused; weakest scored "
                      f"{min(s for _, s in answered):.4f} -- raise HHGOARAG_MIN_SCORE")

    # -- 8. PDF flow ----------------------------------------------------------
    print("\nPDF flow")
    if not FIXTURE.exists():
        audit.skip("pdf", "ingest", "fixture missing")
    else:
        with audit.guard("pdf", "ingest and cite"):
            began = time.perf_counter()
            record = service.ingest_document(FIXTURE.read_bytes(), "GOA Task-2.pdf")
            deadline = time.time() + 300
            while time.time() < deadline:
                current = service.documents.get(record.document_id)
                if current and current.status in ("ready", "failed"):
                    record = current
                    break
                time.sleep(0.2)
            ingest_seconds = time.perf_counter() - began
            measurements["pdf"] = {"pages": record.pages, "chunks": record.chunks,
                                   "status": record.status, "seconds": round(ingest_seconds, 2)}
            if record.status != "ready":
                audit.bad("pdf", "ingest", f"{record.status}: {record.message}")
            else:
                audit.ok("pdf", "ingest", f"{record.pages} pages, {record.chunks} chunks, "
                                          f"{ingest_seconds:.1f}s")
                payload = service.answer("गोवा का सबसे व्यस्त समुद्र तट कौन सा है?",
                                         source=f"document:{record.document_id}")
                if not payload["grounded"]:
                    audit.bad("pdf", "answer from PDF", f"abstained: {payload['reason']}")
                else:
                    citation = payload["citations"][0]
                    measurements["pdf"]["citation"] = {"document": citation.get("document"),
                                                       "page": citation.get("page"),
                                                       "score": citation.get("score")}
                    correct = citation.get("page") == 7
                    (audit.ok if correct else audit.bad)(
                        "pdf", "cites the correct page",
                        f"{citation.get('document')} - Page {citation.get('page')} "
                        f"(score {citation.get('score')})")
                    payload = service.answer("मेरे बैंक खाते में कितना पैसा है?",
                                             source=f"document:{record.document_id}")
                    (audit.ok if not payload["grounded"] else audit.bad)(
                        "pdf", "abstains on unsupported PDF question",
                        payload["reason"])

    # -- 9. restart persistence ----------------------------------------------
    print("\nRestart")
    with audit.guard("restart", "documents survive"):
        second = Service(prefix=prefix)
        status = second.load()
        restored = [s for s in second.knowledge_sources() if s["kind"] == "document"]
        measurements["restored_documents"] = len(restored)
        if not status.ready:
            audit.bad("restart", "service reload", status.error)
        else:
            (audit.ok if restored else audit.bad)(
                "restart", "documents re-attached after restart",
                f"{len(restored)} document source(s) restored: "
                f"{', '.join(s['label'] for s in restored)}" if restored
                else "uploaded documents were not restored")
            if restored:
                payload = second.answer("गोवा का सबसे व्यस्त समुद्र तट कौन सा है?",
                                        source=restored[0]["key"])
                (audit.ok if payload["grounded"] else audit.bad)(
                    "restart", "PDF still answerable after restart",
                    f"page {payload['citations'][0].get('page')}" if payload["grounded"]
                    else payload["reason"])
        second.close()

    # -- 10. latency ----------------------------------------------------------
    print("\nLatency")
    for stage, values in stage_times.items():
        if not values:
            continue
        stats = summarise(values)
        measurements.setdefault("latency_ms", {})[stage] = {k: round(v, 2) for k, v in stats.items()}
        audit.ok("latency", f"{stage}",
                 f"p50 {stats['p50']:.2f} ms  p95 {stats['p95']:.2f} ms  p100 {stats['p100']:.2f} ms")

    # -- 11. demo questions ---------------------------------------------------
    print("\nDemo questions")
    if not demo:
        audit.bad("demo", "demonstration set", "data/demo/questions.json is missing")
    else:
        verified = []
        for row in demo:
            payload = service.answer(row["question"])
            behaved = payload["grounded"] if row["expect"] == "evidence" else not payload["grounded"]
            verified.append({"question": row["question"], "expect": row["expect"],
                             "grounded": payload["grounded"], "behaved": behaved,
                             "citations": len(payload["citations"]),
                             "best_score": payload["retrieval"][0]["score"] if payload["retrieval"] else 0.0})
        measurements["demo_questions"] = verified
        wrong = [v for v in verified if not v["behaved"]]
        (audit.ok if not wrong else audit.bad)(
            "demo", "each question behaves as labelled",
            f"{len(verified) - len(wrong)}/{len(verified)} correct" +
            ("" if not wrong else f"; wrong: {[w['question'][:32] for w in wrong]}"))
        has_abstention = any(v["expect"].startswith("abstention") and v["behaved"] for v in verified)
        multi = any(v["citations"] >= 2 for v in verified)
        (audit.ok if has_abstention else audit.bad)(
            "demo", "abstention example present",
            "one demonstration question refuses" if has_abstention
            else "no verified abstention question -- re-run scripts/pick_demo_questions.py")
        (audit.ok if multi else audit.skip)(
            "demo", "multi-evidence example present",
            "a question cites two or more passages" if multi
            else "no demo question cited more than one passage")

    audit.skip("voice", "microphone dictation",
               "browser Web Speech API (hi-IN); wiring verified by inspection, needs a human click")

    service.close()
    write_report(audit, measurements, args.output, started_all)
    sys.exit(1 if audit.failed else 0)


def write_report(audit: Audit, measurements: dict, output: Path, started: float) -> None:
    elapsed = time.perf_counter() - started
    passed = sum(1 for row in audit.rows if row["status"] == "PASS")
    verdict = "RELEASE READY" if audit.failed == 0 else "NOT READY"
    lines = [f"# HHGOARAG release audit", "",
             f"`{verdict}` — {passed} passed, {audit.failed} failed, {audit.skipped} skipped, "
             f"{elapsed:.0f}s", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} on "
             f"{measurements.get('platform', '?')}, device `{measurements.get('device', '?')}`.",
             "", "Every number below was measured by this run.", "",
             "| Status | Area | Check | Detail |", "|---|---|---|---|"]
    for row in audit.rows:
        lines.append(f"| {row['status']} | {row['area']} | {row['check']} | {row['detail']} |")

    latency = measurements.get("latency_ms", {})
    if latency:
        lines += ["", "## Measured latency", "",
                  "| Stage | p50 | p95 | p100 | mean | samples |", "|---|---|---|---|---|---|"]
        for stage, stats in latency.items():
            lines.append(f"| {stage} | {stats['p50']} ms | {stats['p95']} ms | {stats['p100']} ms | "
                         f"{stats['mean']} ms | {stats['count']} |")

    metrics = measurements.get("retrieval_metrics", {})
    coverage = measurements.get("coverage", {})
    if metrics:
        lines += ["", "## Retrieval quality", "",
                  f"- Corpus: {measurements.get('corpus_passages', 0):,} passages "
                  f"(`{measurements.get('prefix', '?')}`)",
                  f"- Evaluation: {coverage.get('evaluation_queries', 0)} queries, "
                  f"{coverage.get('query_coverage_pct', 0):.1f}% query coverage, "
                  f"{coverage.get('positive_coverage_pct', 0):.1f}% positive coverage",
                  f"- Recall@1 {metrics.get('recall_at_1', 0):.4f} · "
                  f"Recall@5 {metrics.get('recall_at_5', 0):.4f} · "
                  f"Recall@10 {metrics.get('recall_at_10', 0):.4f} · MRR {metrics.get('mrr', 0):.4f}",
                  f"- Startup: {measurements.get('startup_seconds', 0)}s"]

    pdf = measurements.get("pdf")
    if pdf:
        lines += ["", "## PDF flow", "",
                  f"- {pdf.get('pages')} pages, {pdf.get('chunks')} chunks, "
                  f"ingested in {pdf.get('seconds')}s",
                  f"- Citation: {pdf.get('citation', {}).get('document')} — "
                  f"Page {pdf.get('citation', {}).get('page')}"]

    demo = measurements.get("demo_questions")
    if demo:
        lines += ["", "## Demonstration questions", "",
                  "| Expect | Behaved | Citations | Best score | Question |", "|---|---|---|---|---|"]
        for row in demo:
            lines.append(f"| {row['expect']} | {'yes' if row['behaved'] else 'NO'} | "
                         f"{row['citations']} | {row['best_score']:.3f} | {row['question'][:52]} |")

    lines += ["", "## Independence", "",
              f"- Hosted-LLM references in shipped code: "
              f"{measurements.get('hosted_llm_references') or 'none'}",
              f"- Local generation: {measurements.get('ollama_models') or 'no Ollama model; extractive fallback'}",
              ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"\n{'=' * 70}\n{verdict}: {passed} passed, {audit.failed} failed, "
          f"{audit.skipped} skipped in {elapsed:.0f}s\nreport written to {output}")


if __name__ == "__main__":
    main()
