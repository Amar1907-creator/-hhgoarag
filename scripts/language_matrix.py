#!/usr/bin/env python3
"""The language matrix: what is actually built, measured and usable.

Every number is read from a manifest produced by a real run. A language with no
build shows "not built" rather than a blank that could be mistaken for a zero,
and a language whose evaluation coverage is too low has its metrics withheld
rather than printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.languages import LANGUAGES, SPEECH_NONE  # noqa: E402

PROCESSED = Path("data/processed")
MANIFESTS = Path("data/manifests")
MIN_COVERAGE = 95.0


def newest_prefix(code: str) -> str | None:
    candidates = []
    for path in PROCESSED.glob(f"{code}-train-*-corpus.jsonl"):
        prefix = path.name[: -len("-corpus.jsonl")]
        if (PROCESSED / f"{prefix}-index" / "index.faiss").exists():
            candidates.append((path.stat().st_size, prefix))
    return max(candidates)[1] if candidates else None


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def row_for(language) -> dict:
    row = {"code": language.code, "native": language.native, "english": language.english,
           "script": language.script,
           "voice": language.speech_locale or "none",
           "voice_status": language.speech_status,
           "corpus": None, "queries": None, "coverage": None,
           "recall_at_5": None, "recall_at_10": None, "mrr": None,
           "status": "not built", "prefix": ""}
    if not language.has_train:
        row["status"] = "no train data in this revision"
        return row
    prefix = newest_prefix(language.code)
    if prefix is None:
        return row
    row["prefix"] = prefix
    build = read(MANIFESTS / f"{prefix}-build.json")
    row["corpus"] = build.get("unique_passages")
    row["status"] = "built, not evaluated"

    pipeline = read(MANIFESTS / f"{prefix}-pipeline.json")
    coverage = pipeline.get("coverage", {})
    bench = read(MANIFESTS / f"{prefix}-benchmark.json")
    metrics = bench.get("metrics", {})
    if coverage:
        row["queries"] = coverage.get("evaluation_queries")
        row["coverage"] = coverage.get("query_coverage_pct")
    if metrics:
        if row["coverage"] is not None and row["coverage"] < MIN_COVERAGE:
            row["status"] = f"metrics withheld: {row['coverage']:.1f}% coverage"
            return row
        row.update(recall_at_5=metrics.get("recall_at_5"), recall_at_10=metrics.get("recall_at_10"),
                   mrr=metrics.get("mrr"))
        row["status"] = "ready"
    return row


def cell(value, spec="{:.4f}") -> str:
    if value is None:
        return "—"
    return spec.format(value) if isinstance(value, float) else f"{value:,}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/LANGUAGE_MATRIX.md"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = [row_for(language) for language in LANGUAGES]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    header = ("| Language | Script | Corpus | Queries | Coverage | Recall@5 | Recall@10 | MRR "
              "| Voice | Status |")
    divider = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [f"# HHGOARAG language matrix", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
             f"Every figure comes from a manifest written by a real run; nothing here is estimated.",
             "", header, divider]
    for row in rows:
        voice = row["voice"] if row["voice"] != "none" else "none"
        if row["voice_status"] == SPEECH_NONE:
            voice = "none known"
        elif row["voice"] != "none":
            voice = f"{row['voice']} ({row['voice_status']})"
        lines.append(
            f"| {row['native']} · {row['english']} | {row['script']} | {cell(row['corpus'])} "
            f"| {cell(row['queries'])} | {cell(row['coverage'], '{:.1f}%')} "
            f"| {cell(row['recall_at_5'])} | {cell(row['recall_at_10'])} | {cell(row['mrr'])} "
            f"| {voice} | {row['status']} |")

    ready = [r for r in rows if r["status"] == "ready"]
    built = [r for r in rows if r["prefix"]]
    lines += ["", f"**{len(ready)} of {len(rows)} languages are evaluated and ready**; "
                  f"{len(built)} have a corpus and index built.", "",
              "Build another language with:", "", "```bash",
              "python3 scripts/run_pipeline.py --language ta --limit 5000",
              "# or several at once",
              "python3 scripts/build_languages.py --languages ta,bn,mr",
              "```", "",
              "A dash means the run that would produce that number has not happened. Metrics are "
              "withheld entirely when evaluation coverage is below "
              f"{MIN_COVERAGE:.0f}%, because below that they measure corpus coverage rather than "
              "retrieval quality.", "",
              "Voice locales are what the browser is asked to use. `untested` means no human has "
              "confirmed dictation in that language yet; the interface disables the microphone and "
              "says so if the browser rejects the locale.", ""]

    text = "\n".join(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
