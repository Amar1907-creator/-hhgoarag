#!/usr/bin/env python3
"""P50 / P70 / P100 latency across the pipeline, measured over many queries.

The task specification asks for P50/P70/P100 across a reasonable number of test
queries, and a sub-200 ms target for the full path. Both modes are measured and
reported separately, because they answer different questions:

  retrieval mode    embedding + vector search + evidence selection + guardrails,
                    answering from the retrieved passage itself. This is the path
                    the 200 ms budget applies to.
  generated mode    the same, plus a local language model writing the answer.
                    A local 3B model cannot produce tokens in 200 ms, and this
                    report says so with numbers rather than omitting the mode.

    python3 scripts/latency_report.py --queries 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.service import Service, discover_prefix, resolve_device  # noqa: E402

STAGES = ("embed", "search", "lookup", "generate", "total")
# Wall-clock measured by the caller, including service overhead the pipeline's
# own "total" does not see. This is the number the 200 ms budget applies to.
END_TO_END = "end_to_end"
TARGET_MS = 200.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarise(values: list[float]) -> dict:
    return {"p50": percentile(values, .50), "p70": percentile(values, .70),
            "p100": max(values), "mean": statistics.fmean(values), "n": len(values)}


def run(service: Service, questions: list[str], warmup: int) -> dict:
    for question in questions[:warmup]:
        service.answer(question)
    collected: dict[str, list[float]] = {stage: [] for stage in STAGES}
    collected[END_TO_END] = []
    grounded = 0
    for question in questions:
        began = time.perf_counter()
        payload = service.answer(question)
        wall = (time.perf_counter() - began) * 1e3
        for stage in STAGES:
            if stage in payload["timings_ms"]:
                collected[stage].append(payload["timings_ms"][stage])
        collected[END_TO_END].append(wall)
        grounded += bool(payload["grounded"])
    return {"stages": {stage: summarise(values) for stage, values in collected.items() if values},
            "grounded": grounded, "queries": len(questions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docs/LATENCY.md"))
    args = parser.parse_args()

    prefix = discover_prefix(args.prefix or "")
    if prefix is None:
        raise SystemExit("no built corpus found; run scripts/run_pipeline.py first")
    evaluation = Path("data/processed") / f"{prefix.replace('-train-', '-validation-')}-evaluation.jsonl"
    questions: list[str] = []
    if evaluation.exists():
        with evaluation.open(encoding="utf-8") as handle:
            questions = [json.loads(line)["query"] for line in handle if line.strip()]
    while len(questions) < args.queries and questions:
        questions += questions                     # repeat rather than invent queries
    questions = questions[: args.queries]
    if not questions:
        raise SystemExit("no evaluation queries available to measure with")

    device = resolve_device(None)
    modes = {}
    for label, kind in (("retrieval", "extractive"), ("generated", "auto")):
        service = Service(prefix=prefix, generator_kind=kind, device=device)
        status = service.load()
        if not status.ready:
            raise SystemExit(status.error)
        if kind == "auto" and not status.generator_available:
            print("no local model installed; the generated mode is skipped rather than faked",
                  file=sys.stderr)
            service.close()
            continue
        print(f"measuring {label} mode over {len(questions)} queries…", flush=True)
        modes[label] = run(service, questions, args.warmup)
        modes[label]["generator"] = status.generator
        service.close()

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = ["# HHGOARAG latency", "",
             f"Measured {stamp} on device `{device}`, corpus `{prefix}`, "
             f"{len(questions)} queries per mode after {args.warmup} warm-up queries.", ""]
    for label, report in modes.items():
        lines += [f"## {label} mode ({report['generator']})", "",
                  "| Stage | P50 | P70 | P100 | mean | samples |", "|---|---|---|---|---|---|"]
        for stage, stats in report["stages"].items():
            lines.append(f"| {stage} | {stats['p50']:.2f} ms | {stats['p70']:.2f} ms "
                         f"| {stats['p100']:.2f} ms | {stats['mean']:.2f} ms | {stats['n']} |")
        total = report["stages"].get(END_TO_END, {})
        if total:
            verdict = ("within the 200 ms target at P50, P70 and P100"
                       if total["p100"] <= TARGET_MS else
                       "within the 200 ms target at P50 and P70, but not at P100"
                       if total["p70"] <= TARGET_MS else
                       "above the 200 ms target")
            lines += ["", f"End to end: **P50 {total['p50']:.1f} ms · P70 {total['p70']:.1f} ms · "
                          f"P100 {total['p100']:.1f} ms** — {verdict}.", ""]
    lines += ["## Speech-to-text", "",
              "Transcription latency is measured per call and returned with every transcript "
              "(`latency_ms` on `/api/transcribe`). It is a network round trip to the provider and "
              "is reported separately rather than folded into the retrieval numbers, because it "
              "depends on the provider and the connection rather than on this pipeline.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
