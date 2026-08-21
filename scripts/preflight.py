#!/usr/bin/env python3
"""Preflight for MSMARCO-XI ingestion: prove every precondition before a build.

Runs cheap local checks first and only then touches the network, so a broken
environment fails in milliseconds instead of after a multi-GB download. Every
check prints PASS, WARN or FAIL with the measured value behind it; any FAIL
exits non-zero and no corpus build should be started.

With --limit N (default 5) it also runs the tiny diagnostic: N real records
pulled through the exact production path -- nested passage decoding, schema
validation, normalization, stable passage IDs and deduplication -- printed
record by record. Nothing is written to the corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.deduplicate import ExactDeduplicator, passage_id
from src.data.loader import (
    DATASET_ID,
    PINNED_REVISION,
    coerce_record,
    hub_parquet_path,
    parquet_relative_path,
    remote_handle_factory,
    resolve_language_code,
)
from src.data.normalize import normalize_text
from src.data.remote_parquet import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BUFFER_SIZE,
    iter_batches,
    open_parquet,
    peak_rss_mb,
)
from src.data.schema import REQUIRED_FIELDS, validate_record

# Facts recorded from the pinned revision for hi/train. Used as assertions, not guesses.
KNOWN = {
    ("hi", "train"): {
        "relative_path": "train/hintrain.parquet",
        "bytes": 3_719_813_179,
        "rows": 778_638,
        "row_groups": 1,
        "first_query_id": 1_185_869,
        "target_lang": "hin_Deva",
    }
}
PASSAGE_COLUMNS = ("English_passages", "Translated_passages", "is_selected")
# Reading a prefix must not drag down the object. Enforces the "do not download
# 3.7 GB to process 5 records" rule.
MAX_PREFIX_FRACTION = 0.10
# Rough planning figures for the corpus that will follow.
BYTES_PER_100K_BUILD = 4.0e9


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.failed = 0
        self.warned = 0

    def add(self, name: str, status: str, detail: str, **data) -> None:
        self.checks.append({"check": name, "status": status, "detail": detail, **data})
        if status == "FAIL":
            self.failed += 1
        elif status == "WARN":
            self.warned += 1
        marker = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[status]
        print(f"  [{marker}] {name:<28} {detail}", flush=True)

    def ok(self, name, detail, **data): self.add(name, "PASS", detail, **data)
    def warn(self, name, detail, **data): self.add(name, "WARN", detail, **data)
    def fail(self, name, detail, **data): self.add(name, "FAIL", detail, **data)


def total_memory_gb() -> float | None:
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return None


def check_environment(report: Report) -> None:
    print("\nLocal environment")
    version = sys.version_info
    detail = f"{platform.python_version()} ({platform.system()} {platform.machine()})"
    if version >= (3, 10):
        report.ok("python_version", detail, version=platform.python_version())
    else:
        report.fail("python_version", f"{detail} -- 3.10+ required")

    pins: dict[str, str] = {}
    requirements = Path("requirements.txt")
    if requirements.exists():
        for line in requirements.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                name, _, pinned = line.partition("==")
                pins[name.strip()] = pinned.strip()

    from importlib import metadata
    for package in ("huggingface_hub", "pyarrow", "fsspec", "faiss-cpu", "sentence-transformers"):
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            required_now = package in ("huggingface_hub", "pyarrow", "fsspec")
            (report.fail if required_now else report.warn)(
                f"dependency:{package}", "not installed" + ("" if required_now else " (not needed for ingestion)"))
            continue
        pinned = pins.get(package)
        if pinned and pinned != installed:
            report.warn(f"dependency:{package}", f"{installed} installed, requirements.txt pins {pinned}",
                        installed=installed, pinned=pinned)
        else:
            report.ok(f"dependency:{package}", installed, installed=installed)

    if "datasets" in {m.metadata["Name"].lower() for m in metadata.distributions() if m.metadata["Name"]}:
        report.warn("datasets_not_used", "installed but deliberately unused: its streaming "
                    "conversion fails on this nested passages struct")


def check_outputs(report: Report, output_dir: Path, manifest_dir: Path) -> None:
    print("\nOutput paths")
    for label, directory in (("processed", output_dir), ("manifests", manifest_dir)):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".preflight-{os.getpid()}"
            probe.write_text("ok")
        except Exception as exc:
            report.fail(f"writable:{label}", f"{directory}: {exc}")
            continue
        try:
            probe.unlink()
        except OSError as exc:
            report.warn(f"writable:{label}",
                        f"{directory} is writable but the probe could not be removed ({exc}); "
                        f"delete {probe} by hand")
            continue
        report.ok(f"writable:{label}", str(directory))


def check_resources(report: Report, output_dir: Path) -> None:
    print("\nResources")
    memory = total_memory_gb()
    if memory is None:
        report.warn("memory_total", "could not determine physical memory")
    elif memory < 8:
        report.warn("memory_total", f"{memory:.1f} GB -- tight for a large index build", gb=memory)
    else:
        report.ok("memory_total", f"{memory:.1f} GB", gb=memory)
    report.ok("memory_this_process", f"{peak_rss_mb():.0f} MB peak RSS so far", mb=peak_rss_mb())

    usage = shutil.disk_usage(output_dir if output_dir.exists() else Path("."))
    free = usage.free / 1e9
    detail = f"{free:.1f} GB free"
    if free < 2:
        report.fail("disk_free", f"{detail} -- not enough for any build")
    elif free * 1e9 < BYTES_PER_100K_BUILD:
        report.warn("disk_free", f"{detail} -- a later 100k build needs about "
                                 f"{BYTES_PER_100K_BUILD/1e9:.0f} GB", gb=free)
    else:
        report.ok("disk_free", detail, gb=free)


def check_hub(report: Report, config: str, split: str, revision: str) -> dict:
    print("\nHugging Face access")
    facts: dict = {}
    if len(revision) != 40:
        report.fail("pinned_revision", f"{revision!r} is not a 40-character commit SHA")
        return facts
    report.ok("pinned_revision", revision)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        report.fail("hub_import", str(exc))
        return facts

    started = time.perf_counter()
    try:
        info = HfApi().dataset_info(DATASET_ID, revision=revision, files_metadata=False)
    except Exception as exc:
        report.fail("revision_accessible", f"{type(exc).__name__}: {exc}")
        return facts
    api_ms = (time.perf_counter() - started) * 1e3
    facts["dataset_info_ms"] = api_ms
    report.ok("revision_accessible", f"{DATASET_ID}@{info.sha[:12]} in {api_ms:.0f} ms", sha=info.sha)
    if info.sha != revision:
        report.fail("revision_pinned", f"resolved to {info.sha}, expected {revision}")

    relative = parquet_relative_path(split=split, config=config)
    siblings = {sibling.rfilename for sibling in (info.siblings or [])}
    if siblings and relative not in siblings:
        report.fail("parquet_exists", f"{relative} not present in the revision")
    else:
        report.ok("parquet_exists", relative, path=relative)

    if api_ms > 5000:
        report.warn("network_latency", f"metadata call took {api_ms:.0f} ms -- slow link")
    else:
        report.ok("network_latency", f"metadata round trip {api_ms:.0f} ms")

    timeout = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT")
    if timeout is None:
        report.warn("download_timeout", "HF_HUB_DOWNLOAD_TIMEOUT unset (default 10s); "
                                        "export HF_HUB_DOWNLOAD_TIMEOUT=60 for a flaky link")
    else:
        report.ok("download_timeout", f"HF_HUB_DOWNLOAD_TIMEOUT={timeout}s")
    return facts


def check_parquet(report: Report, config: str, split: str, revision: str, limit: int) -> tuple[dict, list[dict]]:
    print("\nRemote Parquet (bounded reads only)")
    facts: dict = {}
    records: list[dict] = []
    known = KNOWN.get((config, split))

    opener = remote_handle_factory(split=split, config=config, revision=revision,
                                   block_size=DEFAULT_BLOCK_SIZE, counting=True)
    started = time.perf_counter()
    try:
        reader = opener()
    except Exception as exc:
        report.fail("remote_open", f"{type(exc).__name__}: {exc}")
        return facts, records
    open_ms = (time.perf_counter() - started) * 1e3
    report.ok("remote_open", f"opened in {open_ms:.0f} ms via {hub_parquet_path(split=split, config=config, revision=revision)}")

    try:
        started = time.perf_counter()
        parquet_file = open_parquet(reader, buffer_size=DEFAULT_BUFFER_SIZE)
        metadata = parquet_file.metadata
        footer_ms = (time.perf_counter() - started) * 1e3
        footer_bytes = reader.bytes_read
        facts.update(rows=metadata.num_rows, row_groups=metadata.num_row_groups,
                     footer_bytes=footer_bytes, footer_ms=footer_ms)
        report.ok("parquet_metadata",
                  f"{metadata.num_rows:,} rows, {metadata.num_row_groups} row group(s), "
                  f"footer read {footer_bytes/1e6:.2f} MB in {footer_ms:.0f} ms")
        if known:
            if metadata.num_rows != known["rows"]:
                report.fail("row_count", f"{metadata.num_rows:,} != expected {known['rows']:,}")
            else:
                report.ok("row_count", f"{metadata.num_rows:,} matches the pinned inventory")
            if metadata.num_row_groups != known["row_groups"]:
                report.warn("row_groups", f"{metadata.num_row_groups} row groups, inventory recorded "
                                          f"{known['row_groups']}")

        names = set(parquet_file.schema_arrow.names)
        missing = [field for field in REQUIRED_FIELDS if field not in names]
        if missing:
            report.fail("schema_fields", f"missing columns: {missing}")
        else:
            report.ok("schema_fields", f"all {len(REQUIRED_FIELDS)} documented columns present")
        passages_type = parquet_file.schema_arrow.field("passages").type
        nested = [passages_type.field(i).name for i in range(passages_type.num_fields)]
        if set(nested) != set(PASSAGE_COLUMNS):
            report.fail("passages_struct", f"unexpected nested fields: {nested}")
        else:
            report.ok("passages_struct", "struct<English_passages, Translated_passages, is_selected>")

        started = time.perf_counter()
        for _, batch in iter_batches(parquet_file, batch_size=max(limit, 1)):
            columns = batch.to_pydict()
            rows = batch.num_rows
            del batch
            for index in range(min(rows, limit)):
                records.append(coerce_record({name: values[index] for name, values in columns.items()}))
            break
        first_batch_ms = (time.perf_counter() - started) * 1e3
        prefix_bytes = reader.bytes_read
        facts.update(prefix_bytes=prefix_bytes, first_batch_ms=first_batch_ms,
                     records_read=len(records))

        file_bytes = known["bytes"] if known else None
        if file_bytes:
            fraction = prefix_bytes / file_bytes
            detail = (f"{len(records)} records cost {prefix_bytes/1e6:.1f} MB "
                      f"= {100*fraction:.2f}% of the {file_bytes/1e9:.2f} GB file, in {first_batch_ms:.0f} ms")
            if fraction > MAX_PREFIX_FRACTION:
                report.fail("range_read_bounded", detail + " -- the reader is not streaming")
            else:
                report.ok("range_read_bounded", detail, prefix_bytes=prefix_bytes, fraction=fraction)
        else:
            report.ok("range_read_bounded", f"{len(records)} records cost {prefix_bytes/1e6:.1f} MB")

        report.ok("memory_after_read", f"{peak_rss_mb():.0f} MB peak RSS", mb=peak_rss_mb())
    except Exception as exc:
        report.fail("parquet_read", f"{type(exc).__name__}: {exc}")
    finally:
        try:
            reader.close()
        except Exception:
            pass
    return facts, records


def check_records(report: Report, records: list[dict], config: str, split: str) -> list[dict]:
    print("\nRecord decoding and validation")
    known = KNOWN.get((config, split))
    rows: list[dict] = []
    if not records:
        report.fail("first_record", "no records were read")
        return rows

    first = records[0]
    if known:
        if first.get("query_id") == known["first_query_id"]:
            report.ok("first_record", f"query_id={first['query_id']} matches the recorded first row")
        else:
            report.fail("first_record", f"query_id={first.get('query_id')} != expected {known['first_query_id']}")
        if first.get("target_lang") == known["target_lang"]:
            report.ok("target_lang", first["target_lang"])
        else:
            report.fail("target_lang", f"{first.get('target_lang')} != {known['target_lang']}")
    print(f"         query: {first.get('query', '')[:80]}")

    aligned = True
    for record in records:
        lengths = {name: len(record["passages"][name]) for name in PASSAGE_COLUMNS}
        if len(set(lengths.values())) != 1:
            aligned = False
            report.fail("nested_passages", f"query_id={record['query_id']} list lengths differ: {lengths}")
            break
    if aligned:
        counts = [len(record["passages"]["Translated_passages"]) for record in records]
        report.ok("nested_passages", f"{len(records)} records decoded, "
                                     f"{min(counts)}-{max(counts)} passages each, all three lists aligned")
        flags = sorted({int(flag) for record in records for flag in record["passages"]["is_selected"]})
        report.ok("is_selected_flags", f"values {flags}")

    invalid = [(record["query_id"], validate_record(record)) for record in records]
    bad = [(query_id, result) for query_id, result in invalid if not result.valid]
    if bad:
        report.fail("schema_validation", f"{len(bad)} of {len(records)} records invalid: {bad[0][1].errors[:2]}")
    else:
        warnings = sum(len(result.warnings) for _, result in invalid)
        report.ok("schema_validation", f"{len(records)} records valid ({warnings} field warnings)")

    with tempfile.TemporaryDirectory() as directory:
        deduper = ExactDeduplicator(Path(directory) / "seen.sqlite3", reset=True)
        unique = duplicates = empty = 0
        for record in records:
            language = record["target_lang"]
            row = {"query_id": record["query_id"], "passages": 0, "unique": 0, "duplicate": 0,
                   "first_passage_id": None, "first_chars": 0}
            for text in record["passages"]["Translated_passages"]:
                normalized = normalize_text(text)
                if not normalized:
                    empty += 1
                    continue
                row["passages"] += 1
                stable = passage_id(language, normalized)
                if row["first_passage_id"] is None:
                    row["first_passage_id"] = stable
                    row["first_chars"] = len(normalized)
                if deduper.add(stable):
                    unique += 1
                    row["unique"] += 1
                else:
                    duplicates += 1
                    row["duplicate"] += 1
            rows.append(row)
        total = unique + duplicates
        report.ok("normalization_and_ids",
                  f"{total} passages -> {unique} unique, {duplicates} duplicate, {empty} empty")
        deduper.close()

    print("\n  Per-record diagnostic")
    print(f"  {'query_id':>10} {'passages':>9} {'unique':>7} {'dup':>4} {'chars':>7}  passage_id")
    for row in rows:
        print(f"  {row['query_id']:>10} {row['passages']:>9} {row['unique']:>7} {row['duplicate']:>4} "
              f"{row['first_chars']:>7}  {row['first_passage_id'][:26] if row['first_passage_id'] else '-'}...")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hi", help="dataset language config, e.g. hi")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default=PINNED_REVISION, help="40-character dataset commit SHA")
    parser.add_argument("--limit", type=int, default=5, help="records for the tiny diagnostic")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--skip-network", action="store_true", help="local checks only")
    args = parser.parse_args()

    started = time.perf_counter()
    report = Report()
    print(f"Preflight: {DATASET_ID} {args.config}/{args.split} @ {args.revision[:12]} "
          f"(language file {resolve_language_code(args.config)})")

    check_environment(report)
    check_outputs(report, args.output_dir, args.manifest_dir)
    check_resources(report, args.output_dir)

    facts: dict = {}
    records: list[dict] = []
    if args.skip_network:
        print("\nNetwork checks skipped (--skip-network)")
    elif report.failed:
        print("\nNetwork checks skipped: fix the local failures above first")
    else:
        facts.update(check_hub(report, args.config, args.split, args.revision))
        if not report.failed:
            parquet_facts, records = check_parquet(report, args.config, args.split, args.revision, args.limit)
            facts.update(parquet_facts)
            if records:
                check_records(report, records, args.config, args.split)

    elapsed = time.perf_counter() - started
    print(f"\n{'-' * 72}")
    status = "FAILED" if report.failed else ("PASSED WITH WARNINGS" if report.warned else "PASSED")
    print(f"PREFLIGHT {status}: {len(report.checks)} checks, {report.failed} failed, "
          f"{report.warned} warnings, {elapsed:.1f}s, peak RSS {peak_rss_mb():.0f} MB")

    manifest = {
        "dataset_id": DATASET_ID, "revision": args.revision, "config": args.config, "split": args.split,
        "timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}", "limit": args.limit,
        "status": status, "elapsed_seconds": elapsed, "peak_rss_mb": peak_rss_mb(),
        "measurements": facts, "checks": report.checks,
    }
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_dir / f"preflight-{args.config}-{args.split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"report written to {manifest_path}")
    if report.failed:
        print("Do NOT start a corpus build until every check passes.")
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
