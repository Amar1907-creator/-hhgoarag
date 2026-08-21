#!/usr/bin/env python3
"""Build the full pipeline for several languages with one command.

Each language runs the same pipeline into its own prefix, so a failure in one
never touches another's artifacts. Languages already built are skipped.

    python3 scripts/build_languages.py --languages ta,bn,mr
    python3 scripts/build_languages.py --all --limit 5000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.languages import TRAINABLE, get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="", help="comma-separated codes, e.g. ta,bn,mr")
    parser.add_argument("--all", action="store_true", help="every language with train data")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    args = parser.parse_args()

    if args.all:
        codes = [language.code for language in TRAINABLE]
    else:
        codes = [code.strip() for code in args.languages.split(",") if code.strip()]
    if not codes:
        raise SystemExit("pass --languages ta,bn or --all")

    languages = []
    for code in codes:
        try:
            language = get(code)
        except KeyError as exc:
            raise SystemExit(str(exc))
        if not language.has_train:
            print(f"skipping {language.english}: {language.note}", file=sys.stderr)
            continue
        languages.append(language)

    thousands = args.limit // 1000
    minutes = 25 * len(languages)
    print(f"Building {len(languages)} language(s) at {args.limit:,} records each.")
    print(f"Rough cost: {minutes} minutes and about {0.1 * len(languages):.1f} GB of index.\n")

    results = []
    for position, language in enumerate(languages, start=1):
        prefix = f"{language.code}-train-{thousands}k"
        built = (Path("data/processed") / f"{prefix}-index" / "index.faiss").exists()
        print(f"\n{'=' * 74}\n[{position}/{len(languages)}] {language.native} ({language.english})"
              f"{'  — already built, skipping' if built else ''}\n{'=' * 74}", flush=True)
        if built:
            results.append((language, "skipped", 0.0))
            continue
        command = [sys.executable, "scripts/run_pipeline.py", "--language", language.code,
                   "--limit", str(args.limit), "--skip-preflight"]
        if args.device:
            command += ["--device", args.device]
        started = time.perf_counter()
        outcome = subprocess.run(command, cwd=ROOT)
        elapsed = time.perf_counter() - started
        status = "built" if outcome.returncode == 0 else f"failed ({outcome.returncode})"
        results.append((language, status, elapsed))
        if outcome.returncode != 0 and not args.continue_on_error:
            break

    print(f"\n{'=' * 74}\nSummary\n{'=' * 74}")
    for language, status, elapsed in results:
        print(f"  {language.native:<12} {language.english:<11} {status:<14} "
              f"{elapsed / 60:.1f} min" if elapsed else
              f"  {language.native:<12} {language.english:<11} {status}")
    print("\nLanguage matrix:  python3 scripts/language_matrix.py")
    failures = [r for r in results if r[1].startswith("failed")]
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
