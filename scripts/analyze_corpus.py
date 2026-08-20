#!/usr/bin/env python3
"""Summarize a built JSONL corpus without loading it into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("corpus", type=Path); args = parser.parse_args()
    languages = Counter(); count = 0
    with args.corpus.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); count += 1; languages[row["language"]] += 1
    print(json.dumps({"unique_passages": count, "passages_per_language": languages}, indent=2))


if __name__ == "__main__": main()
