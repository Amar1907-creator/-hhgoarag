# Data artifacts

`processed/` is generated and must not be committed when it contains downloaded corpus data. A build produces a deduplicated runtime corpus JSONL and a separate provenance/evaluation JSONL. `manifests/` stores small reproducibility and measurement reports that may be committed.

Run `python3 scripts/inspect_repository.py --revision <40-char-sha>` to inventory Parquet footers, then `python3 scripts/validate_loader.py --config hi --revision <40-char-sha> --limit 1` to validate the pinned remote Parquet loader on a tiny sample. Bounded inspection and corpus builds use the same loader and require `huggingface_hub`, `pyarrow`, and a pinned revision.
