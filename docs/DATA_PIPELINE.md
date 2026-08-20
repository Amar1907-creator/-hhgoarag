# Reproducibility

The pipeline targets the official `ai4bharat/MSMARCO-XI` repository. A corpus build requires a 40-character Hugging Face dataset commit SHA; branch names such as `main` are rejected. Record the identifier, immutable revision, language configuration, split, command line, and processing timestamp in a run manifest before publishing results. The loader reads pinned remote language Parquet files through `huggingface_hub` plus PyArrow batch iteration, so it does not load the 55.6 GB source repository into RAM.

Install the pinned runtime with `python3 -m pip install -r requirements.txt`. Unit tests use only the standard library and do not need the dataset download.

The Phase 1 inventory is pinned to `bf5cdc1f26e581e519018e434db14edd1b77602b`; [`data/manifests/repository-inventory.json`](../data/manifests/repository-inventory.json) records the command's timestamp, exact file list, schema strings, rows, row groups, and byte sizes. It was produced using Hub metadata and remote Parquet footers only.

Run configurations are explicit: loader language configuration (for example `hi` or `hin`), split, revision, optional record limit, normalization version, and output paths. Given an unchanged revision/configuration, deterministic normalization and SHA-256 IDs produce the same corpus content. Output line order follows the streamed source order within a pinned Parquet file; consumers must use `passage_id` rather than line number.

# Validation and Error Handling

`src/data/schema.py` validates required documented fields, their types, parallel passage-list alignment, relevance flags, language metadata, and translation provenance object presence. It never mutates records. Invalid records are excluded from the runtime corpus but written to a validation-error JSONL artifact with source position and error reasons; their count and path are in the manifest.

Empty translated passages are counted and omitted from the retrievable corpus. Empty questions/answers are warnings because they matter to evaluation but are not required to build a passage corpus.

# Canonical Corpus and Deduplication

The runtime corpus contains only `passage_id`, translated `text`, `language`, a representative `source_id`, and normalization metadata. `passage_id` is `p_` plus SHA-256 of `language + NUL + normalized text`; it is stable across runs and avoids conflating identical text in different languages. Provenance/evaluation data is written separately: each source occurrence retains query ID, candidate position, split, and `is_selected` relevance label.

Exact deduplication is disk-backed SQLite, preventing a full in-memory set for millions of passages. The manifest reports total/unique/duplicate passages and duplicate percentage. Semantic deduplication is intentionally not implemented: it needs a demonstrated quality/cost justification.

# Length and Language Measurement

`inspect_dataset.py` reports min, mean, p50, p75, p90, p95, p99, and max for translated-passage length. The current method is character count after Unicode NFKC normalization and whitespace collapse, explicitly not model-token count. This is portable and inexpensive; add tokenizer-specific counts only when selecting an embedding/generation model.

The repository currently exposes the `default` loader configuration, not a language-specific configuration. A full per-language report must account for the actual 13 train and 14 validation files in the pinned inventory; Telugu has validation data but no train file. No language count should be inferred from the language list.

# Split Safety

Build a corpus per split and keep its evaluation mapping separate. The default safe retrieval experiment indexes only the chosen training corpus and evaluates with validation queries/positive labels. Do not index validation answers, queries, or relevance labels. Whether validation candidate passages may join a global corpus is an explicit alternate protocol, not the default: it risks evaluating a query against passages supplied alongside it and must be reported separately.

# Chunking Preparation

`src/data/chunking.py` exposes whole-passage, sentence-aware windows, overlapping sentence windows, and a caller-supplied semantic boundary interface. It does not choose a policy. Chunk identifiers and parent offsets will be added only when a benchmark selects a splitting strategy.

# Performance and Outputs

Inspection and build manifests record elapsed time and records/second. The streaming loader plus SQLite deduplication is designed for bounded process memory; measure actual RSS, CPU, source throughput, and generated-file size in the target deployment environment. The scripts deliberately do not claim throughput, memory, or corpus measurements until a pinned dataset run completes.
