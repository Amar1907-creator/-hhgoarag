# Objective

Establish the smallest credible local multilingual dense-retrieval baseline before lexical, hybrid, reranking, generation, or service infrastructure is introduced.

# Corpus

**MEASURED:** The pinned source revision is `bf5cdc1f26e581e519018e434db14edd1b77602b`; it has 10,080,140 train records and 1,371,174 validation records. No Phase 1 train corpus JSONL has yet been materialized, so **no passages have been indexed** in this phase.

The benchmark accepts only a Phase 1 canonical corpus made from the train split. It stores stable `passage_id` values in the index and keeps language/provenance in the JSONL metadata sidecar. Validation queries, answers, and relevance flags are never embedded as documents.

# Evaluation Set

An evaluation JSONL row must contain `query_id`, `query`, and `positive_passage_ids`, derived only from validation `is_selected` labels joined through the separate provenance artifact. Telugu must be excluded from the default train-corpus benchmark: it has validation data but no training file in the pinned revision. A Telugu cross-lingual experiment is a separate declared protocol, never the default.

# Embedding Candidates

| Candidate | Coverage / dimension | Local-fit assessment | Status |
| --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` | multilingual E5; 384 dimensions | Smallest credible local starting point | configured, not corpus-tested |
| `intfloat/multilingual-e5-base` | multilingual E5; 768 dimensions | higher index/model cost | deferred |
| `BAAI/bge-m3` | multilingual; 1024 dimensions | substantially higher footprint and complexity | deferred |

No candidate has been quality-tested yet; claiming a selected model or quality winner would be false. The harness uses E5's required `query: ` / `passage: ` prefixes and normalizes vectors.

# Selected Embedding

**PROVISIONAL:** `intfloat/multilingual-e5-small`, subject to retrieval evaluation against a train-only corpus. It is a deployment-oriented first candidate, not a measured winner.

# Index Design

Local FAISS `IndexHNSWFlat` with inner-product metric over normalized `float32` vectors. It is built offline, persists as `index.faiss` plus an ordered `ids.json` mapping, and performs query embedding -> ANN search -> sidecar metadata resolution. FAISS is intentionally embedded, not a service.

# Retrieval Metrics

For validation queries with at least one explicit positive passage ID: Recall@K is the fraction whose top-K contains any positive. MRR is the mean reciprocal rank of the first positive in the retrieved ranking. These are not computed until provenance has been joined to canonical IDs.

# Latency Methodology

The benchmark warms model/index with a configured number of queries, then reports min, mean, p50, p70, and p100 separately for query embedding, ANN search, metadata lookup, and their total. Cold-start is intentionally excluded from warm-query measurements and must be captured in a separate process-start experiment.

# Benchmark Results

**MEASURED:** No corpus benchmark has run; therefore Recall@K, MRR, latency, model memory, index size, and metadata size are unmeasured.

# Memory/Storage

**ESTIMATED:** A 384-dimensional float32 vector uses 1,536 bytes before FAISS graph overhead. Multiply this by the measured final unique passage count; do not substitute source-record count. HNSW adds graph memory and persisted index overhead that must be measured from the built artifact.

# Limitations

Every train-language source Parquet file is one 3.3–4.0 GB row group. The loader reads pinned remote Parquet files directly with PyArrow batch iteration; bounded samples and declared corpus builds must still respect the on-disk size of a full language file when `--limit` is omitted. A synthetic fixture would not be a valid MSMARCO-XI benchmark.

# Baseline Decision

The code-level baseline is FAISS HNSW plus multilingual E5-small, but it is **not yet an accepted measured baseline**. First materialize one declared train-language corpus, derive its matching validation evaluation mapping, then run `scripts/benchmark_retrieval.py --rebuild` with at least 100 queries. Only then choose between E5-small and one larger candidate based on quality, warm latency, and artifact size.
