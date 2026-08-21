# Project status: hi-train-5k

Generated 2026-08-21T19:08:54+00:00

- **1. Dataset**: ai4bharat/MSMARCO-XI @ bf5cdc1f26e5 (hi/train)
- **2. Corpus size**: 49,511 unique passages from 5,000 records (0.75% duplicates)
- **3. Index size**: 49,511 vectors, 89.5 MB, FAISS HNSW inner-product
- **4. Model**: intfloat/multilingual-e5-small (384 dimensions)
- **5. Evaluation queries**: 93
- **6. Positive coverage**: 100.00% of 96 positive IDs; 100.00% of queries
- **7. Recall@5**: 0.5914
- **8. Recall@10**: 0.7204
- **9. MRR**: 0.3948
- **10. p50 latency**: 11.76 ms
- **11. p95 latency**: 33.83 ms
- **12. Answer generation**: implemented; no local model installed; extractive fallback in use (optional: ollama pull qwen2.5:3b-instruct); requires no API key
- **13. Tests**: skipped

## 14. Remaining limitations

- The evaluation set only contains validation queries whose gold passage is byte-identical to a passage in the train corpus. That is a biased sample of queries, so Recall here is a pipeline health measure, not a claim about Hindi retrieval quality in general.
- Corpus covers 5,000 of 778,638 Hindi train records.
- Single language (Hindi). No BM25, fusion, or reranking; dense retrieval only.
- Voice input uses the browser's own hi-IN recognition, so it needs Chrome, Edge or Safari; typing works everywhere.
- Generated answers need a local Ollama model; without one the system quotes retrieved evidence verbatim, which is still fully grounded.

## 15. Final demo command

```bash
./run.sh          # builds anything missing, then opens the app
```

Evaluation scan: 93 queries from 97,941 validation records (join rate 0.095%).
