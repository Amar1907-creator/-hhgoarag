# HHGOARAG release audit

`NOT READY` — 6 passed, 3 failed, 1 skipped, 0s

Generated 2026-08-21T19:19:22+00:00 on Linux aarch64, device `cpu`.

Every number below was measured by this run.

| Status | Area | Check | Detail |
|---|---|---|---|
| SKIP | tests | unittest discover | skipped by flag |
| PASS | independence | no hosted LLM reference | none in src/, scripts/, static/, run.sh, requirements, README |
| PASS | independence | runs without an API key | no key is read by the product |
| FAIL | artifacts | corpus/index alignment | ModuleNotFoundError: No module named 'faiss' |
| PASS | artifacts | validation errors | 0 |
| PASS | artifacts | evaluation coverage | 100.0% query coverage, 93 queries |
| PASS | artifacts | recorded retrieval metrics | R@1 0.2473 R@5 0.5914 R@10 0.7204 MRR 0.3948 |
| PASS | startup | device selection | cpu (mps available: False) |
| FAIL | startup | service load | RuntimeError: install sentence-transformers to use a production embedder |
| FAIL | startup | service load | SystemExit: 1 |

## Retrieval quality

- Corpus: 0 passages (`hi-train-5k`)
- Evaluation: 93 queries, 100.0% query coverage, 100.0% positive coverage
- Recall@1 0.2473 · Recall@5 0.5914 · Recall@10 0.7204 · MRR 0.3948
- Startup: 0.0s

## Independence

- Hosted-LLM references in shipped code: none
- Local generation: no Ollama model; extractive fallback

