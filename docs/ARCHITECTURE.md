# Engineering Goals

Build a voice-enabled, multilingual RAG system that is measurable, grounded, and deployable without premature framework complexity. The stated end-to-end latency target is under 200 ms, but this document makes no claim that it has been achieved. Speech-to-text and remote generation are likely to dominate that budget.

Primary objectives are correct evidence retrieval, transparent abstention, predictable tail latency, and a benchmarkable minimal stack. Every optional stage has an explicit acceptance test.

## Baseline Architecture

The strongest credible first baseline is:

`audio -> Sarvam or ElevenLabs STT -> local normalization/language detection -> multilingual dense top-k retrieval over deduplicated translated passages -> evidence packing -> grounded answer generation -> citation/grounding check -> response`

Use a small HTTP service with typed request/response objects and one explicit pipeline function. Store the index locally or colocated with the service. Do not introduce LangChain, LlamaIndex, agent loops, a distributed vector database, reranking, or multiple LLM calls until benchmarks show a specific need.

## Candidate Retrieval Strategies

| Option | Problem solved | Cost | Acceptance test |
| --- | --- | --- | --- |
| Dense multilingual ANN (baseline) | semantic matching across all scripts | encoder + ANN lookup | language-wise Recall@k/MRR/nDCG and latency |
| BM25 lexical | exact entities/numbers, cheap fallback | local lexical lookup and index storage | incremental quality over dense, especially entities |
| Reciprocal-rank fusion of dense + BM25 | complementary retrieval errors | two searches + merge | meaningful quality lift within budget |
| Cross-encoder reranker | improve top-k ordering | substantial model inference | lift at fixed evidence budget and acceptable p100 |

Begin with dense ANN. Add BM25 only if error analysis shows lexical misses; add fusion only if it beats dense decisively. Do not use remote reranking on the hot path.

## Chunking Strategies

Start with deduplicated whole passages. Evaluate whole-passage, sentence-aware long-passage chunks, and fixed token windows as documented in `DATASET_ANALYSIS.md`. Use the same corpus, query split, top-k, and embedding model for every comparison. Compare retrieval metrics, evidence coverage, index cardinality, embedding time, search p50/p70/p100, and generation-context tokens.

## Indexing Options

Use an embedded ANN index with cosine or inner-product search over normalized embeddings; select exact/brute-force for a small pilot and an ANN structure only after measuring corpus size and p100 latency. Keep a compact sidecar metadata store keyed by vector ID. Partition/filter by `target_lang` when language confidence is high, with fallback to global search.

An optional local BM25 index is the only initial lexical index candidate. A hosted vector database is not justified for a single deployable service unless corpus scale, concurrency, or operational requirements prove that an embedded index is insufficient.

## Offline Pipeline

1. Pin and stream dataset revision.
2. Validate rows and preserve an audit manifest.
3. Normalize and deduplicate passages.
4. Create the three controlled chunking artifacts.
5. Embed documents in batches; record model/version/dimension and failures.
6. Build vector index, optional BM25 index, and query-to-positive evaluation mappings.
7. Run retrieval and latency benchmarks; publish immutable result files.

This pipeline is offline. It must never execute during a user query.

## Query-Time Hot Path

1. Validate request size/media type and apply an audio duration cap.
2. Use **one** permitted STT provider (Sarvam or ElevenLabs) behind a small adapter with timeouts.
3. Normalize transcript conservatively; detect language locally.
4. Embed the text query and retrieve top-k candidates.
5. Apply deterministic relevance/evidence thresholds; pack only the best evidence within a token cap.
6. Generate an answer constrained to that evidence, then return citations keyed to passages.

Generation should receive the user question and enumerated evidence only; it must be instructed to say insufficient evidence when the context does not support an answer.

## Generation

Generation is optional in the retrieval benchmark and should be isolated from it. Use one model/provider configuration, low temperature, a fixed maximum output length, a hard timeout, and a structured response: `answer`, `citations`, `grounded`, `abstained`, `reason`, and timing fields. Compare extractive evidence-only answers with generated answers before choosing a model. Remote generation may make the end-to-end 200 ms goal infeasible; measure it separately and consider an extractive/templated answer fallback.

## Guardrails

Apply deterministic checks before and after model calls:

- Off-topic: classify/score against stated supported scope; clarify or decline outside it.
- Unsafe input: use provider moderation where available plus local policy rules; do not send disallowed content into an unnecessary generation path.
- Insufficient evidence: require minimum retrieval score, a score margin, and at least one eligible evidence item; otherwise abstain.
- Ungrounded answer: require citations, validate cited IDs are in the evidence set, and use a lightweight entailment/claim-support check only if its measured benefit exceeds its latency cost.

Guardrails must emit machine-readable reason codes and never fabricate a citation. Initially prefer abstention and citation membership validation over a second remote "judge" call.

## Harness

Implement a small explicit orchestration harness, not an agent framework: typed schemas; stage names; per-stage deadlines; bounded retries only for idempotent transient failures; cancellation propagation; structured error codes; request IDs; and timing spans. The pipeline should allow deterministic stub implementations for STT, retriever, and generator in tests.

Suggested input: `audio|text`, language hint, request ID, and debug flag restricted to trusted users. Suggested output: answer/abstention, evidence citations, stage timings, model/index versions, and error/recovery state. Define fallback paths in advance: STT timeout -> retry once or return retryable error; language uncertainty -> global index; retrieval miss -> abstain; generation timeout -> evidence-only response if safe.

## Latency Strategy

Instrument every stage with monotonic clocks: upload/decoding, STT, text normalization, embedding, retrieval, reranking, evidence packing, generation, guardrails, and serialization. Report end-to-end and per-stage p50, p70, and p100, along with sample count, concurrency, hardware/region, warm/cold state, index size, provider/model versions, timeout policy, and error rate.

Benchmark separately for text-only retrieval, text RAG, and full voice RAG. The first establishes what is locally controllable; the latter reveals provider network cost. Do not combine retries into a hidden success metric—report them.

## Benchmark Plan

Use a pinned validation split with language-stratified samples and an unseen held-out development subset for design decisions. Establish a full-corpus/deduplicated-corpus evaluation contract before reporting global retrieval quality. Evaluate Recall@1/3/5/10, MRR@10, nDCG@10, answer evidence coverage, abstention precision/recall, and unsafe/off-topic false decisions. For latency, collect enough repetitions per language and input-duration bucket to expose tails; publish raw timing logs or summaries with reproducible commands.

## Deployment

Package one service and one offline build job. Keep model/index artifacts versioned and colocate the ANN index with the query service. Put credentials only in environment-managed secret storage. Deploy only after the baseline is benchmarked; choose a region near the STT/generation provider and measure cold starts. A live demo can use text fallback when microphone/STT fails, but must clearly indicate the mode.

## Risks

- STT and remote generation network variance can exceed the total 200 ms target.
- Translation quality and script coverage can create uneven language retrieval quality.
- Candidate-passage relevance labels do not by themselves represent a global web-corpus benchmark.
- Duplicate candidate passages can inflate results or waste index capacity.
- Answer generation can sound plausible despite weak evidence; abstention must win over fluency.

## Decisions To Validate

1. Exact per-config/split counts, lengths, duplicates, and passage-list cardinality.
2. Multilingual embedding candidates and language-wise quality/latency.
3. Whether whole passages already fit model/context limits.
4. Dense-only versus BM25/fusion incremental value.
5. Whether reranking is worth its p100 cost.
6. STT provider language coverage, endpoint latency, and transcription quality.
7. Whether a generated answer can fit the end-to-end budget; if not, the appropriate evidence-first fallback.
