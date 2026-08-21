# HH Goa 2026 Task 2 — requirement map

Each requirement, where it is satisfied, and how to verify it.

## Pipeline shape

> Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation

```
microphone (MediaRecorder)
  → POST /api/transcribe  → Sarvam saarika          src/speech/providers.py
  → guardrail: input screening                      src/rag/guardrails.py
  → multilingual-E5 query embedding                 src/retrieval/embedding.py
  → FAISS vector search over chunked corpus         src/retrieval/index.py
  → evidence selection + score floor                src/rag/evidence.py
  → local generation (Ollama) or extractive         src/rag/generator.py
  → guardrail: citation + grounding check           src/rag/guardrails.py
  → answer + citations + confidence                 src/rag/pipeline.py
```

## 1. Speech-to-text — Sarvam or ElevenLabs

**Sarvam** (`saarika`), in `src/speech/providers.py`. Audio is recorded in the
browser and posted to `/api/transcribe`; the server calls the provider. An
ElevenLabs adapter is included and selectable with `HHGOARAG_STT_PROVIDER`, but
Sarvam is the default: it is built for Indian languages.

Configure with `SARVAM_API_KEY`. The browser's own recogniser remains only as a
clearly labelled offline fallback for development, and the interface shows which
path produced a transcript.

Verify: `GET /api/speech`, or `python3 -m unittest tests.app.test_speech` (16 tests
covering transport, retries, bad keys, empty transcripts, oversized audio).

## 2. Chunking — a considered strategy, not one naive scheme

Six strategies in `src/data/strategies.py`, all sharing one interface so they are
interchangeable by configuration:

| Strategy | Axis it explores |
|---|---|
| `whole` | no split — the passage is already a retrieval unit |
| `fixed` | fixed character windows with overlap; the meaning-blind baseline |
| `sentence` | sentence-packed to a target with a carried tail |
| `sliding` | overlapping sentence windows; every sentence appears more than once |
| `semantic` | boundaries where adjacent-sentence similarity dips below the passage mean |
| `metadata` | bounded by page/section, with the heading prepended into the chunk text |

Overlap handling is explicit in three of them. Semantic boundaries are
**relative** — a dip below this passage's own mean — so no threshold needs
recalibrating between an embedding similarity and a lexical one.

`scripts/benchmark_chunking.py` re-chunks the same passages six ways, embeds each
with the same model, indexes them identically and evaluates against the same
queries, crediting each retrieved chunk to its parent passage. Results in
`docs/CHUNKING_BENCHMARK.md`. The corpus ships with `whole` because MSMARCO
passages are already short and self-contained — a choice the benchmark tests
rather than assumes. PDF ingestion uses page-bounded sentence packing, because a
chunk that crosses a page cannot cite a page truthfully.

## 3. Latency target — under 200 ms

Retrieval measured at **P50 11.76 ms, P95 33.83 ms** on the 49,511-passage index.
`scripts/latency_report.py` reports the full path.

Stated plainly: the retrieval path — embedding, vector search, evidence
selection, guardrails — is comfortably inside 200 ms. A **local** language model
writing an answer is not, and no honest configuration makes it so. Both modes are
measured and reported separately rather than quoting only the flattering one.
The extractive mode is a complete grounded answer with citations and stays inside
the budget end to end.

## 4. Latency analytics — P50 / P70 / P100

`python3 scripts/latency_report.py --queries 100` → `docs/LATENCY.md`, per stage
and end to end, over 100 real evaluation queries after warm-up. Speech-provider
latency is returned with every transcript and reported separately, since it is a
network round trip rather than pipeline work.

## 5. Harness

`src/rag/pipeline.py` is an explicit staged orchestration, not a prompt call:

- named stages, each independently timed
- structured output at every boundary (`Verdict`, `Transcript`, `EvidenceSet`, `Answer`)
- retries with exponential backoff on the speech provider, classified into
  transient and permanent so a bad key is not retried
- generator fallback: local model → extractive, with the degradation surfaced
- error recovery: a failed upload, an unreachable model or a rejected locale
  degrade the feature, never the request
- an index build that checkpoints and resumes

## 6. Guardrails

Three layers, all deterministic and model-free — a guardrail that needs an LLM
adds the failure mode it exists to prevent:

1. **Input** (`screen_input`): unsafe requests, prompt injection, degenerate
   input, refused before an embedding is computed. Patterns require an *action*,
   not a topic: "how to make a bomb" is refused, "what did the atomic bomb do to
   the war?" is answered — with a test holding that line. Self-harm is checked
   first and answered with a support route rather than a flat refusal.
2. **Retrieval** (`select_evidence`): a similarity floor abstains when nothing
   supports the question. Against an open-domain corpus this *is* the off-topic
   test; a topic classifier would be the wrong instrument.
3. **Output** (`screen_output`): every citation must be in the retrieved set, and
   the answer's content words must overlap the cited evidence by ≥0.60 —
   calibrated on real answers, where faithful ones scored 0.83–1.00 and ones that
   drifted into outside knowledge scored 0.17–0.40.

Verify: `python3 -m unittest tests.rag.test_guardrails` (13 tests).

## Dataset

`ai4bharat/MSMARCO-XI`, pinned to `bf5cdc1f26e581e519018e434db14edd1b77602b`.
Hindi is built and evaluated: 49,511 passages, 93 evaluation queries at 100%
positive coverage, Recall@5 0.5914, Recall@10 0.7204, MRR 0.3948. The pipeline is
language-configured and the other twelve trainable languages build with one
command — see `docs/LANGUAGE_MATRIX.md`.

## Submission

- **Repo** — this repository.
- **Live link** — `Dockerfile` + `render.yaml`; see `docs/DEPLOY.md`.
- **Videos** — team process (90s) and demo. `docs/JUDGE_CHECKLIST.md` is the demo
  script.
- Every post on Instagram and X, by every member, must carry **#RAGInGoa**.
