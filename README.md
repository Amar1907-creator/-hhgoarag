# HHGOARAG

**Hindi grounded retrieval-augmented answering.** Ask a question in Hindi, get an
answer that is either supported by cited passages from a real corpus, or an
honest refusal to answer.

Runs entirely on your machine. **No API key. No hosted API. No subscription.**

---

## The problem

A retrieval system that sounds confident is easy. A retrieval system that is
*checkable* is not. Two failure modes matter more than headline accuracy:

1. **Ungrounded answers.** A language model asked a question it cannot support
   from the evidence will usually produce something plausible anyway.
2. **Misleading metrics.** Recall@10 of 0.63 means nothing if only 2% of the
   evaluation queries have their answer passage inside the indexed corpus. That
   measures the corpus, not the retriever.

HHGOARAG treats both as first-class engineering concerns. Retrieval decides what
is true; the language model only phrases it. Every claim carries a citation that
is validated against the retrieved set, and the evaluation pipeline refuses to
report metrics when its coverage cannot support them.

## Architecture

```
Hindi question
     ↓
query embedding                 multilingual-e5-small, "query: " prefix, MPS when available
     ↓
FAISS HNSW search               inner product over L2-normalised float32 vectors
     ↓
top-k Hindi passages
     ↓
evidence selection              score floor, margin, item and character budgets
     ↓                          ── weak retrieval abstains here, before any generation
local language model            Ollama (qwen2.5), temperature 0, evidence-only prompt
     ↓
citation validation             cited IDs must exist in the retrieved evidence
     ↓                          ── an answer with only invented citations is discarded
Hindi answer + evidence + confidence
```

Two properties are enforced in code, not by prompting:

- **Abstention happens before generation.** If the best retrieval score is below
  the floor, no model is called at all.
- **Citations are numbers, not hashes.** The model cites `[1]`, `[2]`; those map
  back to passage IDs here. A model asked to copy 66-character hashes will
  eventually corrupt one, and a corrupted citation cannot be told apart from an
  invented one.

## Your own PDFs

Beyond the Hindi corpus, HHGOARAG answers questions about documents you upload.
Everything happens on your machine: `pypdf` is a pure-Python reader, and the
uploaded file is never transmitted anywhere.

```
PDF → validate → extract text per page → chunk (never across a page boundary)
    → embed → per-document FAISS index → retrieve → grounded answer
    → citation naming the actual page
```

In the interface: drop a PDF on the upload panel, watch it move through
**uploading → extracting → chunking → embedding → indexing → ready**, then pick
it in the **Knowledge source** selector and ask. The answer's **Sources**
section names the file and the exact pages used:

> **GOA Task-2.pdf** — `Page 7`

Every retrieved chunk carries its document ID, document name, page number,
chunk ID and text, so a citation always points at a page a human can turn to.
Chunks never span pages: that costs a little packing efficiency and buys the
page number being true.

**Ask by voice.** The microphone button uses the browser's built-in Hindi
speech recognition (`hi-IN`), so speech-to-text needs no service from us and no
key. Speak, watch the Hindi text appear, and it asks automatically. Voice works
against an uploaded PDF exactly as it does against the corpus.

**What happens with a difficult PDF**

| Input | Behaviour |
|---|---|
| Scanned / image-only | Refused with "appears to be scanned images… this build does not run OCR". No empty document is created. |
| Corrupted or truncated | Refused as damaged; the server stays up. |
| Password protected | Refused as encrypted. |
| Empty, or not a PDF | Rejected before any processing. |
| Very large | Truncated at a page and character budget, and the record says so. |
| Uploaded twice | Content-addressed: the same bytes return the existing index instead of re-embedding. |
| Hindi, English or mixed | All extracted; the fixture covers all three. |

Uploaded documents live under `data/documents/<document_id>/`, entirely apart
from the Hindi corpus artifacts. Uploading a PDF cannot touch the corpus, and
documents indexed in an earlier run are re-attached at startup.

### Document API

| Endpoint | Purpose |
|---|---|
| `POST /api/documents` | multipart PDF upload; returns immediately, ingestion continues in the background |
| `GET /api/documents` | all documents with status and progress |
| `GET /api/documents/{id}` | one document — poll this for progress |
| `DELETE /api/documents/{id}` | remove a document and its index |
| `GET /api/sources` | selectable knowledge sources |

`POST /api/query` takes an optional `source`: `"corpus"` or `"document:<id>"`.

```bash
curl -s -F file=@"GOA Task-2.pdf" localhost:8000/api/documents
curl -s localhost:8000/api/query -H 'content-type: application/json' \
  -d '{"question":"गोवा का सबसे व्यस्त समुद्र तट कौन सा है?","source":"document:doc_3a0305a3b5035904"}'
```

## Dataset

[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI),
pinned to revision `bf5cdc1f26e581e519018e434db14edd1b77602b`. Hindi
(`hin_Deva`) train split, 778,638 records in a single 3.72 GB Parquet row group.

Ingestion reads that file with bounded memory: `pre_buffer=False` plus an
explicit `buffer_size`, which is the difference between pulling the whole 3.72 GB
into ~4 GB of RSS and pulling a few tens of MB in range requests. See
`src/data/remote_parquet.py` — the numbers are in its docstring.

## Quick start

```bash
git clone <this repo> && cd hhgoarag
./run.sh
```

That single command installs dependencies, builds the corpus, evaluation set and
FAISS index if they are missing, picks demonstration questions from the real
data, and opens the application at <http://127.0.0.1:8000>.

First run takes roughly 20 minutes, unattended, and downloads about 500 MB.
Re-runs start in seconds because every stage is skipped when its artifacts exist.

### Optional: generated answers instead of quoted evidence

```bash
./run.sh --pull        # installs qwen2.5:3b-instruct (~2 GB) via Ollama
```

Without a local model the application still works end to end — it quotes the
best retrieved passage verbatim, which is never ungrounded. With one, it
synthesises a Hindi answer from the evidence and cites it. Nothing about this is
required for the demo.

Model preference order is in `src/rag/generator.py`; the strongest installed
model wins. `qwen2.5:7b-instruct` gives noticeably better Hindi if you have the
RAM for it.

## Building the data yourself

```bash
python3 scripts/preflight.py                    # verify environment and dataset access
python3 scripts/run_pipeline.py --limit 5000    # corpus → evaluation → index → benchmark
python3 scripts/project_status.py --prefix hi-train-5k
```

`run_pipeline.py` verifies between every stage and stops with a precise
recommendation if the evaluation set is too small to benchmark honestly. Each
stage is resumable: re-run the same command after an interruption.

Artifacts use a prefix (`hi-train-5k`), and every builder refuses to overwrite
artifacts it did not create, so experiments never clobber each other.

## Running the application

```bash
./run.sh --app-only              # skip building, start the server
python3 scripts/run_app.py       # equivalent, with more flags
```

| Flag | Meaning |
|---|---|
| `--prefix hi-train-15k` | serve a different corpus |
| `--generator extractive` | force evidence-only answers |
| `--device mps` | override device detection |
| `--top-k 20` | retrieve more candidates |
| `--port 8080` | bind elsewhere |

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness, corpus/index alignment, generator status |
| `GET /api/stats` | corpus, index, evaluation and latency figures |
| `GET /api/demo` | demonstration questions selected from real data |
| `POST /api/query` | `{"question": "..."}` → answer, citations, evidence, confidence, timings |

```bash
curl -s localhost:8000/api/query -H 'content-type: application/json' \
  -d '{"question":"मैनहट्टन परियोजना की सफलता का क्या प्रभाव पड़ा?"}' | python3 -m json.tool
```

The embedding model, FAISS index and passage store are loaded once at startup
and reused for every request.

## Example Hindi queries

| Question | Demonstrates |
|---|---|
| मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा? | strong single-passage evidence |
| मैग्नीशियम क्या है? | definition drawn from retrieved text |
| एक सख्त उबला हुआ अंडा कितने समय तक पकाते हैं? | procedural answer with citation |
| क्या बृहस्पति ग्रह पर मानव बस्तियाँ स्थापित हो चुकी हैं? | **safe abstention** — no supporting evidence |
| मेरे बैंक खाते में कितना पैसा है? | out-of-corpus refusal |

`scripts/pick_demo_questions.py` regenerates this list from whatever corpus is
actually built, so the demo never depends on a hand-guessed example.

## Evaluation

```bash
python3 scripts/verify_evaluation.py \
  --corpus data/processed/hi-train-5k-corpus.jsonl \
  --evaluation data/processed/hi-validation-5k-evaluation.jsonl
```

Reports Recall@1/5/10, MRR, and per-stage latency at p50/p95/p100, alongside
corpus size, evaluation query count, positive coverage and query coverage.

**Coverage gates the metrics.** When query coverage falls below 95%, Recall and
MRR are labelled `INVALID (low coverage)` in every report and the pipeline
refuses to proceed. This is not decoration: an earlier build had 994 passages,
1.94% positive coverage, and a Recall@5 of 0.54 that meant nothing at all.

**Known bias, stated plainly:** the evaluation set only contains validation
queries whose gold passage is byte-identical to a passage in the train corpus.
That is a biased sample. These numbers are a pipeline health measure, not a
claim about Hindi retrieval quality in general.

## Performance

Measured, not assumed. Ingestion configuration was chosen from an instrumented
byte-counting experiment (`src/data/remote_parquet.py` docstring); `to_pydict()`
was benchmarked against `to_pylist()` and column projection before being left
alone, because it was already the fastest safe option.

- Models and indexes load once at startup, never per request.
- Passage metadata uses a byte-offset index rather than an in-RAM dict:
  ~1 GB → ~160 MB at a million passages, at 0.09 ms per lookup.
- Index construction streams the corpus, checkpoints, and resumes.
- Apple Silicon MPS is detected and used automatically for embedding.

## Project structure

```
src/data/          pinned Parquet ingestion, normalization, dedup, schema
src/retrieval/     embedding, FAISS index, metadata sidecar, metrics
src/documents/     PDF extraction, page-aware chunking, per-document index, store
src/rag/           evidence selection, local generation, grounding, citations, sources
src/app/           service (loaded once) and HTTP API
static/            single-file web interface, no build step, no CDN
scripts/           preflight, builders, benchmark, pipeline driver, app, demo
tests/             170 tests across data, retrieval, RAG, documents, app
tests/fixtures/    a real 8-page Hindi/English PDF used by the document tests
data/processed/    generated corpora and indexes (git-ignored)
data/manifests/    reproducibility and measurement records (tracked)
data/documents/    uploaded PDFs, their chunks and indexes (git-ignored)
```

## Limitations

- Hindi only. The loader supports 13 languages; nothing else has been evaluated.
- Dense retrieval only — no BM25, fusion or reranking. Those were deliberately
  deferred until the dense baseline was measurable.
- The evaluation sample is biased as described above.
- The corpus covers a slice of the 778,638 available Hindi train records.
- Speech is the browser's own recognition, so voice input needs Chrome, Edge or
  Safari; there is a typed fallback everywhere.
- Scanned PDFs are refused rather than OCR'd. Adding OCR would mean a Tesseract
  dependency, and refusing clearly beats guessing badly.

## Release audit

```bash
python3 scripts/release_audit.py
```

Runs the whole test suite, verifies corpus/index alignment, loads the real
service, measures startup and per-stage latency, exercises the corpus flow, the
abstention path, PDF ingestion with page citation, restart persistence and
determinism, checks that no hosted-LLM reference exists in shipped code, and
verifies each demonstration question behaves as labelled. It writes
`docs/RELEASE_AUDIT.md` with every measured number and exits non-zero if any
check fails. Checks it cannot run are recorded as SKIP with the reason, never
as PASS.

## Judge demonstration

The full five-minute sequence is in [`docs/JUDGE_CHECKLIST.md`](docs/JUDGE_CHECKLIST.md).

```bash
./run.sh
```

Then, in the browser:

1. Click **मैनहट्टन परियोजना…** — a grounded answer with a cited passage and a
   high-confidence badge.
2. Expand **All retrieved passages** — every candidate with its similarity score.
3. Click the **बृहस्पति ग्रह** question — the system declines, states why, and
   shows the best score it found. Nothing is invented.
4. **Upload a PDF** (`tests/fixtures/goa-task-2-sample.pdf` works, or any of your
   own). Watch it move through the ingestion stages to **ready**.
5. Pick it in **Knowledge source**, click the **microphone**, and ask in Hindi:
   *"गोवा का सबसे व्यस्त समुद्र तट कौन सा है?"* The recognised Hindi appears in the
   box, the answer comes back grounded, and **Sources** reads
   **GOA Task-2.pdf — Page 7**.
6. Ask the same PDF something it cannot support — the system abstains.

No API key is entered at any point. Disconnect from the network after startup
and every one of these still works.
