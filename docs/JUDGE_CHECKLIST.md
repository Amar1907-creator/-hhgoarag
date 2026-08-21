# HHGOARAG — judge demonstration

**No API key. No account. No internet after startup.**
Disconnect the network once the app is running and every step below still works.

---

## Before the room

```bash
cd ~/hhgoarag
python3 scripts/release_audit.py        # ~3 min: runs every claim and writes docs/RELEASE_AUDIT.md
./run.sh --app-only                     # starts the app, opens the browser
```

The audit must end **`RELEASE READY`**. If it does not, the report names the
failing check. Optionally `ollama pull qwen2.5:3b-instruct` first — without it
answers quote the retrieved passage verbatim, which is still fully grounded and
still cites correctly.

---

## The five-minute sequence

**1. What it is (30s)**
Point at the status pill: passages indexed, device, generator. Say the one
sentence that matters: *retrieval decides what is true, the model only phrases
it, and every claim is tied to a passage you can read.*

**2. Strong retrieval (45s)**
Click the first demonstration chip. Show the **grounded · high confidence**
badge, the answer, and the cited passage underneath with its similarity score.
Expand **All retrieved passages** — every candidate with its score, nothing
hidden.

**3. Multiple evidence (45s)**
Click the chip labelled *answer drawn from several passages*. Two or more
citations, each independently checkable.

**4. Safe abstention (60s)** — *the important one*
Click the abstention chip (a private or non-existent-entity question). The
system returns **पर्याप्त प्रमाण उपलब्ध नहीं है** — insufficient evidence — states
why, and shows the best score it found. Nothing is invented. Say plainly: the
threshold is applied **before** the model is called, so no text was generated at
all.

**5. Your own PDF (90s)**
Drag `tests/fixtures/goa-task-2-sample.pdf` onto the upload panel (or any PDF a
judge hands you). Watch the stage strip: *extracting → chunking → embedding →
indexing → ready*, with page and chunk counts. It becomes a selectable
**Knowledge source**.

Ask: **गोवा का सबसे व्यस्त समुद्र तट कौन सा है?**

The **Sources** section reads **GOA Task-2.pdf — Page 7**. Open the PDF at page
7 and show that the sentence is there.

**6. Voice (45s)**
Click **बोलें · Voice**, speak a Hindi question, watch the recognised Devanagari
appear and the query run automatically. Voice works against the PDF exactly as
against the corpus.

**7. Close (15s)**
Ask the PDF something it cannot support — it abstains again. Uploading a PDF
never touches the Hindi corpus; both remain selectable.

---

## Questions a judge will ask

**"Does this call an API?"** No. Retrieval is FAISS on this machine; generation
is a local Ollama model, or verbatim evidence if none is installed. Speech is
the browser's own recogniser. There is no API key anywhere, and a test fails the
build if a hosted-LLM reference reappears in the shipped code.

**"How do you know it isn't making things up?"** Three mechanisms, all in code
rather than in a prompt: a similarity floor that abstains *before* generation; a
citation check that discards any answer whose citations are not in the retrieved
set; and confidence taken from retrieval similarity, never from the model.

**"Are those retrieval numbers real?"** They come from 93 validation queries at
100% coverage — every query's gold passage is in the indexed corpus. An earlier
build reported Recall@5 0.54 at 1.94% coverage; that number was meaningless, and
the tooling now refuses to display metrics below 95% coverage without stamping
them invalid.

**"What are the limits?"** Stated in the README and repeated here: the evaluation
sample is biased toward queries whose answer passage is byte-identical to a
corpus passage; the corpus is 5,000 of 778,638 available records; Hindi only;
dense retrieval only, no BM25 or reranking; scanned PDFs are refused rather than
OCR'd.

---

## If something misbehaves

| Symptom | Do this |
|---|---|
| App will not start | `python3 scripts/run_app.py` prints the reason; usually the corpus is missing → `./run.sh` |
| Answers quote instead of summarising | Expected without Ollama. `ollama serve` then `ollama pull qwen2.5:3b-instruct` |
| Microphone does nothing | Chrome, Edge or Safari only; type the question instead |
| A demo question misbehaves | `python3 scripts/pick_demo_questions.py --corpus data/processed/hi-train-5k-corpus.jsonl --index-dir data/processed/hi-train-5k-index --evaluation data/processed/hi-validation-5k-evaluation.jsonl` |
| PDF rejected | Read the message on the card — scanned, encrypted and corrupt files are refused deliberately, not silently |
