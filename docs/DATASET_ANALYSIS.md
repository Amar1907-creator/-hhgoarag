# Dataset Overview

This report covers `ai4bharat/MSMARCO-XI` at immutable revision `bf5cdc1f26e581e519018e434db14edd1b77602b`, inspected on 2026-08-20. It is a translated MS MARCO-style dataset for Indic-language question answering, passage ranking, and RAG experimentation. Remote Parquet-footer inspection measured **11,451,314 rows** in **55,619,599,557 bytes** (55.62 GB decimal) across 27 files. Its size tag is 10M–100M rows.

The dataset has **10,080,140 train** rows and **1,371,174 validation** rows. No test split is present in the pinned repository.

## Schema

Each row is a translated version of a source MS MARCO QA record. The documented fields are:

| Field | Documented type | Runtime role |
| --- | --- | --- |
| `source_lang` | string | provenance; retain as metadata |
| `target_lang` | string | language filter/routing metadata |
| `meta` | dictionary | translation provenance; retain offline/audit only |
| `query` | string | query/evaluation input; do not index as a corpus document |
| `Answer` | string | reference answer; evaluation only |
| `query_id` | integer | query/evaluation join key; metadata only |
| `query_type` | string | analysis/evaluation metadata |
| `passages.is_selected` | list (0/1 labels) | passage relevance labels; evaluation/training only |
| `passages.English_passages` | list of strings | English corpus text; index only in an explicit English/cross-lingual variant |
| `passages.Translated_passages` | list of strings | target-language corpus text; primary index text |
| `Eng_Query` | string | source English query; evaluation/cross-lingual analysis only |
| `Eng_Answer` | string | source English reference answer; evaluation only |

Remote schema metadata confirms every listed scalar string, `query_id` as `int64`, all documented translation numeric settings as `int64`, and the three passage fields as lists (`is_selected` list of `int64`; both passage arrays list of strings). The card calls `Answer` a string; it does not document it as the original MS MARCO list-of-answers structure. Do not silently rename or assume an `answers` field.

## Splits

The pinned Parquet repository has two splits: `train` (13 files, 10,080,140 rows) and `validation` (14 files, 1,371,174 rows). Validation has 97,941 rows per language file. Training counts vary: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Punjabi, Sanskrit, and Tamil have 778,638 each; Marathi 765,873; Nepali 754,154; Odia 782,282; Urdu 770,089. These facts are recorded in [`data/manifests/repository-inventory.json`](../data/manifests/repository-inventory.json).

## Languages

The listed target languages are Assamese (`as`), Bengali (`bn`), Gujarati (`gu`), Hindi (`hi`), Kannada (`kn`), Malayalam (`ml`), Marathi (`mr`), Nepali (`ne`), Odia (`or`), Punjabi (`pa`), Sanskrit (`sa`), Tamil (`ta`), Telugu (`te`), and Urdu (`ur`): **14 languages**. `source_lang` is English (`eng_Latn` in the card example); `target_lang` uses script-aware codes such as `asm_Beng`.

The file inventory confirms all 14 language files in validation, but only 13 in train: **no Telugu training Parquet file exists at the pinned revision**. Language-level record counts should be derived from this inventory, not assumed balanced. The abbreviated file codes (`asm`, `ben`, etc.) are file naming metadata; joining them to `target_lang` codes still requires a record-level verification run.

The repository does not provide accessible per-language row counts. Since all target rows originate from English records, records are intended to be cross-language aligned through the source record / `query_id`; that alignment should be verified by comparing the set of `query_id` values and candidate-passage positions across configurations before relying on it operationally.

## Data Relationships

A row has one query and one translated answer, alongside parallel lists of candidate passages and relevance labels. The lists are positional: `is_selected[i]` labels `English_passages[i]` and `Translated_passages[i]`. The example contains ten candidates, but the card does not state a universal cardinality; validate list lengths and alignment in ingestion.

This is a candidate-passage dataset, not automatically a global, independently identified document corpus. A practical retrieval corpus must be built by exploding the passage lists, deduplicating exact text within each language, and retaining a stable generated `passage_key` plus all source-row/query provenance. Treat a selected candidate as a positive relevance judgment for that row, not as evidence that all unselected passages are irrelevant globally.

## Passage Analysis

The public dataset viewer was unavailable (its job manager reported a crash). More importantly, each inspected training Parquet file is a single row group of approximately 3.3–4.0 GB; metadata-only access can establish rows and schema but cannot calculate text distributions. Therefore no passage or query length distribution, percentiles, language-specific lengths, exact duplicate count, or near-duplicate estimate is reported here.

Phase 1 must measure Unicode-character length, whitespace-token length, model-token length, sentence count, candidate-list length, empty/null rate, exact deduplication rate, and MinHash/SimHash near-duplicate rate by language and split. Report p50/p90/p95/p99 and max, not just averages.

## Metadata

Runtime metadata per indexed passage should include: `target_lang`, generated `passage_key`, source `query_id` values, candidate position(s), selection label(s) only in the evaluation artifact, and a content hash. Keep source/target language and translation metadata in the offline manifest for provenance. Do not place long answers, queries, or translation-generation parameters in the hot-path vector record unless a demonstrated use requires them.

## Data Quality

Important validation checks:

- Parallel lists must have equal lengths; quarantine malformed rows.
- Preserve Unicode and right-to-left handling for Urdu; normalize only presentation-equivalent forms and retain raw text.
- Deduplicate before embedding; the same web passage can recur across candidate lists and source rows.
- Track source versus translation text separately. Translation artifacts can affect lexical matching and answer grounding.
- Validate that every evaluation query has at least one selected candidate before computing recall metrics; report the excluded rate.

## Retrieval Implications

The retrievable unit is a translated candidate passage after deduplication, not the query or answer. Indexing queries risks train/evaluation leakage and answer indexing lets a generator retrieve labels rather than evidence. `Answer`, `Eng_Answer`, and `is_selected` are evaluation labels and must be excluded from a production retrieval index.

The selected-passage flags support closed-candidate retrieval evaluation: for each validation query, rank its candidate passages and compute Recall@k, MRR@10, and nDCG@10. They do not alone prove quality against a whole-web corpus. A stronger realistic evaluation should also use a deduplicated shared corpus and mark all known positives for each query.

## Chunking Implications

The source unit is already called a passage, so no second chunking should be assumed. First benchmark a **whole-passage** index. Only split passages that exceed a tokenizer-derived threshold. Test:

1. whole passage (baseline; preserves labels and minimizes index size),
2. sentence-boundary chunks with a small overlap for long passages, and
3. token-window chunks with fixed overlap as a control.

Each child chunk needs `parent_passage_key`, `chunk_index`, offsets, language, and text. A child is relevant when its parent is selected; answer evaluation must additionally check whether the chunk actually contains supporting evidence. Choose thresholds after measured model-token percentiles, not before.

## Multilingual Considerations

A single multilingual embedding model is required for target-language queries and passages. It should support all fourteen languages and relevant scripts; test language-wise retrieval rather than relying on aggregate scores. Language-aware filtering is expected to reduce search work and cross-script false matches for monolingual queries, but it can harm code-switched or cross-lingual queries. Detect language with a fast local classifier, use a confidence threshold, and fall back to an unfiltered multilingual search when confidence is low.

## Recommended Preprocessing

1. Pin the dataset revision and write an ingestion manifest with per-config/split counts and hashes.
2. Stream records; validate schema and positional list alignment.
3. Preserve raw text, create a conservatively normalized retrieval form, and record normalization version.
4. Explode translated passages; generate deterministic keys from language plus normalized text hash.
5. Exact-deduplicate within language, retaining all provenance and relevance mappings.
6. Measure lengths and duplicates before choosing any chunking policy.
7. Build split-safe corpus/evaluation artifacts so labels cannot enter the runtime index.

## Sources and Measurement Boundary

Dataset facts above come from the [official dataset card](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) and the pinned repository's remotely read Parquet footers, captured in [`data/manifests/repository-inventory.json`](../data/manifests/repository-inventory.json). All unobserved distributional quantities are intentionally marked unmeasured; they are not estimates.
