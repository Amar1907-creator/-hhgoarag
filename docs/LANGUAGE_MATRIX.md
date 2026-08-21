# HHGOARAG language matrix

Generated 2026-08-21T19:59:35+00:00. Every figure comes from a manifest written by a real run; nothing here is estimated.

| Language | Script | Corpus | Queries | Coverage | Recall@5 | Recall@10 | MRR | Voice | Status |
|---|---|---|---|---|---|---|---|---|---|
| অসমীয়া · Assamese | Bengali | — | — | — | — | — | — | none known | not built |
| বাংলা · Bengali | Bengali | — | — | — | — | — | — | bn-IN (untested) | not built |
| ગુજરાતી · Gujarati | Gujarati | — | — | — | — | — | — | gu-IN (untested) | not built |
| हिन्दी · Hindi | Devanagari | 49,511 | 93 | 100.0% | 0.5914 | 0.7204 | 0.3948 | hi-IN (untested) | ready |
| ಕನ್ನಡ · Kannada | Kannada | — | — | — | — | — | — | kn-IN (untested) | not built |
| മലയാളം · Malayalam | Malayalam | — | — | — | — | — | — | ml-IN (untested) | not built |
| मराठी · Marathi | Devanagari | — | — | — | — | — | — | mr-IN (untested) | not built |
| नेपाली · Nepali | Devanagari | — | — | — | — | — | — | ne-NP (untested) | not built |
| ଓଡ଼ିଆ · Odia | Odia | — | — | — | — | — | — | none known | not built |
| ਪੰਜਾਬੀ · Punjabi | Gurmukhi | — | — | — | — | — | — | pa-Guru-IN (untested) | not built |
| संस्कृतम् · Sanskrit | Devanagari | — | — | — | — | — | — | none known | not built |
| தமிழ் · Tamil | Tamil | — | — | — | — | — | — | ta-IN (untested) | not built |
| اردو · Urdu | Arabic | — | — | — | — | — | — | ur-IN (untested) | not built |
| తెలుగు · Telugu | Telugu | — | — | — | — | — | — | te-IN (untested) | no train data in this revision |

**1 of 14 languages are evaluated and ready**; 1 have a corpus and index built.

Build another language with:

```bash
python3 scripts/run_pipeline.py --language ta --limit 5000
# or several at once
python3 scripts/build_languages.py --languages ta,bn,mr
```

A dash means the run that would produce that number has not happened. Metrics are withheld entirely when evaluation coverage is below 95%, because below that they measure corpus coverage rather than retrieval quality.

Voice locales are what the browser is asked to use. `untested` means no human has confirmed dictation in that language yet; the interface disables the microphone and says so if the browser rejects the locale.
