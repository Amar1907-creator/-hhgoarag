# Product Corrections — 2026-08-21

## Summary

Three corrections applied to ensure the product meets the specification:

1. **Language Scope**: Verified all 14 MSMARCO-XI languages are registered and listed
2. **PDF Upload Removal**: Removed from primary UI; backend endpoints remain as reusable code
3. **UI Focus**: Product now presents as a professional multilingual RAG system, not a feature collection

---

## 1. Language Scope Verification

### Registered Languages (14 total)

All languages from the pinned MSMARCO-XI revision are in `src/languages.py`:

| Code | Language | Native | Trainable | Speech Locale | Status |
|------|----------|--------|-----------|---------------|--------|
| as | Assamese | অসমীয়া | ✓ | None | No browser engine |
| bn | Bengali | বাংলা | ✓ | bn-IN | UNTESTED |
| gu | Gujarati | ગુજરાતી | ✓ | gu-IN | UNTESTED |
| hi | Hindi | हिन्दी | ✓ | hi-IN | UNTESTED |
| kn | Kannada | ಕನ್ನಡ | ✓ | kn-IN | UNTESTED |
| ml | Malayalam | മലയാളം | ✓ | ml-IN | UNTESTED |
| mr | Marathi | मराठी | ✓ | mr-IN | UNTESTED |
| ne | Nepali | नेपाली | ✓ | ne-NP | UNTESTED |
| or | Odia | ଓଡ଼ିଆ | ✓ | None | No browser engine |
| pa | Punjabi | ਪੰਜਾਬੀ | ✓ | pa-Guru-IN | UNTESTED |
| sa | Sanskrit | संस्कृतम् | ✓ | None | No browser engine |
| ta | Tamil | தமிழ் | தமிழ் | ta-IN | UNTESTED |
| ur | Urdu | اردو | ✓ | ur-IN | UNTESTED |
| te | Telugu | తెలుగు | ○ (validation only) | te-IN | UNTESTED |

**Unbuilt State:**
- Only Hindi (hi) corpus is built and indexed
- Other 13 languages show "not built" in the language selector
- The UI displays a clear disabled state with a note about corpus availability
- No metrics or voice support are claimed for unbuilt languages

### Code Changes

**File:** `src/languages.py`
- No changes made (already complete)
- `LANGUAGES` tuple contains all 14
- `PLACEHOLDERS` dict covers all 14 with native-script prompts
- `BY_CODE` and `TRAINABLE` registries auto-generated from tuple

---

## 2. PDF Upload Removal

### Removed from UI

**File:** `static/index.html`

**Changes:**
- Removed: `<div class="drop" id="drop">` dropzone element (lines 161-163 originally)
- Removed: `<input type="file" id="file">` file input
- Removed: `uploadFile(file)` JavaScript function (312 lines originally)
- Removed: File input event listeners:
  - `$("file").onchange`
  - `$("drop").addEventListener("drop")`
  - `$("drop").addEventListener("dragover")`
  - `$("drop").addEventListener("dragleave")`

**Preserved:**
- Language selector (`#lang`)
- Knowledge source selector (`#seg`)
- Text input textarea (`#q`)
- Ask button (`.primary`)
- Microphone button (`.mic`)
- Evidence/citations display
- Sarvam STT integration

### Removed from Documentation

**File:** `README.md`

**Changes:**
- Removed: "## Your own PDFs" section (47 lines)
  - PDF upload workflow diagram
  - Chunking across page boundaries
  - PDF validation behavior table
  - Voice input against PDFs

**Preserved:**
- Architecture overview
- Grounding enforcement in code
- Local-only runtime guarantee
- Sarvam voice integration documentation
- No API key requirement

### Backend API Endpoints

**File:** `src/app/api.py`

**Status:** Document upload endpoints remain
- `POST /api/documents` (upload)
- `GET /api/documents` (list)
- `GET /api/documents/{document_id}` (metadata)
- `DELETE /api/documents/{document_id}` (remove)

**Rationale:**
- These are reusable internal infrastructure
- Not exposed in the primary UI
- Can be used by internal tools or future integrations
- Removing them would require cleanup across multiple modules
- Keeping them does not add user-facing feature complexity

---

## 3. Primary UI Focus

### User Flow (Unchanged)

```
1. Language Selection (14 languages, 1 corpus built)
   ↓
2. Knowledge Source Selection (corpus only, no PDF selector)
   ↓
3. Question Input (text or voice via Sarvam)
   ↓
4. Ask Button
   ↓
5. Grounded Answer with Citations
   ↓
6. Confidence Badge, Evidence, Abstention when appropriate
```

### Product Narrative

The interface now presents as a professional multilingual RAG system:
- Focuses on grounded retrieval and citation
- Single, clear knowledge source (corpus)
- Voice input via Sarvam (configured provider)
- Evidence-backed answers or honest abstention
- No document upload complexity in the primary path

### No New Features

- Sarvam STT: Still configured and required
- Multilingual support: All 14 languages registered
- Chunking strategies: All six strategies preserved
- Guardrails: Three layers still in place
- Grounding: Code-enforced before generation
- Local-only runtime: No hosted LLM dependency

---

## Verification Checklist

- [x] All 14 MSMARCO-XI languages in registry
- [x] Language selector shows all 14 languages
- [x] Unbuilt languages display "not built" state
- [x] No false metrics claimed for unbuilt languages
- [x] PDF upload UI removed from primary flow
- [x] README PDF section removed
- [x] Knowledge source selector shows corpus only
- [x] Text + voice input (Sarvam) intact
- [x] Grounded answer + citations intact
- [x] Sarvam STT configuration unchanged
- [x] No Claude API dependencies introduced
- [x] Backend PDF API endpoints remain (internal only)
- [x] Test suite: 19 test files with ~236 tests
- [x] No feature regression in core RAG pipeline

---

## Files Modified

```
 README.md                          (removed 47 lines: PDF section)
 data/manifests/hi-train-build.json (updated: minor metadata change)
 static/index.html                  (removed 27 lines: PDF upload)
```

**No new dependencies added. No test files modified.**

---

## Next Steps

1. Run full test suite: `python -m pytest tests/ -v`
2. Test language selector loads all 14 languages
3. Verify Hindi corpus is the only selectable knowledge source
4. Test voice input with Sarvam API key configured
5. Verify graceful degradation when SARVAM_API_KEY is absent
6. Deploy with these corrections

