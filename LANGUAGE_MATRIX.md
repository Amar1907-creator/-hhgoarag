# Language Support Matrix

## All 14 MSMARCO-XI Languages

| Code | Language | Native | Script | Direction | Speech Locale | Browser Engine | Corpus | Train Data | Status |
|------|----------|--------|--------|-----------|---------------|----------------|--------|-----------|--------|
| as | Assamese | অসমীয়া | Bengali | ltr | None | ❌ | ❌ | ✓ | Not built, no speech |
| bn | Bengali | বাংলা | Bengali | ltr | bn-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| gu | Gujarati | ગુજરાતી | Gujarati | ltr | gu-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| **hi** | **Hindi** | **हिन्दी** | **Devanagari** | **ltr** | **hi-IN** | **⚠️ Untested** | **✓** | **✓** | **Built & Ready** |
| kn | Kannada | ಕನ್ನಡ | Kannada | ltr | kn-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| ml | Malayalam | മലയാളം | Malayalam | ltr | ml-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| mr | Marathi | मराठी | Devanagari | ltr | mr-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| ne | Nepali | नेपाली | Devanagari | ltr | ne-NP | ⚠️ Untested | ❌ | ✓ | Not built |
| or | Odia | ଓଡ଼ିଆ | Odia | ltr | None | ❌ | ❌ | ✓ | Not built, no speech |
| pa | Punjabi | ਪੰਜਾਬੀ | Gurmukhi | ltr | pa-Guru-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| sa | Sanskrit | संस्कृतम् | Devanagari | ltr | None | ❌ | ❌ | ✓ | Not built, no speech |
| ta | Tamil | தமிழ் | Tamil | ltr | ta-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| ur | Urdu | اردو | Arabic | rtl | ur-IN | ⚠️ Untested | ❌ | ✓ | Not built |
| te | Telugu | తెలుగు | Telugu | ltr | te-IN | ⚠️ Untested | ❌ | ○ | Validation only, not built |

## Legend

- **Code:** Two-letter language code (ISO 639-1)
- **Script:** Unicode script block name
- **Direction:** Text direction (ltr = left-to-right, rtl = right-to-left)
- **Speech Locale:** BCP-47 language tag for browser Web Speech API
- **Browser Engine:** ✓ confirmed, ⚠️ untested (attempted), ❌ unavailable
- **Corpus:** ✓ built, ❌ not built
- **Train Data:** ✓ in pinned MSMARCO-XI revision, ○ validation-only, ❌ absent
- **Status:** Current state

## Implementation Notes

1. **Hindi (hi):** Production-ready. Corpus built, indexed, and deployed.
2. **Other 13 languages:** All 14 registered in `src/languages.py`. Language selector shows all with clear "not built" indicator for unbuilt languages.
3. **Speech:** BCP-47 tags are attempts, not guarantees. Browser determines support.
4. **No false claims:** Unbuilt languages do not report metrics or enabled speech support.

## Build Requirements

To build a corpus for any language:
1. Pinned MSMARCO-XI Parquet file must be available (requires Hugging Face access)
2. Corpus file is ingested → chunked → embedded → indexed
3. Embedding model: `multilingual-e5-small` (100M parameters)
4. FAISS index: HNSW, L2-normalised float32 vectors
5. Each language gets an artifact directory: `data/corpus/{code}-*`

## Unbuilt Languages

When a language is not built:
- Language selector shows it as disabled
- Help text displays: "Corpus not built yet"
- No metrics are claimed
- Selecting it raises an error (API level)
- Voice engine availability is probed at runtime, not claimed statically

## Speech Engine Testing

Current status for browser Web Speech API:
- Hindi (hi): Untested but locale available
- Bengali (bn), Gujarati (gu), Kannada (kn), Malayalam (ml), Marathi (mr), Nepali (ne), Punjabi (pa), Tamil (ta), Telugu (te), Urdu (ur): Untested
- Assamese (as), Odia (or), Sanskrit (sa): No known browser engines

To verify speech for a language:
1. Open the app in a browser
2. Select the language
3. Click the microphone button
4. Browser will request microphone permission
5. Attempt to record audio
6. Check console for Web Speech API success/failure

