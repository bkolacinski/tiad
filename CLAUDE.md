# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

Two independent university assignments, each a standalone Python desktop app with its own virtual environment and build scripts:

- `zadanie1/` — XLSX-to-DOCX/PDF converter (tkinter GUI)
- `zadanie2/` — Recipe Voice Filter (main project, CustomTkinter GUI)

## zadanie2 — Recipe Voice Filter

### Development Commands

```bash
# First-time setup (downloads Whisper model ~460MB, spaCy model, translation packages, builds recipe index)
python setup.py

# Run the application
python app.py

# Run smoke tests (ingredient extraction, recipe search, translation)
python smoke_test.py

# Build standalone Windows EXE → release\RecipeVoiceFilter\RecipeVoiceFilter.exe
build.bat
```

### Architecture (Pipeline)

The app processes: **audio → transcription → ingredient extraction → recipe search → display**

| Module | Role |
|---|---|
| `stt.py` | Records mic or loads file, transcribes with Whisper (`small.pt`), detects language |
| `ingredient_extractor.py` | Extracts ingredients via spaCy Polish NLP (falls back to regex); fuzzy-deduplicates with rapidfuzz |
| `recipe_matcher.py` | Loads 15 JSON recipe files, builds TF-IDF index (cached to `recipe_index_cache.pkl`), searches in `"any"` (similarity-ranked) or `"all"` (strict, auto-relaxes) modes |
| `translator.py` | Offline translation via Argos Translate packages in `translation_packages/`; raises `TranslationUnavailable` if a pair is missing |
| `text_utils.py` | Polish token normalization, suffix-stripping for lemmatization, fuzzy token matching |
| `app.py` | CustomTkinter dark-themed GUI; all ML operations run in background threads |

### Key Design Constraints

- **Fully offline**: Whisper model in `models/`, Argos Translate packages in `translation_packages/`, recipe data in `data/`. None of these are in git (`.gitignore` excludes `models/`, `data/`, `*.pt`).
- **Polish-language optimized**: Custom suffix stripping in `text_utils.py`, spaCy `pl_core_news_sm` model, 48-word stoplist in both `ingredient_extractor.py` and `recipe_matcher.py`.
- **Cache invalidation**: `recipe_matcher.py` checks recipe JSON file timestamps against `recipe_index_cache.pkl`; delete the `.pkl` to force rebuild.
- **KMP_DUPLICATE_LIB_OK=TRUE** must be set before importing numpy/whisper (done in `app.py` and `build.bat`).
- Build uses `RecipeVoiceFilter.spec` for PyInstaller; hidden imports and `collect_all` directives are already configured there.

### Language Pairs (translator.py)

Polish ↔ English ↔ German/French/Spanish. Multi-hop translation goes through English as pivot.
