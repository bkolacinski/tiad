# Recipe Voice Filter (Assignment 2)

Desktop app that **turns speech into text**, **extracts ingredient names**, and **searches a recipe database** scraped from **AniaGotuje.pl**. Everything runs **offline** after setup: Whisper for speech-to-text, spaCy for Polish ingredient phrases, scikit-learn for recipe ranking, Argos Translate for optional transcription translation.

---

## What it does

1. Record from the microphone or load an audio file.
2. **Whisper** transcribes audio and detects the spoken language.
3. **NLP + heuristics** turn the transcript into a deduplicated ingredient list (Polish-focused).
4. **TF-IDF search** over JSON recipes finds the best matches (“best match” vs “strict match” modes).
5. Optionally **translate** the transcript (offline packages) to another language.

---

## Main libraries


| Library                                   | Role                                                          |
| ----------------------------------------- | ------------------------------------------------------------- |
| **openai-whisper**                        | Speech-to-text (`small` model in `models/`)                   |
| **customtkinter** (+ **Pillow**)          | Modern GUI and image assets                                   |
| **sounddevice**, **soundfile**            | Microphone capture and audio file I/O                         |
| **numpy**, **scipy**                      | Numerical stack (Whisper / sklearn)                           |
| **scikit-learn**                          | TF-IDF vectorizer + similarity for recipe search              |
| **spacy** (`pl_core_news_sm`)             | Polish tokenization/entities for ingredient extraction        |
| **rapidfuzz**                             | Fuzzy deduplication and token matching                        |
| **argostranslate** (uses **ctranslate2**) | Offline translation; packages live in `translation_packages/` |
| **certifi**                               | TLS certificates for downloads during `setup.py`              |
| **pyinstaller**                           | Building the Windows folder-style EXE (dev/build only)        |


---

## Smoke test (`smoke_test.py`)

Quick **sanity check** (no pytest): loads the recipe index, runs a few ingredient-extraction cases, runs a search, and verifies translation packages respond. **Not required** for normal use, but useful **before a demo**, **after changing NLP/search code**, and **automatically** at the end of `build.bat` so a broken bundle is less likely. Run:

```bat
python smoke_test.py
```

---

## First-time setup (from source)

From the `zadanie2` folder, with Python 3.10+ on PATH:

```bat
python setup.py
```

This installs dependencies, downloads **Whisper `small`**, **spaCy Polish**, **Argos language pairs**, and builds the **recipe index cache** under `data/` (ignored by git; large files stay local).

Then:

```bat
python app.py
```

---

## Build the Windows EXE

```bat
build.bat
```

The script creates a temporary venv, runs `setup.py`, runs `smoke_test.py`, then **PyInstaller** (`--onedir --windowed`) and writes:

`release\RecipeVoiceFilter\RecipeVoiceFilter.exe`

### What’s inside the release folder


| Item                    | Purpose                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `RecipeVoiceFilter.exe` | Launches the app (no console window)                                 |
| `_internal\`            | Bundled Python runtime, Whisper, spaCy, sklearn, CustomTkinter, etc. |
| `data\`                 | Recipe JSON + `recipe_index_cache.pkl`                               |
| `models\`               | `small.pt` Whisper weights                                           |
| `translation_packages\` | Offline Argos models                                                 |


You can zip `**release\RecipeVoiceFilter`** and run on another Windows machine **without installing Python**, as long as the folder layout stays intact (EXE next to `data`, `models`, `translation_packages`).

---

## Order of operations (cheat sheet)

1. `python setup.py` — once per machine (or after wiping `models/`, `data/`, `translation_packages/`).
2. `python smoke_test.py` — optional quick validation.
3. `python app.py` — daily development.
4. `build.bat` — when you need a portable EXE; expect a long first run (downloads + PyInstaller).

---

## Repo note

Large assets (`data/`, `models/`, `translation_packages/`, build output) are **gitignored**; only code and config live in the repo. Run `setup.py` locally to materialize them.