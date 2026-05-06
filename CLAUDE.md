# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

Three independent university assignments:

- `zadanie1/` — XLSX-to-DOCX/PDF converter (tkinter GUI)
- `zadanie2/` — Recipe Voice Filter (CustomTkinter GUI)
- `zadanie3/` — CNN image classification (marimo notebook, TensorFlow/Keras)

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

## zadanie3 — CNN Image Classification

### Development Commands

```bash
# Create venv and install deps
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Download dataset (requires ~/.kaggle/kaggle.json or access_token)
kaggle datasets download -d gpiosenka/cards-image-datasetclassification -p data/ --unzip

# Run full training pipeline (5 models × 5 splits = 25 experiments, ~70-90 min on RTX 5080)
python notebook.py

# Interactive notebook
marimo edit notebook.py
```

### Architecture

Pipeline: dataset CSV → stratified train/test split → TF Dataset (decode + augment + preprocess) → two-phase transfer learning → metrics + plots saved to `results/`.

| Component | Details |
|---|---|
| `notebook.py` | Single marimo `.py` file — all cells from data loading to final plots |
| Models | MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, VGG16 (weights=imagenet) |
| Head | GlobalAveragePooling2D → Dropout(0.5) → Dense(NUM_CLASSES, softmax) |
| Phase 1 | Frozen backbone, Adam lr=1e-3, up to 8 epochs |
| Phase 2 | Top 1/3 backbone unfrozen (BN kept frozen), Adam lr=1e-5, up to 12 epochs |
| Splits | 50/50, 60/40, 70/30, 80/20, 90/10 |

### Key Design Constraints

- **Dataset**: Cards Image Dataset (Kaggle, gpiosenka) — 8154 images, 224×224 RGB, **14 card types** (ace, 2–10, jack, queen, king, joker; suit-independent grouping), ~580 images/class.
- **Fully offline after setup**: dataset in `data/`, saved weights in `models/` — both gitignored.
- **Resume support**: training loop checks `results/metrics.csv` and skips already-completed (model, split) pairs.
- **GPU**: RTX 5080 (Blackwell sm_120, 16 GB VRAM), CUDA 12.x, cuDNN 9.x, WSL2 Ubuntu 22.04.
- **VRAM management**: `set_memory_growth(True)` + `tf.keras.backend.clear_session()` + `gc.collect()` after each experiment.
- **BN frozen during fine-tuning**: BatchNorm layers stay in inference mode (ImageNet running stats) to avoid corruption from small batches.
- **Augmentation order**: augmentation runs before `preprocess_input` (which rescales/shifts pixels).
- **No horizontal flip**: cards are orientation-dependent.
