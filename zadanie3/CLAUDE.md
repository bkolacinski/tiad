# Zadanie 3 — Klasyfikacja obrazów (TensorFlow / Keras)

## Cel zadania (na ocenę 5)

1. Ocenić skuteczność **co najmniej 3 różnych modeli** (np. ResNet, Inception, VGG, MobileNet, EfficientNet).
2. Analiza skuteczności dla **różnych podziałów %** zbioru uczącego/testowego.
3. Miary jakości: macierz pomyłek, accuracy, precision, recall, F1.
4. Krzywa **ROC/AUC** (z `sklearn`).

Decyzja: bierzemy **więcej niż 3 modele** (4–5), żeby porównanie było bogatsze.

## Dataset

**Cards Image Dataset-Classification** (Kaggle, autor: gpiosenka)

- Link: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
- 53 klasy (pełna talia + joker)
- ~8000 obrazów, 224×224 RGB
- Rozmiar: ~150 MB
- Posiada gotowy split train/valid/test, ale my robimy własne podziały % do analizy

Pobranie:
```bash
kaggle datasets download -d gpiosenka/cards-image-datasetclassification
```

Wymaga `~/.kaggle/kaggle.json` z tokenem.

## Stack technologiczny

- **Środowisko**: WSL2 Ubuntu (TF ≥ 2.11 nie ma już GPU na natywnym Windowsie)
- **GPU**: RTX 5080 (Blackwell). Wymaga **TF ≥ 2.18** + CUDA 12.x + cuDNN 9.x
- **Notebook**: marimo (zamiennik Jupyter) — `pip install marimo`
- **Biblioteki**: `tensorflow`, `keras`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`, `pandas`, `pillow`
- **Pobranie danych**: `kaggle` CLI

Sprawdzenie GPU na starcie:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Plan implementacji

### Modele (transfer learning, weights="imagenet", freeze backbone, własna głowa)

1. **MobileNetV2** — szybki, lekki baseline
2. **ResNet50**
3. **EfficientNetB0**
4. **InceptionV3**
5. **VGG16** (opcjonalnie — wolny, ale klasyk)

Każdy z własnym `preprocess_input` z `keras.applications.<model>`.

Architektura głowy:
```
backbone(trainable=False) → GlobalAveragePooling2D → Dropout → Dense(53, softmax)
```

### Podziały train/test do analizy

- 50/50
- 60/40
- 70/30
- 80/20

(test split = walidacja końcowa; w trakcie treningu można wydzielić mały val z train do early stopping)

### Trenowanie

- Image size: 224×224
- Batch size: 32 (dostosować do VRAM 5080 — można 64)
- Epochs: 10–15 (z `EarlyStopping` po `val_loss`)
- Optimizer: Adam, lr=1e-3 dla głowy
- Augmentacja: `RandomFlip`, `RandomRotation`, `RandomZoom` (lekko)
- Pętla: `for model in models: for split in splits: train + evaluate`

### Ewaluacja

Per (model, split):
- `confusion_matrix` + heatmap (seaborn)
- `classification_report` (precision/recall/F1 per klasa + macro/weighted)
- ROC/AUC: one-vs-rest dla 53 klas, AUC macro + micro
- Krzywa uczenia (loss/accuracy)

Porównanie zbiorcze:
- Tabela: model × split → accuracy / macro F1 / macro AUC
- Wykresy słupkowe / liniowe

## Konwencje kodu (BARDZO WAŻNE)

- **Nie pisać komentarzy w kodzie.** Bez `# this does X`, bez docstringów wyjaśniających oczywistości.
  - Wyjątek: tylko jeśli WHY jest nieoczywiste (subtelny constraint, workaround). Jeśli usunięcie komentarza nie zmyli czytelnika — nie piszemy go.
- Czyste, krótkie nazwy. Bez over-engineering, bez przedwczesnych abstrakcji.
- marimo cells robić tematyczne (load data / build model / train / evaluate).

## Struktura plików (proponowana)

```
zadanie3/
├── CLAUDE.md
├── Zadanie 3.pdf
├── notebook.py            # główny marimo notebook
├── data/                  # dataset (gitignore)
├── models/                # zapisane wagi (gitignore)
├── results/               # CSV z metrykami, wykresy
└── requirements.txt
```

`.gitignore` dla `data/` i `models/` (duże).

## Status

- [x] Branch `zadanie3` utworzony
- [x] Wybrany dataset: Cards (53 klasy)
- [ ] Setup WSL2 + TF GPU + sprawdzenie że karta jest widoczna
- [ ] Pobranie datasetu
- [ ] Notebook marimo z pipeline
- [ ] Trening 4–5 modeli × 4 podziały
- [ ] Metryki + wykresy + ROC
- [ ] README / podsumowanie

## Workflow

Praca w WSL2. Po przejściu na WSL: kontynuujemy konwersację w zdalnym Claude Code w tym samym katalogu (`/mnt/c/Users/nikod/Documents/uni/tiad` lub sklonowane do `~`).
