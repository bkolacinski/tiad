# Zadanie 3 — Klasyfikacja obrazów kart

Projekt na przedmiot **Techniki Inteligentnej Analizy Danych** (semestr letni 2025/2026, Politechnika Łódzka).

**Autorzy:** Nikodem Nowak (251598), Bartosz Kołaciński (251554)
**Data oddania:** 2026-05-06

## Cel

Porównanie pięciu architektur konwolucyjnych sieci neuronowych w klasyfikacji obrazów kart do gry. Dla każdego modelu zbadano wpływ proporcji podziału zbioru train/test (50/50, 60/40, 70/30, 80/20, 90/10) — łącznie 25 niezależnych eksperymentów.

## Stack technologiczny

| Narzędzie | Wersja | Rola |
|---|---|---|
| Python | 3.12 | Środowisko |
| TensorFlow + Keras | 2.21 | Definicja i trening sieci |
| keras.applications | (TF) | Modele wraz z wagami z ImageNet |
| scikit-learn | 1.8 | Macierz pomyłek, P/R/F1, ROC/AUC |
| matplotlib + seaborn | — | Wykresy (krzywe uczenia, heatmapy CM, ROC) |
| pandas + numpy | — | Tabele wyników, operacje numeryczne |
| marimo | — | Notebook reaktywny (plik `.py` zamiast Jupytera) |
| kaggle CLI | 2.1 | Pobranie datasetu z Kaggle |

GPU: **NVIDIA GeForce RTX 5080** (Blackwell sm_120, 16 GB VRAM), CUDA 12.x, cuDNN 9.x.
Środowisko: WSL2 Ubuntu 22.04.

## Modele

Wszystkie ładowane z wagami z ImageNet, z głową dostosowaną do zadania (GlobalAveragePooling2D → Dropout(0.5) → Dense(14, softmax)):

| Model | Rok | Parametry | Główna idea |
|---|---|---|---|
| MobileNetV2 | 2018 | 3.5M | Depthwise separable conv, inverted residuals |
| ResNet50 | 2015 | 25M | Residual connections (skip connections) |
| EfficientNetB0 | 2019 | 5M | Compound scaling |
| InceptionV3 | 2015 | 24M | Moduły Inception (multi-scale conv) |
| VGG16 | 2014 | 138M (z czego 15M w backbone) | Klasyczne bloki Conv-Conv-Pool |

## Dataset

[Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) (Kaggle, autor: gpiosenka).
- 8154 obrazów 224×224 RGB
- Pierwotnie 53 klasy (52 karty + joker), w pracy zredukowane do **14 typów** (ace, 2-10, jack, queen, king, joker) dla lepszej liczebności klas (~580 obrazów/klasa zamiast ~150)

Dataset jest pobierany przez `kaggle datasets download` i nie jest commitowany do repo (`.gitignore`).

## Metoda treningu

**Trening dwufazowy:**
1. **Faza 1** — backbone zamrożony, trenowana tylko głowa. Adam, lr=1e-3, do 8 epok
2. **Faza 2** — odmrożenie 1/3 najwyższych warstw backbone'u (BN pozostawione w trybie inference). Adam, lr=1e-5 (100× mniejsze), do 12 epok

**EarlyStopping** monitorujący `val_loss` z `patience=5` i `restore_best_weights=True`.

**Augmentacja** (tylko na zbiorze uczącym): rotacja ±10.8°, zoom ±8%, translacja ±5%, jasność ±15%, kontrast ±15%. Świadomie pominięto horizontal flip (karty są zorientowane).

## Wyniki (skrót)

Najlepszy wynik: **VGG16 przy split 0.90 → accuracy 0.852, AUC macro 0.991, F1 0.858**.

| Model | acc 0.5 | acc 0.6 | acc 0.7 | acc 0.8 | acc 0.9 |
|---|---|---|---|---|---|
| MobileNetV2 | 0.69 | 0.73 | 0.74 | 0.76 | 0.76 |
| ResNet50 | 0.78 | 0.81 | 0.80 | 0.85 | 0.84 |
| EfficientNetB0 | 0.72 | 0.75 | 0.76 | 0.78 | 0.77 |
| InceptionV3 | 0.66 | 0.68 | 0.70 | 0.72 | 0.75 |
| **VGG16** | 0.80 | 0.81 | 0.82 | 0.84 | **0.85** |

Pełne wyniki: [`results/metrics.csv`](results/metrics.csv).

## Struktura projektu

```
zadanie3/
├── README.md                  # ten plik
├── OPIS_PROJEKTU.md           # szczegółowy opis pipeline'u i decyzji projektowych (teoria)
├── CLAUDE.md                  # plan początkowy
├── Zadanie 3.pdf              # treść zadania
├── notebook.py                # główny pipeline (marimo)
├── requirements.txt           # zależności Pythona
├── .gitignore
├── data/                      # dataset (.gitignore — pobrać przez kaggle CLI)
├── models/                    # zapisane wagi (.gitignore)
├── results/                   # wyniki bieżącej (14-klasowej) wersji
│   ├── metrics.csv
│   ├── metrics_vs_split.png
│   ├── learning_curves.png
│   ├── cm_<best>.png
│   ├── roc_<best>.png
│   ├── cm/<model>_<split>.npy
│   ├── probs/<model>_<split>_y_*.npy
│   └── history/<model>_<split>.json
├── results_v3_53class/        # backup — wyniki przy 53 klasach (do porównania)
├── results_v1_baseline/       # backup — wyniki bez fine-tuningu (do porównania)
└── sprawozdanie/
    ├── sprawozdanie.tex       # właściwe sprawozdanie LaTeX
    ├── bibliografia.bib       # bibliografia BibTeX
    └── images/                # wykresy z results/ skopiowane do sprawozdania
```

## Jak odtworzyć wyniki

### 1. Środowisko
```bash
# w katalogu zadanie3/
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Token Kaggle
Wygenerować w Kaggle (Settings → API → Create New Token) i umieścić:
```bash
mkdir -p ~/.kaggle
# Legacy API Key: ~/.kaggle/kaggle.json (chmod 600)
# nowy API Token: ~/.kaggle/access_token (chmod 600)
```

### 3. Pobranie datasetu
```bash
kaggle datasets download -d gpiosenka/cards-image-datasetclassification -p data/ --unzip
```

### 4. Trening (5 modeli × 5 splitów = 25 runów, ~70-90 min na RTX 5080)
```bash
python notebook.py
```

Pętla zapisuje wyniki do `results/metrics.csv` po każdym runie i ma wsparcie wznowienia (kolejne uruchomienie pomija ukończone wpisy z CSV).

### 5. Notebook interaktywny
```bash
marimo edit notebook.py
```

### 6. Sprawozdanie PDF
```bash
cd sprawozdanie
pdflatex sprawozdanie.tex
bibtex sprawozdanie
pdflatex sprawozdanie.tex
pdflatex sprawozdanie.tex
```

## Iteracje projektu

Projekt przeszedł trzy iteracje przed osiągnięciem finalnej wersji:

| Wersja | Konfiguracja | Wynik | Backup w |
|---|---|---|---|
| v1 | 53 klasy, frozen backbone, RandomFlip horizontal | 38-52% accuracy | `results_v1_baseline/` |
| v2 | 53 klasy, fine-tuning faza 2, bez flipa | 60-72% accuracy | `results_v3_53class/` |
| v3 | 53 klasy, dropout 0.5, color augmentation | (pominięty — przejście na 14 klas) | — |
| v3-final | **14 klas**, dropout 0.5, color aug, dłuższe epoki | **66-85% accuracy** | `results/` |

Szczegóły kolejnych decyzji w [`OPIS_PROJEKTU.md`](OPIS_PROJEKTU.md), sekcja "Co poszło nie tak — 3 iteracje".

## Bibliografia (najważniejsze)

- Treść zadania: WIKAMP, kurs Techniki Inteligentnej Analizy Danych
- Dataset: G. Piosenka, *Cards Image Dataset-Classification*, Kaggle, 2022
- Architektury: VGG (Simonyan & Zisserman 2015), ResNet (He et al. 2016), InceptionV3 (Szegedy et al. 2016), MobileNetV2 (Sandler et al. 2018), EfficientNet (Tan & Le 2019)
- Transfer learning: Yosinski et al. 2014
- Pełna bibliografia w [`sprawozdanie/bibliografia.bib`](sprawozdanie/bibliografia.bib).
