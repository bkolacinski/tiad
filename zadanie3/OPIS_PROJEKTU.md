# Zadanie 3 — Klasyfikacja obrazów kart (opis i teoria)

Dokument wyjaśniający co dokładnie robimy, jakie narzędzia, jak działa pipeline,
co dostajemy na wyjściu i jak czytać metryki. Pisane pod oddanie zadania.

---

## 1. Cel zadania (krótko)

Mamy zbiór **8154 obrazów kart do gry** (53 klasy: 52 karty + joker, każda 224×224 RGB).
Zadanie: **trenujemy sieci neuronowe które po obejrzeniu obrazu mówią "to jest król kier"**.

Na ocenę 5 musimy:
1. Porównać **min. 3 modele** (my robimy 5: MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, VGG16)
2. Sprawdzić jak wpływa **podział train/test** (4 podziały: 50/50, 60/40, 70/30, 80/20)
3. Pokazać metryki: **accuracy, precision, recall, F1, confusion matrix**
4. Pokazać **krzywą ROC i AUC** (z scikit-learn)

To daje **5 modeli × 4 splity = 20 niezależnych eksperymentów** + tabele zbiorcze + wykresy.

---

## 2. Stack technologiczny

| Narzędzie | Co robi | Dlaczego |
|---|---|---|
| **TensorFlow 2.21 + Keras** | Definiowanie i trenowanie sieci | Standard branżowy. `keras.applications` ma gotowe modele z wagami z ImageNet |
| **scikit-learn** | Metryki (`confusion_matrix`, `classification_report`, `roc_curve`, `roc_auc_score`) | TF ma niektóre, ale sklearn ma kompletny zestaw i jest standardem do raportowania |
| **matplotlib + seaborn** | Wykresy (krzywe uczenia, heatmapy CM, ROC, słupkowe) | Standard naukowy. Seaborn = ładniejsze heatmapy |
| **pandas + numpy** | Wczytywanie CSV, tabele wyników, operacje numeryczne | Standardowe biblioteki |
| **kaggle CLI** | Pobranie datasetu z Kaggle | Tak autor zadania kazał |
| **marimo** | Notebook reaktywny (zamiennik Jupyter) | Plik to czysty `.py`, nie JSON; cells wykonują się w kolejności zależności |
| **CUDA 12.x + cuDNN 9.x** | Trening na GPU (RTX 5080) | Bez tego wszystko by trwało ~50× dłużej na CPU |

**Twoje pytanie "korzystamy z tensorflow do tego i już?"** — głównie tak. TF/Keras robi 90% roboty:
ładuje modele, trenuje, przewiduje. Sklearn dorzuca metryki bo TF nie ma wszystkich (zwłaszcza ROC/AUC).
Matplotlib rysuje. Reszta to glue code.

---

## 3. Cały pipeline — co się dzieje od A do Z

```
┌──────────────────────────────────────────────────────────────────────┐
│  Dataset z Kaggle (Cards Image Dataset, 8154 obrazów, 53 klasy)      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                           pd.read_csv("data/cards.csv")
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DataFrame: ścieżki + etykiety + indeks klasy (0..52)                │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                  for split in [0.50, 0.60, 0.70, 0.80]:
                       train_test_split(stratify=labels, train_size=split)
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Lista obrazów train + lista obrazów test (zachowany balans klas)    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                       tf.data.Dataset z paths, labels
                       map: read → decode_jpeg → resize 224x224
                       train: shuffle + augment (rotate/zoom/translate/brightness/contrast)
                       map: preprocess_input modelu (normalizacja zgodna z ImageNet treningiem)
                       batch(32) + prefetch
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Strumień batchy gotowych do trenowania (GPU może je czytać)         │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                  for model_name in [MobileNetV2, ResNet50, ...]:
                       backbone = keras.applications.X(weights="imagenet", include_top=False)
                       backbone.trainable = False                  # FAZA 1: zamrożony
                       head = Dropout(0.5) → Dense(53, softmax)
                       
                       PHASE 1: model.fit(epochs=6, lr=1e-3)        # uczy tylko głowę
                       PHASE 2: backbone.trainable = True (top 1/3) # odmrażamy górę backbone'u
                                model.fit(epochs=10, lr=1e-5)        # delikatne dostrojenie
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Wytrenowany model dla pary (architektura, split)                    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                        y_prob = model.predict(test_ds)
                        y_pred = argmax(y_prob, axis=1)
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Metryki przez sklearn:                                              │
│    confusion_matrix(y_true, y_pred)                                  │
│    classification_report → accuracy, precision, recall, F1           │
│    roc_auc_score(y_true_onehot, y_prob, multi_class="ovr")           │
│  Zapis do results/metrics.csv (1 wiersz = 1 eksperyment)             │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Wykresy (pełna pętla skończona):                                    │
│    metrics_vs_split.png  — accuracy/F1/AUC × model × split           │
│    cm_<best>.png          — heatmapa confusion matrix najlepszego    │
│    roc_<best>.png         — krzywe ROC dla 5 najlepszych + 5 najgorszych klas │
│    learning_curves.png    — loss/accuracy w czasie treningu          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dataset szczegółowo

**Źródło:** [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

Pobrane przez `kaggle datasets download -d gpiosenka/cards-image-datasetclassification`.

Struktura:
```
data/
├── cards.csv          # główne źródło prawdy: filepath, label, class_index, data set
├── train/             # ~7624 obrazów, ~143 per klasa
│   ├── ace of clubs/
│   ├── ace of diamonds/
│   └── ...            # 53 podfoldery
├── valid/             # 265 obrazów (5 per klasa)
└── test/              # 265 obrazów (5 per klasa)
```

**Co robimy z tym splitem?** Autor dał gotowy podział, ale wymóg zadania mówi "zbadać wpływ
podziału %". Więc **łączymy train+valid+test w jeden zbiór** i robimy własne **stratified
split** (zachowujący balans klas) dla każdego z 4 procentów. Stąd `make_split(0.50)` daje
4077 train + 4077 test, `make_split(0.80)` daje 6523 + 1631, itd.

**Dlaczego `stratify=labels`?** Bez tego losowy split mógłby dać "0 obrazów klasy X w treningu"
co by wywaliło trening. `stratify` gwarantuje że każda klasa ma proporcjonalny udział w obu
zbiorach.

---

## 5. Transfer learning — kluczowy koncept

To jest **najważniejsza rzecz do zrozumienia**.

### Naiwne podejście

Mamy 8000 obrazów na 53 klas. Gdybyśmy uczyli sieć **od zera** (random weights), potrzebowalibyśmy
miliony obrazów (jak w ImageNet) i tygodni treningu. Z 8000 obrazów dostalibyśmy ~5-15% accuracy.
Sieci konwolucyjne mają miliony parametrów które wymagają **dużo danych** żeby się dobrze ustawić.

### Co to jest transfer learning

Modele jak ResNet50 zostały już wcześniej wytrenowane na **ImageNet** (1.2M obrazów, 1000 klas
zwierząt/obiektów codziennych). Ten trening kosztuje $10k-100k na klastrze GPU. Wyniki tego
treningu — **wagi sieci** — są publicznie udostępnione.

`keras.applications.ResNet50(weights="imagenet")` to ResNet50 z **gotowymi wagami z ImageNet**.

**Co takie modele "umieją"?** Konwolucyjne warstwy nauczyły się rozpoznawać:
- Pierwsze warstwy: krawędzie, gradienty kolorów, plamy
- Środkowe: tekstury, kształty proste (kółka, linie, narożniki)
- Głębsze: części obiektów (oko, ucho, koło, prostokąt)
- Najgłębsze: całe obiekty

Te cechy są **uniwersalne** — pies/kot/karta/samochód wszystko korzysta z tych samych "cegiełek".

### Jak to wykorzystujemy

Robimy tak (`build_model` w notebook.py):

```python
backbone = ResNet50(weights="imagenet", include_top=False, pooling="avg")
backbone.trainable = False              # ZAMRAŻAMY backbone

inputs = Input(shape=(224, 224, 3))
x = backbone(inputs, training=False)    # backbone wypluwa wektor 2048 cech
x = Dropout(0.5)(x)
outputs = Dense(53, activation="softmax")(x)   # nasza głowa: 53 klasy

model = Model(inputs, outputs)
```

`include_top=False` mówi: "nie chcemy oryginalnej głowy 1000-klasowej z ImageNetu".
`pooling="avg"` dodaje GlobalAveragePooling2D — uśrednia mapy cech do jednego wektora.

`backbone.trainable = False` oznacza: **gradienty nie aktualizują wag backbone'u**. Trenujemy
tylko Dropout+Dense (~110 tys. parametrów zamiast 25M). To jest:

- **Szybkie** — kilkadziesiąt sekund na epokę zamiast godzin
- **Stabilne** — z 8000 obrazów nie zepsujemy gotowych dobrych ekstraktorów cech
- **Wystarczająco dobre** dla większości problemów

Backbone działa jako **ekstraktor cech**: zamienia obraz `[224,224,3]` (150528 liczb)
na wektor `[2048]` (dla ResNet50) reprezentujący "co jest na obrazie". Nasza głowa
uczy się tylko mapowania `[2048] → 53 klasy`.

### Trening dwufazowy (co dodaliśmy w v3)

Sam zamrożony backbone okazał się za słaby (45-50% accuracy). Dodaliśmy **fine-tuning**:

```python
# PHASE 1: trenuj tylko głowę
backbone.trainable = False
model.compile(optimizer=Adam(1e-3))
model.fit(epochs=6)

# PHASE 2: odmrażaj górne 1/3 backbone'u
backbone.trainable = True
for layer in backbone.layers[:-n_layers//3]:
    layer.trainable = False                  # niższe warstwy nadal zamrożone
for layer in backbone.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False              # BN zostają w trybie inference
model.compile(optimizer=Adam(1e-5))           # MUCH mniejszy lr — bo nie chcemy zepsuć
model.fit(epochs=10)
```

**Dlaczego top 1/3, a nie cały backbone?**
- Niższe warstwy (krawędzie, kolory) są uniwersalne — nie ma sensu ich zmieniać
- Górne warstwy (parts, objects) są specyficzne do ImageNetu — warto je dostroić do kart
- Odmrażanie wszystkiego dałoby za dużo trenowanych parametrów (overfitting)

**Dlaczego lr=1e-5 w fazie 2?** W fazie 1 mamy `lr=1e-3` bo trenujemy tylko głowę (świeże losowe
wagi). W fazie 2 modyfikujemy *wagi które już są dobre* — duży lr by je zniszczył. Reguła kciuka:
fine-tuning ma 100× mniejszy lr niż faza head.

**Dlaczego BatchNormalization zostają zamrożone?** BN przechowuje statystyki populacji (średnia,
wariancja). W fine-tuningu chcemy używać statystyk z ImageNetu (na których backbone był trenowany),
a nie aktualizować je małym batchem 32 obrazów kart.

---

## 6. Pięć modeli — czym się różnią

Wszystkie robią to samo (obraz → wektor cech → klasyfikacja), ale różnie pod maską.

| Model | Rok | Idea | Params | Ile cech wypluwa |
|---|---|---|---|---|
| **MobileNetV2** | 2018 | Depthwise separable conv (rozkłada zwykłą konwolucję na dwie tańsze) + inverted residuals | 3.5M | 1280 |
| **ResNet50** | 2015 | Residual connections — `f(x) + x`. Pozwala iść głęboko (50/101/152 warstwy) bez problemu zanikania gradientu | 25M | 2048 |
| **EfficientNetB0** | 2019 | Compound scaling — matematycznie wyznaczone optymalne proporcje głębokości/szerokości/rozdzielczości | 5M | 1280 |
| **InceptionV3** | 2015 | "Moduły Inception" — równolegle conv 1×1, 3×3, 5×5 + pool, sklejone razem. Patrzy na wiele skal naraz | 24M | 2048 |
| **VGG16** | 2014 | Po prostu wiele bloków `Conv-Conv-Pool`. Bardzo proste. 138M params (większość to tail Dense, my go odcinamy) | ~15M (bez głowy) | 512 |

**Każdy model ma swój `preprocess_input`!** To kluczowe:

```python
from keras.applications import resnet50, mobilenet_v2, efficientnet, inception_v3, vgg16

resnet50.preprocess_input(x)         # odejmuje średnią ImageNetu z każdego kanału, BGR
mobilenet_v2.preprocess_input(x)     # skaluje do [-1, 1]
efficientnet.preprocess_input(x)     # skaluje do [0, 1]
inception_v3.preprocess_input(x)     # skaluje do [-1, 1]
vgg16.preprocess_input(x)            # odejmuje średnią ImageNetu, BGR
```

Jeśli użyjesz złego preprocess_input, model dostaje obrazy w "obcym formacie" i daje śmieci.
W naszym `build_model` dla każdej architektury bierzemy *jej* preprocess_input.

---

## 7. Augmentacja danych

Augmentacja = **w trakcie treningu losowo modyfikujemy obraz** zanim trafi do sieci.
Cel: sieć widzi "nowe" obrazy w każdej epoce → mniej overfittingu, lepsza generalizacja.

W naszym `build_dataset`:

```python
augment = Sequential([
    RandomRotation(0.03),                            # ±10.8 stopnia (0.03 * 360)
    RandomZoom(0.08),                                # ±8% zoom
    RandomTranslation(0.05, 0.05),                   # ±5% przesunięcie
    RandomBrightness(0.15, value_range=(0, 255)),    # ±15% jasność
    RandomContrast(0.15),                            # ±15% kontrast
])
```

**Augmentacja działa tylko na train**, nigdy na teście. W naszym kodzie:
```python
if training:
    ds = ds.map(lambda x, y: (augment(x, training=True), y))
```

### Czego NIE używamy i dlaczego

W pierwszej wersji mieliśmy `RandomFlip("horizontal")`. **Usunęliśmy** bo karta jest
**obrazem orientowanym** — King obrócony do góry nogami nie wygląda jak King
zwykły. Horizontal flip uczyłby sieć błędnych wzorców.

Dlaczego nie `RandomFlip("vertical")`? Tym bardziej karty mają orientację pionową.

Dlaczego nie większe rotacje (np. 0.25 = 90 stopni)? Karta obrócona o 90° to dla człowieka
inna karta (na bok), więc dla sieci by to było mylące.

### Co jeszcze pomogło (color augmentation)

`RandomBrightness` i `RandomContrast` dodaliśmy w v3. Dataset zawiera obrazy z różnych źródeł
(prawdziwe zdjęcia + obrazy AI-generated) o różnym styliście. Augmentacja kolorystyczna pomaga
sieci być odporną na te różnice.

---

## 8. Metryki — jak czytać wyniki

### 8.1 Confusion matrix (macierz pomyłek)

Tablica `[N_klas × N_klas]` gdzie `CM[i][j]` = ile razy obraz klasy `i` zostało zaklasyfikowane
jako klasa `j`. **Diagonala** = poprawne klasyfikacje.

Przykład dla 3-klasowej sieci (kot/pies/koń):

```
              przewidziane:
              kot   pies  koń
prawdziwe:
kot           45    3     2     ← było 50 kotów, sieć poprawnie 45
pies          5     38    7     ← było 50 psów, sieć poprawnie 38, 7 razy myli z koniem
koń           1     2     47    ← było 50 koni, sieć poprawnie 47
```

W naszym przypadku CM jest 53×53, więc rysujemy heatmapę (`seaborn.heatmap`). Im jaśniejsza
przekątna, tym lepiej.

Generujemy w komórce evaluation:
```python
cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
```

### 8.2 Accuracy, Precision, Recall, F1

Dla każdej klasy `c` definiujemy:
- **TP** (true positive): obraz klasy `c` rozpoznany jako `c`
- **FP** (false positive): obraz innej klasy rozpoznany jako `c`
- **FN** (false negative): obraz klasy `c` rozpoznany jako inna
- **TN** (true negative): obraz innej klasy rozpoznany jako inna

| Metryka | Wzór | Co mówi |
|---|---|---|
| **Accuracy** | `(TP + TN) / total` | Jaki % obrazów został trafnie zaklasyfikowany. **Dla 53 klas accuracy = po prostu dobrze sklasyfikowane / wszystkie** |
| **Precision** | `TP / (TP + FP)` | Spośród przewidzianych jako `c`, ile faktycznie było `c`? "Jak czysta jest klasa `c` w wynikach" |
| **Recall** | `TP / (TP + FN)` | Spośród prawdziwych `c`, ile sieć znalazła? "Jak dobrze sieć łapie wszystkie `c`" |
| **F1** | `2 * P * R / (P + R)` | Harmoniczna średnia P i R. Karze niezbalansowanie. |

**Macro vs micro:**
- **Macro F1** = średnia F1 po wszystkich klasach (każda klasa ma równą wagę)
- **Micro F1** = liczone globalnie (klasy z więcej obrazów mają więcej wagi)
- W zbalansowanym datasecie (jak nasz, gdzie każda klasa ma ~143 obrazy) **macro ≈ micro**

W naszym kodzie:
```python
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
report["accuracy"]                  # ogólny accuracy
report["macro avg"]["precision"]    # macro precision
report["macro avg"]["recall"]       # macro recall
report["macro avg"]["f1-score"]     # macro F1
```

### 8.3 ROC i AUC

ROC (Receiver Operating Characteristic) i AUC (Area Under Curve) to metryki **niezależne od progu
decyzyjnego**.

**Problem:** sieć daje prawdopodobieństwo. Domyślnie wybieramy klasę z najwyższym prawd. (top-1).
Ale gdybyśmy chcieli być bardziej "ostrożni" — np. przewidywać klasę X tylko gdy prawd. > 0.7 —
to `accuracy` by się zmieniła. **Accuracy zależy od arbitralnego progu.**

ROC patrzy na to inaczej. Dla każdej klasy `c` (one-vs-rest):
1. Sortuj obrazy po `y_prob[c]` (od najwyższego do najniższego)
2. Idź od najwyższego progu (prawd. ≥ 1.0) do najniższego (prawd. ≥ 0.0)
3. Dla każdego progu policz **TPR** (true positive rate = recall) i **FPR** (false positive rate)
4. Narysuj punkty (FPR, TPR) — to **krzywa ROC**

**AUC = Area Under Curve.** Liczba między 0 i 1:
- `AUC = 0.5` → klasyfikator losowy
- `AUC = 1.0` → idealny (rozróżnia perfekcyjnie)
- `AUC = 0.95` → bardzo dobry
- `AUC > 0.99` → potencjalnie data leakage albo overfit

**Dla 53 klas:** robimy **one-vs-rest** (OvR) — dla każdej klasy traktujemy ją jako "pozytywną"
a 52 inne jako "negatywne", liczymy ROC, potem **uśredniamy AUC** (macro lub micro).

W kodzie:
```python
from sklearn.metrics import roc_auc_score, roc_curve
y_true_oh = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)   # one-hot encoding
auc_macro = roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr")
```

W wykresie ROC nie rysujemy 53 krzywych (nieczytelne), tylko **5 najgorszych + 5 najlepszych klas**.

### 8.4 Co mówi accuracy 0.70 vs AUC 0.98 (jak u nas)

Mieliśmy ResNet50 z `acc=0.70, auc=0.98`. Co to znaczy?
- W 70% przypadków sieć trafiła klasę top-1
- Ale gdy nie trafiła, prawdziwa klasa była zwykle w **top-2 lub top-3**
- Czyli sieć "wie" która klasa to jest, tylko jest niezdecydowana między 2-3 podobnymi
- Dla kart to logiczne — np. trójka pik vs trójka kier wyglądają bardzo podobnie

Wysoka AUC z umiarkowaną accuracy to typowa "rozmazana" klasyfikacja. Można by to poprawić
mocniejszym treningiem albo lepszą augmentacją. Dla zadania wystarczy zaraportować obie liczby.

---

## 9. Co dokładnie zapisujemy do `results/`

Po każdym z 20 runów dopisujemy wiersz do `metrics.csv`:

| Kolumna | Co to |
|---|---|
| `model` | nazwa architektury (np. "ResNet50") |
| `split` | proporcja train (np. 0.7) |
| `accuracy` | dokładność top-1 |
| `macro_f1` | F1 uśrednione po klasach |
| `macro_precision` | precision uśrednione po klasach |
| `macro_recall` | recall uśrednione po klasach |
| `auc_macro` | AUC OvR uśrednione po klasach (równa waga) |
| `auc_micro` | AUC OvR globalnie (waga proporcjonalna do liczności) |
| `epochs_run` | ile epok faktycznie się wykonało (early stopping mógł skrócić) |
| `train_time_s` | czas treningu w sekundach |

Dodatkowo w podfolderach:
- `results/cm/<model>_<split>.npy` — confusion matrix jako numpy array (53×53)
- `results/probs/<model>_<split>_y_true.npy` — etykiety prawdziwe
- `results/probs/<model>_<split>_y_prob.npy` — prawdopodobieństwa (N×53), do późniejszego rysowania ROC
- `results/history/<model>_<split>.json` — krzywe loss/accuracy w czasie treningu

I obrazy:
- `results/metrics_vs_split.png` — 3 wykresy obok siebie: accuracy/F1/AUC vs split, krzywa per model
- `results/cm_<best>.png` — heatmapa CM najlepszego modelu
- `results/roc_<best>.png` — krzywe ROC dla 5 najgorszych + 5 najlepszych klas
- `results/learning_curves.png` — loss i accuracy w czasie treningu, wszystkie modele/splity

**Zapis po każdym runie** (a nie po wszystkich) jest celowy — gdyby trening crashnął
w połowie, mamy zachowane co już zrobione (i pętla pominie te wpisy).

---

## 10. Co poszło nie tak — 3 iteracje (v1 → v2 → v3)

To nieoczywista część — pierwsze podejście nie dało dobrych wyników. Ważne do zrozumienia
co poprawialiśmy.

### v1: tylko frozen backbone, krótka augmentacja

- `backbone.trainable = False`, trenujemy tylko głowę
- Augmentacja: `RandomFlip(horizontal) + RandomRotation(0.05) + RandomZoom(0.1)`
- 12 epok, lr=1e-3

**Wynik:** wszystkie 5 modeli utknęło w przedziale **38-52% accuracy**. AUC było wysokie (0.92-0.96).

**Diagnoza:** problemy:
1. **RandomFlip horizontal** psuł etykiety (King ≠ lustro Kinga)
2. **Frozen backbone** — ImageNet features są dobre dla naturalnych obrazów, słabsze dla schematycznych kart
3. Wyniki w `results_v1_baseline/`

### v2: usunięto flip, dodano fine-tuning

- Usunięto horizontal flip, zmniejszono rotację
- Dodano dwufazowy trening: faza 1 (frozen) + faza 2 (odmrożone top 1/3, lr=1e-5)

**Smoke ResNet50 80/20:** acc 0.62 (z 0.48). Lepiej, ale wciąż overfit (train 0.91 vs val 0.67).

**Diagnoza:** model się uczy ale nie generalizuje. Dataset ma mieszankę real/AI-generated obrazów
o różnych stylach kolorystycznych.

### v3 (obecna): mocniejszy anti-overfit

- Dropout zwiększony z 0.2 → 0.5
- Dodana **color augmentation** (`RandomBrightness`, `RandomContrast`) — żeby sieć była odporna na różnice stylu
- Faza 2 wydłużona z 6 do 10 epok, patience zwiększony z 3 do 4

**Smoke ResNet50 80/20:** acc 0.72. Pełne wyniki w trakcie generowania.

---

## 11. Struktura pliku `notebook.py`

Marimo dzieli notebook na **komórki** (funkcje z dekoratorem `@app.cell`). Komórki wykonują się
w kolejności **zależności zmiennych**, nie w kolejności pisania.

```
Cell 1   — imports marimo
Cell 2   — markdown: nagłówek
Cell 3   — TF setup, GPU detection, memory growth
Cell 4   — konfiguracja: SPLITS, MODEL_NAMES, hyperparams
Cell 5   — tworzenie katalogów results/, models/
Cell 6   — wczytanie cards.csv → DataFrame
Cell 7   — podgląd DataFrame (marimo UI table)
Cell 8   — funkcja make_split(train_frac) — stratified split
Cell 9   — funkcja build_dataset() — tf.data pipeline z augmentacją
Cell 10  — MODEL_REGISTRY + funkcja build_model(name) — backbone + head
Cell 11  — funkcja train_one() — dwufazowy trening
Cell 12  — funkcja evaluate(model, test_ds) — wszystkie metryki + zapis surowych danych
Cell 13  — GŁÓWNA PĘTLA: for model in MODEL_NAMES: for split in SPLITS:
              z resume support (jeśli wpis już w CSV, skip), z cleanup VRAM
Cell 14  — markdown: nagłówek tabeli
Cell 15  — tabela summary z metrics_df
Cell 16  — pivot tables: accuracy/F1/AUC × model × split
Cell 17  — print pivot
Cell 18  — wykres metrics_vs_split.png
Cell 19  — heatmapa CM najlepszego + sns.heatmap
Cell 20  — ROC dla 5 najgorszych + 5 najlepszych klas najlepszego modelu
Cell 21  — learning_curves.png — loss/accuracy w czasie treningu wszystkich runów
```

---

## 12. Co napisać w sprawozdaniu

Sugerowana struktura:

1. **Wprowadzenie** — co robimy, jakie modele, jakie metryki, jaki dataset
2. **Środowisko** — TF 2.21 GPU, RTX 5080, marimo
3. **Dataset** — 8154 obrazów, 53 klasy, własne stratified splity
4. **Architektury** — krótki opis każdego z 5 modeli (Tabela z sekcji 6 wystarczy)
5. **Pipeline treningowy** — transfer learning, dwufazowy trening (uzasadnij dlaczego)
6. **Wyniki**:
   - Tabela: model × split → accuracy/F1/AUC (z `metrics.csv`)
   - Wykres `metrics_vs_split.png` — wpływ proporcji
   - Confusion matrix najlepszego modelu (`cm_*.png`)
   - Krzywa ROC najlepszego (`roc_*.png`)
   - Krzywe uczenia (`learning_curves.png`)
7. **Wnioski** — który model najlepszy, jak split wpływa, gdzie sieci się mylą
   (analiza CM — które karty są mylone najczęściej? Pewnie wartości tej samej figury w różnych kolorach)

### Pomocne fakty do wniosków

- **Wszystkie modele lepsze przy większym train split** — to oczekiwane (więcej danych = lepszy model)
- **AUC zwykle wysokie (0.95+)** — modele wiedzą która karta to jest, top-1 nie zawsze trafia idealnie
- **EfficientNetB0 zwykle wygrywał w v1** — najnowsza architektura
- **MobileNetV2 nadspodziewanie dobry** mimo małych rozmiarów (3.5M params)
- **VGG16/InceptionV3** wolniejsze, niekoniecznie lepsze — pokazuje że "nowsze ≠ tylko większe"
- **Confusion matrix:** największe mylenie zwykle między **kartami tej samej wartości innego koloru**
  (np. trójka pik ↔ trójka treflowa) — bo w niskich/średnich warstwach sieć patrzy na kształt liczby/figury,
  a kolor kart jest często mały detail

---

## 13. Jak uruchomić to wszystko ręcznie (gdyby coś zaczęło padać)

```bash
cd /home/nikodem/tiad/zadanie3

# Aktywacja venv (LD_LIBRARY_PATH dla CUDA libów ustawia się automatycznie)
source .venv/bin/activate

# Tryb skryptowy — wykonuje wszystkie komórki po kolei
python notebook.py

# Tryb interaktywny w przeglądarce
marimo edit notebook.py

# Tryb prezentacyjny (read-only)
marimo run notebook.py
```

Po treningu artefakty są w `results/`. CSV można otworzyć w Excelu, PNG w przeglądarce.

---

## 14. Krótkie FAQ

**Q: Czemu używamy GPU? Bez GPU?**
A: 1 epoka MobileNetV2 na CPU = ~5 min. Na RTX 5080 = ~5 sekund. 100× speedup.
Bez GPU pełna pętla zajęłaby ~70 godzin zamiast ~70 minut.

**Q: Co to znaczy `tf.data.AUTOTUNE`?**
A: TF sam decyduje ile wątków używać do `map`/`prefetch`. Zwykle dobiera się dynamicznie.

**Q: Co to `prefetch`?**
A: Gdy GPU trenuje na batchu N, CPU już przygotowuje batch N+1. Bez tego GPU czekałoby
na CPU. Z tym GPU pracuje cały czas.

**Q: Czemu `EarlyStopping`?**
A: Jeśli `val_loss` przestaje spadać przez `patience` epok, przerywamy trening. Chroni przed:
- Overfittingiem (model zaczyna się "uczyć na pamięć")
- Marnowaniem czasu (kolejne epoki nic nie poprawiają)
- `restore_best_weights=True` przywraca wagi z najlepszej epoki, nie ostatniej

**Q: Czemu `tf.keras.backend.clear_session()` po każdym runie?**
A: TF trzyma graph i wagi w VRAM. Po skończonym runie zwalniamy żeby kolejny model się zmieścił.
Bez tego po 3-4 runach VRAM się zapełnia i dostajemy OOM.

**Q: Co to `class_index` w cards.csv vs nasz `label_idx`?**
A: Autor datasetu ma swoje numerowanie. My robimy `sorted(unique(labels))` i mapujemy do 0..52.
Nasze `label_idx` jest deterministyczne i niezależne od kolejności wczytywania.

**Q: Czemu nie używamy oryginalnego splitu z `data set` w CSV?**
A: Bo zadanie wymaga zbadania **różnych podziałów %**. Oryginalny split to ~93/3/3 — żadnego
porównania by nie było. Robimy własne stratified splity (50/50, 60/40, 70/30, 80/20).

**Q: Co znaczy "stratified"?**
A: Zachowujący proporcje klas. Bez stratify mogłoby się zdarzyć że klasa "joker" trafia w całości
do testu (ma tylko ~150 obrazów). Z stratify każda klasa ma ten sam procent w train i test.

**Q: A co jeśli wynik mojego konkretnego modelu jest słaby (np. 60%)?**
A: To OK. Pokazujesz 5 modeli w 4 splitach = 20 punktów danych. Możesz w sprawozdaniu
analizować *dlaczego* niektóre lepsze (architektura, rozmiar, etc). Sam fakt że masz pełną
metodologię (CM, P/R/F1, ROC, porównanie modeli, analiza splitu) wystarczy na 5.

---

## 15. Podsumowanie najkrótsze

- **TensorFlow + Keras** robi 90% pracy: ładowanie modeli z ImageNet, trening, predykcja
- **scikit-learn** dorzuca metryki (CM, P/R/F1, ROC/AUC)
- Trenujemy **5 modeli × 4 splity = 20 eksperymentów**
- Każdy eksperyment to **transfer learning**: pre-trained backbone z ImageNet + nasza głowa
- **Dwufazowy trening**: faza 1 (frozen, lr=1e-3, 6 epok) → faza 2 (top 1/3 unfreeze, lr=1e-5, 10 epok)
- **Augmentacja** podczas treningu (NIE w teście) — żeby się nie przeuczyć
- Wyniki: tabela CSV + 4 wykresy PNG + surowe dane (npy/json) do dalszej analizy
