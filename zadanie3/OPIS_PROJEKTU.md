# Zadanie 3 — Klasyfikacja obrazów kart do gry

## 1. Co robimy i po co

Mamy **8154 obrazów kart do gry** w rozdzielczości 224×224 pikseli, każdy w kolorach RGB.
Karty należą do **53 klas**: 52 standardowe karty (4 kolory × 13 wartości) plus joker.

Zadanie polega na nauczeniu sieci neuronowej, żeby po pokazaniu jej zdjęcia karty odpowiedziała:
"to jest król kier" albo "to jest dwójka pik". To klasyczne **zadanie klasyfikacji obrazów** —
jeden obraz → jedna etykieta z predefiniowanego zbioru klas.

Dlaczego to interesujące technicznie? Karty wyglądają podobnie: trójka pik i trójka trefl mają
ten sam kształt cyfry, tylko inny kolor symbolu. As i dama mogą się różnić jednym detalem
w narożniku. Sieć musi nauczyć się rozróżniać te subtelne różnice, a nie tylko grube kształty.

Trenujemy **5 różnych architektur CNN** (MobileNetV2, ResNet50, EfficientNetB0, InceptionV3,
VGG16) każdą na **5 różnych podziałach danych** (50/50, 60/40, 70/30, 80/20, 90/10), co daje
łącznie **25 niezależnych eksperymentów**. Dla każdego liczymy accuracy, precision, recall, F1
oraz krzywą ROC z metryką AUC.

---

## 2. Czym jest sieć konwolucyjna (CNN)

### Problem ze zwykłą siecią Dense

Obraz 224×224 RGB to 224 × 224 × 3 = **150 528 liczb**. Gdybyśmy każdą z nich podali na wejście
zwykłej sieci Dense (w pełni połączonej), każdy neuron w pierwszej warstwie miałby 150 528
wag. Przy 512 neuronach to już 77 milionów parametrów w samej pierwszej warstwie — za dużo
do nauczenia z kilkoma tysiącami obrazów.

Co ważniejsze, sieć Dense nie rozumie, że piksele obok siebie są ze sobą powiązane. Piksel
w pozycji (50, 50) i piksel w (51, 50) są traktowane jak dwie niezwiązane ze sobą zmienne.
Sieć ignoruje **lokalność** i **przestrzenną strukturę** obrazu.

### Warstwa konwolucyjna

Sieć konwolucyjna rozwiązuje oba problemy przez filtr (zwany też kernelem) — małą tablicę
liczb, np. 3×3. Filtr **przesuwa się po całym obrazie** w oknie przesuwnym i w każdej
pozycji liczy iloczyn skalarny z fragmentem obrazu pod spodem. To daje jedną liczbę —
odpowiedź na pytanie "jak bardzo ten wzorzec pasuje do tego fragmentu".

```
Obraz:           Filtr 3×3:       Fragment obrazu:
. . . . .        1  0 -1          4  2  1
. [4 2 1] .      2  0 -2     →   5  1  0   → suma iloczynów = wartość mapy cech
. [5 1 0] .      1  0 -1          3  2  1
. [3 2 1] .
. . . . .
```

Filtr jest przesuwany przez cały obraz, co daje **mapę cech** (feature map) — obraz tej
samej rozdzielczości (lub mniejszej), gdzie każdy piksel mówi "jak mocno tu był wykryty
ten wzorzec". Kluczowe: **jeden filtr ma tylko 9 parametrów** (3×3), a działa na całym obrazie.
To drastyczna redukcja liczby parametrów.

W warstwie konwolucyjnej jest zwykle wiele filtrów równolegle (np. 64 filtry), każdy wykrywa
inny wzorzec. Warstwa zwraca stos 64 map cech.

### Pooling

Po warstwie konwolucyjnej często stosuje się **pooling** (np. MaxPooling 2×2): dzielimy mapę
cech na bloki 2×2 i z każdego bloku zostawiamy tylko maksimum. To zmniejsza rozdzielczość
dwukrotnie w obu wymiarach, redukując ilość danych i wymuszając na sieci patrzenie na
grubsze wzorce.

### Hierarchia cech

Warstwy CNN układają się hierarchicznie. W typowej sieci głębokiej dzieje się tak:

| Poziom warstwy | Co filtrują |
|---|---|
| Pierwsze warstwy | Krawędzie poziome, pionowe, ukośne; gradienty kolorów |
| Środkowe warstwy | Proste kształty: narożniki, kółka, linie; tekstury |
| Głębsze warstwy | Części obiektów: cyfry, symbole, twarze figur |
| Najgłębsze warstwy | Całe obiekty i ich kombinacje |

Ta hierarchia sprawia, że CNN jest **naturalnie dopasowana do rozpoznawania obiektów na
obrazach** — od pikseli do semantyki w jednym łańcuchu operacji, gdzie każda warstwa buduje
na wynikach poprzedniej.

### GlobalAveragePooling i klasyfikacja

Na końcu konwolucyjnej części sieci mamy stos map cech o kształcie np. `[7, 7, 2048]`
(7×7 pikseli, 2048 map). `GlobalAveragePooling2D` uśrednia każdą z 2048 map cech do jednej
liczby, dając wektor `[2048]`. Ten wektor reprezentuje "co jest na obrazie" w kompaktowej formie.
Potem Dense z softmax zamienia ten wektor na prawdopodobieństwa dla każdej klasy.

---

## 3. Czym jest transfer learning

### Problem: za mało danych

Wytrenowanie CNN od zera wymaga **milionów obrazów**. ImageNet — benchmark do trenowania
sieci rozpoznających obiekty — zawiera 1,2 miliona obrazów w 1000 klasach. Trening
ResNet50 na ImageNet na klastrze GPU zajmuje **kilka dni i kosztuje tysiące dolarów**.

My mamy 8154 obrazów. Gdybyśmy uczyli sieć od zera (losowe wagi), prawdopodobnie utknelibyśmy
na kilkunastu procentach accuracy — sieć nie ma wystarczająco dużo danych, żeby nauczyć się
wykrywać krawędzie, kształty, a następnie cyfry i symbole kart.

### Gotowe wagi z ImageNet

Na szczęście wagi modeli wytrenowanych na ImageNet są **publicznie dostępne**. Wywołanie:

```python
backbone = keras.applications.ResNet50(weights="imagenet", include_top=False)
```

pobiera ResNet50 z wagami, które kodują wszystko czego sieć nauczyła się z 1,2 miliona obrazów.
Te wagi "wiedzą" jak wykrywać krawędzie, tekstury, kształty, części obiektów. Ta wiedza jest
**uniwersalna** — karta do gry, pies, samolot — wszystkie korzystają z tych samych podstawowych
wzorców wizualnych.

### Backbone + head

W transfer learning dzielimy sieć na dwie części:

**Backbone** (kręgosłup) — głęboka sieć konwolucyjna z wagami z ImageNet. Zamrażamy go
(`backbone.trainable = False`), co oznacza że gradienty go nie aktualizują. Backbone działa
wyłącznie jako **ekstraktor cech**: zamienia obraz `[224, 224, 3]` na wektor cech
(np. `[2048]` dla ResNet50), który reprezentuje "co jest na obrazie".

**Head** (głowa) — nowe warstwy dodane na wierzchu, z losowo zainicjalizowanymi wagami.
W naszym przypadku:

```python
x = Dropout(0.5)(x)                              # regularizacja
outputs = Dense(53, activation="softmax")(x)     # 53 klasy kart
```

Head uczymy od podstaw. Dostaje wektor cech z backbone i uczy się mapować go na 53 klasy.
Zamiast trenować 25 milionów parametrów ResNet50, trenujemy tylko **~110 tysięcy parametrów**
głowy. To:
- **szybkie** — kilkadziesiąt sekund na epokę zamiast godzin
- **stabilne** — z 8000 obrazów nie psujemy gotowych dobrych wag
- **skuteczne** — backbone już "widzi" kształty i wzorce, głowa uczy się tylko przypisywania ich do klas kart

### Fine-tuning (faza 2)

Sam zamrożony backbone okazał się niewystarczający (accuracy ~45-50%). Wagi z ImageNet
zostały nauczone na naturalnych zdjęciach zwierząt i obiektów, a karty do gry to
schematyczne obrazy z symbolami i cyframi — istotnie inne.

Rozwiązaniem jest **fine-tuning**: po wytrenowaniu głowy odmrażamy **górną 1/3 backbone'u**
i trenujemy całość jeszcze raz, ale ze znacznie mniejszym współczynnikiem uczenia:

```python
# Odmrażamy górną 1/3 warstw backbone'u
n_unfreeze = int(n_layers * 1/3)
for layer in backbone.layers[:-n_unfreeze]:
    layer.trainable = False   # dolne 2/3 nadal zamrożone
# BatchNormalization zawsze zamrożona
for layer in backbone.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False
```

**Dlaczego górna 1/3, a nie cały backbone?**
Dolne warstwy (krawędzie, kolory) są prawdziwie uniwersalne — nie ma sensu ich zmieniać.
Górne warstwy są bardziej specyficzne dla ImageNetu i warto je dostroić do domeny kart.

**Dlaczego `lr=1e-5` zamiast `lr=1e-3`?** W fazie 1 trenujemy losowo zainicjalizowaną głowę —
duży gradient jest potrzebny. W fazie 2 modyfikujemy wagi które **już są dobre**. Duże lr
zniszczyłoby je. Zasada: fine-tuning używa 100× mniejszego lr niż trening od zera.

**Dlaczego BatchNormalization zostają zamrożone?** Warstwy BN przechowują statystyki populacji
(średnia, wariancja) obliczone na ImageNet. Aktualizowanie ich na małym zbiorze 8000 obrazów
kart degradowałoby model. Zostawiamy je w trybie `training=False` — używają zapisanych
statystyk z ImageNet.

---

## 4. Dataset

**Źródło:** [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)
Autor: gpiosenka. Pobranie przez Kaggle CLI:

```bash
kaggle datasets download -d gpiosenka/cards-image-datasetclassification
```

**Zawartość:**
- 8154 obrazów, 224×224 piksele, RGB (pliki JPEG)
- 53 klasy: as/dwójka/.../król w każdym z 4 kolorów + joker
- Każda klasa ma ~143-155 obrazów (zbiór zbalansowany)

Oryginalny dataset zawiera gotowy podział:

```
data/
├── cards.csv          # metadane: ścieżka, etykieta, zbiór (train/valid/test)
├── train/             # ~7624 obrazów (~144 per klasa)
├── valid/             # 265 obrazów (5 per klasa)
└── test/              # 265 obrazów (5 per klasa)
```

**Dlaczego ignorujemy oryginalny podział?** Oryginalny podział daje proporcję ~93% / 3% / 3%.
Zadanie wymaga zbadania wpływu różnych proporcji podziału (50/50, 60/40, 70/30, 80/20, 90/10).
Dlatego **łączymy wszystkie obrazy w jeden zbiór** i tworzymy własne podziały.

### Stratified split

Podział na train/test wykonujemy funkcją `train_test_split` z parametrem `stratify=labels`:

```python
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_paths, all_labels,
    train_size=train_frac,
    stratify=all_labels,
    random_state=42,
)
```

**Co to znaczy stratified?** Podział zachowujący proporcje klas. Jeśli w całym zbiorze
joker stanowi 1/53 wszystkich obrazów, to po stratified split joker stanowi dokładnie 1/53
zarówno w zbiorze treningowym, jak i testowym. Bez `stratify` losowy podział mógłby
przypadkowo wrzucić wszystkie obrazy rzadkiej klasy do jednego zbioru, co uniemożliwiłoby
trening lub ocenę.

Dla podziału 80/20 na zbiorze 8154 obrazów:
- Train: ~6523 obrazów (~123 per klasa)
- Test: ~1631 obrazów (~31 per klasa)

---

## 5. Pięć modeli — co je odróżnia

Wszystkie 5 architektur robimy to samo: przyjmują obraz `[224, 224, 3]`, przepuszczają
przez konwolucyjny backbone, i zwracają wektor cech. Różnią się **jak budują te cechy**.

### MobileNetV2 (2018)

**Idea:** Depthwise separable convolution. Zwykła konwolucja 3×3 działa jednocześnie na
wszystkich kanałach. MobileNetV2 rozkłada ją na dwa tańsze kroki:
1. *Depthwise*: każdy kanał filtrowany osobno (przestrzenna informacja)
2. *Pointwise*: konwolucja 1×1 miksuje kanały ze sobą (kanalowa informacja)

To około 8-9× mniej operacji niż standardowa konwolucja przy zbliżonej jakości.
Dodatkowo używa "inverted residuals" — bloki zwiększają szerokość wewnątrz, a nie na
zewnątrz.

| Właściwość | Wartość |
|---|---|
| Parametry | 3,5M |
| Wektor cech | 1280 |
| Przeznaczenie | Urządzenia mobilne, edge computing |

### ResNet50 (2015)

**Idea:** Residual connections (połączenia resztkowe). Problem głębokich sieci: w sieciach
>20 warstw gradienty zanikają podczas propagacji wstecznej — warstwy na początku sieci
praktycznie przestają się uczyć (problem zanikającego gradientu).

Residual connection rozwiązuje to dodając **skrót**: `output = f(x) + x`. Zamiast uczyć
transformacji `f(x)`, sieć uczy się *reszty* `f(x) = output - x`. Nawet jeśli `f(x)` daje
zerowe gradienty, gradient przepływa wprost przez `+x`. To umożliwiło budowanie sieci
50, 101, 152 i więcej warstw.

| Właściwość | Wartość |
|---|---|
| Parametry | 25M |
| Wektor cech | 2048 |
| Głębokość | 50 warstw |

### EfficientNetB0 (2019)

**Idea:** Compound scaling. Poprzednie sieci były skalowane arbitralnie — dodawano warstwy,
zwiększano szerokość, albo podawano większe obrazy. EfficientNet zbadał matematycznie
**optymalne proporcje** między głębokością (liczba warstw), szerokością (liczba filtrów)
i rozdzielczością wejściową. B0 to punkt bazowy tej rodziny.

| Właściwość | Wartość |
|---|---|
| Parametry | 5,3M |
| Wektor cech | 1280 |
| Efektywność | Najlepsza accuracy/parametr w rodzinie |

### InceptionV3 (2015)

**Idea:** Moduły Inception. Zamiast wybierać "jaki rozmiar filtra" (3×3? 5×5?), stosujemy
**kilka rozmiarów równolegle** i sklejamy wyniki:

```
                ┌── conv 1×1 ──┐
Wejście ─── ──┤── conv 3×3 ──├─── concat ─── Wyjście
                ├── conv 5×5 ──┤
                └── MaxPool ───┘
```

Sieć "patrzy na kilka skal jednocześnie" — małe filtry wykrywają detale, duże — szerokie
konteksty. Konwolucje 5×5 są rozkładane na dwie 3×3 (tańsze obliczeniowo).

| Właściwość | Wartość |
|---|---|
| Parametry | 24M |
| Wektor cech | 2048 |
| Specjalność | Multi-scale features |

### VGG16 (2014)

**Idea:** Prostota przez głębokość. Architektura to po prostu stosy bloków `Conv 3×3 → Conv 3×3 → MaxPool`. Żadnych skrótów, żadnych trików — tylko głębokość i małe filtry.
Był przełomowy w 2014, dziś jest punktem odniesienia i przykładem że "stara architektura
może nadal działać".

| Właściwość | Wartość |
|---|---|
| Parametry (bez głowy) | ~15M |
| Wektor cech | 512 |
| Uwaga | Oryginalnie 138M params, ale większość to Dense na końcu — my go odcinamy |

### Każdy model ma swój preprocess_input

To szczegół krytyczny. Każda architektura była wytrenowana na ImageNet z inną normalizacją
pikseli. Podanie obrazu w "złym formacie" degraduje wyniki:

```python
resnet50.preprocess_input(x)        # odejmuje średnią ImageNet [103.9, 116.8, 123.7], konwertuje BGR
mobilenet_v2.preprocess_input(x)    # skaluje piksele [0, 255] → [-1, 1]
efficientnet.preprocess_input(x)    # skaluje [0, 255] → [0, 1]
inception_v3.preprocess_input(x)    # skaluje [0, 255] → [-1, 1]
vgg16.preprocess_input(x)           # odejmuje średnią ImageNet, BGR
```

W naszym kodzie każdy model niesie ze sobą swój `preprocess_fn` i jest on stosowany
jako ostatni krok w pipeline danych, po augmentacji.

---

## 6. Pipeline od A do Z

### 6.1 Wczytanie danych

```python
df = pd.read_csv("data/cards.csv")
df["full_path"] = "data/" + df["filepaths"]
df = df[df["full_path"].map(os.path.isfile)].reset_index(drop=True)
class_names = sorted(df["card type"].unique())         # 53 klasy w kolejności alfabetycznej
label_to_idx = {name: i for i, name in enumerate(class_names)}
df["label_idx"] = df["card type"].map(label_to_idx)   # etykieta numeryczna 0..52
```

`cards.csv` zawiera kolumny: `filepaths` (ścieżka względna), `card type` (nazwa klasy),
`data set` (oryginalne przypisanie train/valid/test — ignorujemy).

### 6.2 Stratified split

```python
train_paths, test_paths, train_labels, test_labels = train_test_split(
    df["full_path"].to_numpy(),
    df["label_idx"].to_numpy(),
    train_size=0.80,          # np. 80/20
    stratify=df["label_idx"].to_numpy(),
    random_state=42,
)
```

### 6.3 Pipeline tf.data

TensorFlow dostarcza `tf.data.Dataset` — leniwy (lazy) potok danych, który ładuje i
przetwarza obrazy w tle podczas gdy GPU trenuje:

```python
ds = tf.data.Dataset.from_tensor_slices((paths, labels))
if training:
    ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

# Wczytanie i dekodowanie
def _decode(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    return img, label

ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(32)

# Augmentacja (tylko trening)
if training:
    ds = ds.map(lambda x, y: (augment(x, training=True), y))

# Normalizacja specyficzna dla architektury
ds = ds.map(lambda x, y: (preprocess_fn(x), y))
return ds.prefetch(tf.data.AUTOTUNE)
```

`prefetch` sprawia, że gdy GPU trenuje na batchu N, CPU już przygotowuje batch N+1 —
GPU nigdy nie czeka.

`AUTOTUNE` pozwala TensorFlow dynamicznie dobrać liczbę wątków dla operacji `map`.

### 6.4 Augmentacja

Augmentacja polega na losowym modyfikowaniu obrazów **w trakcie treningu** zanim trafią
do sieci. Cel: sieć widzi za każdą epoką nieco inne obrazy → trudniej jej nauczyć się na
pamięć konkretnych przykładów → lepsza generalizacja.

```python
augment = tf.keras.Sequential([
    RandomRotation(0.03),                          # ±10.8° (0.03 × 360°)
    RandomZoom(0.08),                              # ±8% zoom
    RandomTranslation(0.05, 0.05),                 # ±5% przesunięcie poziome i pionowe
    RandomBrightness(0.15, value_range=(0, 255)),  # ±15% jasność
    RandomContrast(0.15),                          # ±15% kontrast
])
```

**Augmentacja działa wyłącznie na zbiorze treningowym.** Na zbiorze testowym stosujemy
obrazy bez modyfikacji — oceniamy model na "normalnych" danych.

**Dlaczego nie `RandomFlip`?** Karta do gry jest **obiektem o stałej orientacji**. Król
pik obrócony poziomo wygląda inaczej niż normalny (twarz figury się odwraca). Uczenie
sieci na poziomo odwróconych kartach uczyłoby ją błędnych wzorców i mieszałoby etykiety.

**Dlaczego nie większe rotacje?** Obrót o 90° daje kartę "na boku", co nie jest naturalnym
widokiem karty. Zachowujemy małe rotacje (do ~11°) symulujące nieidealne trzymanie karty.

**`RandomBrightness` i `RandomContrast`** dodano w odpowiedzi na to, że dataset zawiera
mix realnych zdjęć i obrazów generowanych przez AI o różnych stylach kolorystycznych.
Augmentacja kolorystyczna pomaga sieci być odporna na te różnice.

### 6.5 Budowanie modelu

```python
backbone = ResNet50(
    include_top=False,          # bez oryginalnej głowy 1000-klasowej
    weights="imagenet",         # wagi z ImageNet
    input_shape=(224, 224, 3),
    pooling="avg",              # GlobalAveragePooling2D na końcu backbone'u
)
backbone.trainable = False      # zamrożony w fazie 1

inputs = tf.keras.Input(shape=(224, 224, 3))
x = backbone(inputs, training=False)   # training=False: BN w trybie inference
x = Dropout(0.5)(x)
outputs = Dense(53, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

`include_top=False, pooling="avg"` — odcinamy oryginalną Dense(1000) z ImageNet
i dodajemy GlobalAveragePooling2D, który spłaszcza mapy cech do wektora.

### 6.6 Dwufazowy trening

**Faza 1: tylko głowa (8 epok, lr=1e-3)**

Backbone zamrożony — trenujemy tylko 2 warstwy głowy (~110K parametrów).
Szybko, stabilnie. Dostosowuje losowo zainicjalizowaną głowę do domeny kart.

```python
cb = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
h1 = model.fit(train_ds, validation_data=test_ds, epochs=8, callbacks=[cb])
```

**Faza 2: fine-tuning górnej 1/3 backbone'u (12 epok, lr=1e-5)**

```python
backbone.trainable = True
n_unfreeze = max(1, int(len(backbone.layers) / 3))
for layer in backbone.layers[:-n_unfreeze]:
    layer.trainable = False       # dolne 2/3 nadal zamrożone
for layer in backbone.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False   # BN zawsze zamrożone

model.compile(optimizer=Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h2 = model.fit(train_ds, validation_data=test_ds, epochs=12, callbacks=[cb])
```

**EarlyStopping** przerywa trening gdy `val_loss` nie spada przez 5 epok z rzędu.
`restore_best_weights=True` przywraca wagi z epoki o najniższym `val_loss` — nie ostatniej.
To zabezpieczenie przed przeuczeniem: model przestaje się poprawiać na zbiorze walidacyjnym,
chociaż dalej zmniejsza stratę na zbiorze treningowym.

Historia obu faz jest scalana przed zapisem do pliku JSON.

### 6.7 Ewaluacja

```python
y_prob = model.predict(test_ds, verbose=0)   # macierz [N, 53] prawdopodobieństw
y_pred = y_prob.argmax(axis=1)               # klasa z najwyższym prawdopodobieństwem
y_true = np.concatenate([y.numpy() for _, y in test_ds])  # prawdziwe etykiety
```

Na podstawie `y_true`, `y_pred`, `y_prob` liczymy wszystkie metryki (sekcja 7).

### 6.8 Zarządzanie pamięcią GPU

Po każdym z 25 eksperymentów:

```python
del model, history, test_ds
tf.keras.backend.clear_session()
gc.collect()
```

TensorFlow przechowuje grafy i wagi w VRAM. Bez czyszczenia po 3-4 modelach VRAM
się zapełnia i kolejny model nie może się załadować (błąd OOM — Out Of Memory).

---

## 7. Metryki — co mierzymy i jak to interpretować

### 7.1 Macierz pomyłek (Confusion Matrix)

Macierz pomyłek to tablica `[N_klas × N_klas]`, gdzie komórka `CM[i][j]` zawiera liczbę
obrazów **prawdziwej klasy i** zaklasyfikowanych jako **klasa j**. Diagonala `CM[i][i]`
to poprawne klasyfikacje.

Przykład dla 3-klasowego problemu (uproszczony):

```
              | Przewidziana klasa:
              | pik   kier  trefl
Prawdziwa  pik|  45     3     2    ← było 50 pików, 45 poprawnych
klasa:    kier|   4    39     7    ← było 50 kierów, 39 poprawnych, 7 mylonych z treflem
         trefl|   1     2    47    ← było 50 treflów, 47 poprawnych
```

W naszym projekcie macierz ma rozmiar **53×53** — za duża do czytania jako tabela.
Dlatego rysujemy ją jako **heatmapę** (jasne komórki = duże wartości). Idealna sieć
miałaby jasną przekątną i ciemne pozostałe komórki.

```python
cm = confusion_matrix(y_true, y_pred, labels=list(range(53)))
sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
```

Analizując macierz pomyłek możemy odpowiedzieć: *które karty są ze sobą mylone
najczęściej?* Oczekujemy, że karty tej samej wartości w różnych kolorach (np. trójka
pik i trójka trefl) będą częściej mylone niż karty bardzo różnych wartości.

### 7.2 Accuracy

Najprostsza metryka: jaki procent wszystkich obrazów został poprawnie zaklasyfikowany.

```
Accuracy = liczba poprawnych klasyfikacji / całkowita liczba obrazów
```

Dla zbalansowanego datasetu (każda klasa ma podobną liczbę obrazów — tak jak u nas)
accuracy jest miarodajna. Gdyby klasy były niezbalansowane (np. 99% jednej klasy),
accuracy byłaby myląca.

### 7.3 Precision, Recall, F1

Dla każdej klasy `c` definiujemy:
- **TP** (True Positive): obraz klasy `c` rozpoznany poprawnie jako `c`
- **FP** (False Positive): obraz innej klasy błędnie rozpoznany jako `c`
- **FN** (False Negative): obraz klasy `c` błędnie rozpoznany jako inna klasa

| Metryka | Wzór | Pytanie, na które odpowiada |
|---|---|---|
| **Precision** | TP / (TP + FP) | Spośród wszystkich przewidzianych jako klasa `c`, ile faktycznie było klasą `c`? |
| **Recall** | TP / (TP + FN) | Spośród wszystkich prawdziwych przypadków klasy `c`, ile sieć znalazła? |
| **F1** | 2 × Precision × Recall / (Precision + Recall) | Harmoniczna średnia Precision i Recall |

Precision i Recall często są ze sobą w konflikcie: łatwo uzyskać wysokie Recall (przewiduj
każdy obraz jako klasę `c`), ale wtedy Precision spada. F1 karze za taki kompromis —
wymaga że obie metryki muszą być wysokie jednocześnie.

**Macro F1** = średnia F1 po wszystkich 53 klasach (każda klasa liczy się jednakowo).
Używamy macro, bo nasz dataset jest zbalansowany i chcemy, żeby wszystkie klasy miały
równy wpływ na wynik końcowy.

```python
report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
accuracy      = report["accuracy"]
macro_f1      = report["macro avg"]["f1-score"]
macro_prec    = report["macro avg"]["precision"]
macro_recall  = report["macro avg"]["recall"]
```

### 7.4 Krzywa ROC i AUC

**Problem z accuracy i F1:** obie metryki zależą od wybranego progu decyzyjnego.
Domyślnie wybieramy klasę z najwyższym prawdopodobieństwem (`argmax`). Ale gdybyśmy
podnieśli próg do 0.7 (klasyfikuj jako `c` tylko gdy `P(c) > 0.7`), metryki by się
zmieniły. AUC jest **niezależne od progu**.

**Krzywa ROC dla jednej klasy `c` (one-vs-rest):**

1. Traktujemy klasę `c` jako "pozytywną", a wszystkie pozostałe 52 klasy jako "negatywne"
2. Sortujemy obrazy malejąco po wartości `y_prob[:, c]`
3. Iterujemy po możliwych progach od 1.0 do 0.0
4. Dla każdego progu liczymy:
   - **TPR** (True Positive Rate = Recall): ile prawdziwych `c` zostało złapanych
   - **FPR** (False Positive Rate): ile nie-`c` zostało błędnie zaklasyfikowanych jako `c`
5. Rysujemy krzywą (FPR, TPR) — to **krzywa ROC**

Krzywa losowego klasyfikatora to linia ukośna (FPR = TPR). Idealna sieć daje punkt (0, 1)
— zero fałszywych alarmów, wszystkie prawdziwe złapane.

**AUC (Area Under Curve)** = pole pod krzywą ROC:
- `AUC = 0.5` — klasyfikator losowy (bezużyteczny)
- `AUC = 1.0` — idealny klasyfikator
- `AUC = 0.95-0.99` — bardzo dobry wynik praktyczny

**Dla 53 klas** liczymy AUC oddzielnie dla każdej klasy (one-vs-rest), a następnie
**uśredniamy** (macro average — każda klasa z równą wagą):

```python
y_true_onehot = tf.keras.utils.to_categorical(y_true, 53)   # one-hot encoding
auc_macro = roc_auc_score(y_true_onehot, y_prob, average="macro", multi_class="ovr")
auc_micro = roc_auc_score(y_true_onehot, y_prob, average="micro", multi_class="ovr")
```

`auc_micro` waży klasy proporcjonalnie do ich liczności (dla zbalansowanego datasetu
micro ≈ macro).

Na wykresie ROC rysujemy **5 najgorszych + 5 najlepszych klas** (najniższe i najwyższe AUC
per klasa), żeby zobaczyć skrajne przypadki. Rysowanie wszystkich 53 krzywych byłoby nieczytelne.

**Jak interpretować wysokie AUC przy średniej accuracy?** Jeśli model ma `accuracy = 0.72`
ale `AUC = 0.98`, oznacza to: model w 28% przypadków nie trafił top-1, ale prawdziwa klasa
była zwykle w top-2 lub top-3. Sieć "wie" która to karta, ale jest niezdecydowana między
2-3 podobnymi. Dla kart to logiczne — trójka pik i trójka trefl mają niemal identyczny
kształt.

---

## 8. Co i gdzie zapisujemy

### 8.1 metrics.csv

Główna tabela wyników — 1 wiersz = 1 eksperyment (1 model × 1 split):

| Kolumna | Opis |
|---|---|
| `model` | Nazwa architektury (np. "ResNet50") |
| `split` | Proporcja zbioru treningowego (np. 0.8 = 80%) |
| `accuracy` | Dokładność top-1 na zbiorze testowym |
| `macro_f1` | Macro F1 uśrednione po 53 klasach |
| `macro_precision` | Macro Precision |
| `macro_recall` | Macro Recall |
| `auc_macro` | AUC OvR, macro average |
| `auc_micro` | AUC OvR, micro average |
| `epochs_run` | Ile epok faktycznie wykonano (EarlyStopping może skrócić) |
| `train_time_s` | Całkowity czas treningu w sekundach (faza 1 + faza 2) |

Zapis następuje **po każdym zakończonym eksperymencie**, nie po wszystkich. Gdyby trening
crashnął w połowie, zachowane są wyniki z wszystkich ukończonych runów.

### 8.2 Pliki surowych danych

Przechowywane w podfolderach `results/`:

- `cm/<model>_<split>.npy` — macierz pomyłek jako tablica numpy `[53, 53]`
- `probs/<model>_<split>_y_true.npy` — prawdziwe etykiety testu jako wektor `[N]`
- `probs/<model>_<split>_y_prob.npy` — prawdopodobieństwa jako macierz `[N, 53]`
- `history/<model>_<split>.json` — historia treningu (loss i accuracy per epoka, obie fazy)

Nazwy plików używają formatu `<ModelName>_<split_pct>`, np. `ResNet50_80` dla split=0.80.

### 8.3 Wykresy PNG

- `metrics_vs_split.png` — 3 panele obok siebie (Accuracy / Macro F1 / Macro AUC),
  każdy model jako osobna linia, oś X = proporcja treningowa
- `cm_<best>.png` — heatmapa macierzy pomyłek najlepszego modelu (najwyższe accuracy)
- `roc_<best>.png` — krzywe ROC: 5 klas z najniższym AUC + 5 z najwyższym AUC,
  dla najlepszego modelu
- `learning_curves.png` — krzywe uczenia (loss i accuracy) dla wszystkich 25 runów,
  po jednym panelu na model, różne kolory dla różnych splitów

---

## 9. Mechanizm resume (wznawianie)

Główna pętla sprawdza przed każdym eksperymentem, czy wynik już istnieje w `metrics.csv`:

```python
if os.path.exists(csv_path):
    done = {(row["model"], float(row["split"])) for row in pd.read_csv(csv_path).to_dict("records")}
else:
    done = set()

for model_name, split in runs:
    if (model_name, split) in done:
        print(f"skip {model_name} {split} (cached)")
        continue
    # ... trening i zapis
```

Dzięki temu: jeśli trening przerwiesz i uruchomisz ponownie, pominięte zostaną tylko
eksperymenty, które już się zakończyły i mają wynik w CSV. Nie trzeba zaczynać od nowa.

---

## 10. Jak uruchomić

**Wymagania:** Python z TensorFlow 2.18+, CUDA 12.x, cuDNN 9.x (dla GPU), marimo.

```bash
cd /home/nikodem/tiad/zadanie3
source .venv/bin/activate

# Tryb skryptowy — wykonuje wszystkie komórki sekwencyjnie, wyniki w results/
python notebook.py

# Tryb interaktywny — otwiera przeglądarkę z reaktywnym notebookiem
marimo edit notebook.py

# Tryb prezentacyjny (read-only, bez możliwości edycji)
marimo run notebook.py
```

Czas trwania pełnego treningu (25 eksperymentów) na GPU RTX 5080: ~60-90 minut.
Na CPU: kilkadziesiąt godzin.

Po zakończeniu:
- `results/metrics.csv` — pełna tabela wyników
- `results/metrics_vs_split.png` — wykres porównawczy
- `results/learning_curves.png` — krzywe uczenia
- `results/cm_<best>.png` — macierz pomyłek najlepszego modelu
- `results/roc_<best>.png` — krzywe ROC

---

## 11. Struktura pliku notebook.py

Marimo to reaktywny notebook — plik to czysty Python (nie JSON jak Jupyter). Komórki
to funkcje z dekoratorem `@app.cell`. Marimo wykonuje je w kolejności **zależności
zmiennych**, nie w kolejności pisania: jeśli komórka B używa zmiennej z komórki A,
B wykona się po A.

| Komórka | Zawartość |
|---|---|
| 1 | Import marimo |
| 2 | Markdown: nagłówek projektu |
| 3 | Setup TF: LD_LIBRARY_PATH, GPU memory growth |
| 4 | Konfiguracja: stałe (SPLITS, MODEL_NAMES, lr, epochs, batch size) |
| 5 | Tworzenie katalogów `results/`, `models/` |
| 6 | Wczytanie cards.csv → DataFrame, mapowanie etykiet |
| 7 | Podgląd DataFrame (tabela interaktywna marimo) |
| 8 | Funkcja `make_split(train_frac)` — stratified split |
| 9 | Funkcja `build_dataset(...)` — potok tf.data z augmentacją |
| 10 | `MODEL_REGISTRY` + funkcja `build_model(name)` — backbone + head |
| 11 | Funkcja `train_one(...)` — dwufazowy trening z EarlyStopping |
| 12 | Funkcja `evaluate(model, test_ds)` — obliczanie metryk |
| 13 | Główna pętla: 5 modeli × 5 splitów, resume support, cleanup VRAM |
| 14 | Markdown: nagłówek "Tabela zbiorcza" |
| 15 | Tabela summary z `metrics_df` |
| 16-17 | Pivot tables: accuracy/F1/AUC × model × split |
| 18 | Wykres `metrics_vs_split.png` |
| 19 | Heatmapa CM najlepszego modelu |
| 20 | Wykres ROC dla najlepszego modelu |
| 21 | Krzywe uczenia wszystkich runów (`learning_curves.png`) |

---

## 12. Stack technologiczny

| Biblioteka | Rola |
|---|---|
| **TensorFlow 2.18+ / Keras** | Definicja sieci, trening, predykcja; `keras.applications` dostarcza backbone'y z wagami ImageNet |
| **scikit-learn** | Metryki: `confusion_matrix`, `classification_report`, `roc_auc_score`, `train_test_split` |
| **matplotlib + seaborn** | Wykresy: krzywe uczenia, heatmapy, ROC, słupkowe porównania |
| **pandas + numpy** | Operacje na danych, CSV, macierze |
| **marimo** | Reaktywny notebook: plik `.py` zamiast JSON, komórki wykonują się wg grafu zależności |
| **kaggle CLI** | Pobranie datasetu z Kaggle |
| **CUDA 12.x + cuDNN 9.x** | Akceleracja GPU — bez GPU trening trwałby ~50× dłużej |

TensorFlow z Keras robi większość ciężkiej pracy: ładowanie modeli, propagacja wsteczna,
optymalizacja, obsługa GPU. scikit-learn dostarcza metryki których TF nie ma (pełny ROC/AUC).

---

## 13. Podsumowanie eksperymentu

25 eksperymentów (5 modeli × 5 splitów) dostarcza pełnego obrazu:

- **Porównanie architektur:** która sieć radzi sobie najlepiej z klasyfikacją kart?
  Spodziewamy się, że nowsze architektury (EfficientNetB0, MobileNetV2) mogą wypaść lepiej
  niż VGG16 mimo mniejszej liczby parametrów.

- **Wpływ podziału danych:** czy więcej danych treningowych zawsze daje lepszy model?
  Spodziewamy się monotonicznych wzrostów accuracy przy zwiększaniu proporcji treningowej.

- **Analiza macierzy pomyłek:** które karty są ze sobą mylone? Oczekujemy największych
  pomyłek między kartami tej samej wartości w różnych kolorach (np. trójka pik ↔ trójka trefl),
  bo sieć patrzy głównie na kształt cyfry/figury, a kolor symbolu to mały detal.

- **AUC vs accuracy:** wysoka AUC przy umiarkowanej accuracy wskazuje że sieć zna prawdziwą
  odpowiedź, ale jest niezdecydowana między 2-3 podobnymi klasami.
