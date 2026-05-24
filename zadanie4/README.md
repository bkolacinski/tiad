# Zadanie 4 — DESABOA: modyfikacja SABOA zwiększająca różnorodność roju

Propozycja i empiryczna ocena modyfikacji algorytmu rojowego **SABOA**
(Self-Adaption Butterfly Optimization Algorithm, Fan i in., *IEEE Access* 2020),
zbudowanego na **BOA** (Arora & Singh, *Soft Computing* 2019).

Modyfikacja **DESABOA** (Diversity-Enhanced SABOA) dodaje dwa mechanizmy
przeciwdziałające przedwczesnej zbieżności:

1. **Opposition-Based Learning (OBL)** przy inicjalizacji — szersze, lepiej
   rozłożone pokrycie przestrzeni na starcie.
2. **Mutacja sterowana różnorodnością roju** — gdy różnorodność (średnia odległość
   od centroidu) spada poniżej `θ·D₀` lub najlepsze rozwiązanie stagnuje,
   najgorsze osobniki są perturbowane lotem Lévy'ego wokół najlepszego lub skokiem
   Cauchy'ego, co przywraca różnorodność i pozwala uciec z minimów lokalnych.

Badanie: **CEC2017** (pakiet `opfunu`), **D = 10**, **12 funkcji** z czterech
kategorii, **51 niezależnych przebiegów**, identyczny budżet `MAX_FES = 10000·D`
dla obu algorytmów. Wyższość DESABOA potwierdzana **testem Wilcoxona signed-rank**.

## Instalacja

```bash
uv venv .venv --python 3.12          # lub: python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

> `opfunu` wymaga `pkg_resources`, dlatego przypięto `setuptools<81`
> (w nowszych wydaniach moduł został usunięty).

## Uruchomienie

```bash
# pełne badanie: 12 funkcji × 51 przebiegów × 2 algorytmy (~1.5 h na 4 rdzeniach)
.venv/bin/python compare.py

# szybki test całego pipeline'u (sekundy)
.venv/bin/python compare.py --quick

# test poprawności algorytmów (budżet, monotoniczność zbieżności, determinizm)
.venv/bin/python smoke_test.py
```

Flagi `compare.py`: `--force` (ignoruj cache), `--no-parallel`, `--quick`.

## Architektura

| Moduł | Rola |
|---|---|
| `config.py` | Jedyne źródło prawdy: D, POP, MAX_FES, lista funkcji, parametry DESABOA, SEED |
| `benchmark.py` | Wrapper CEC2017 z licznikiem FE; `evaluate()` zwraca błąd = f(x) − f\* |
| `algorithms/boa.py` | BOA (Arora & Singh 2019) — opcjonalna druga baza |
| `algorithms/saboa.py` | SABOA (Fan i in. 2020) — algorytm bazowy |
| `algorithms/desaboa.py` | **DESABOA** — SABOA + OBL + mutacja sterowana różnorodnością (wkład pracy) |
| `diversity.py` | Pomiar różnorodności, OBL, lot Lévy'ego (Mantegna), skok Cauchy'ego |
| `runner.py` | Siatka algorytmy × funkcje × przebiegi, multiprocessing, cache `.npz` |
| `stats.py` | Test Wilcoxona signed-rank, podsumowanie win/tie/loss |
| `plotting.py` | Krzywe zbieżności (log-y) i boxploty błędu końcowego |
| `compare.py` | Orkiestrator → `results/report.md` + tabele + wykresy |

### Uczciwość porównania (budżet FE)

Pętla każdego algorytmu jest sterowana licznikiem ewaluacji
(`while n_fe + POP <= MAX_FES`), a dodatkowe ewaluacje DESABOA (OBL, re-injekcja)
są wliczane do tego samego budżetu. Krzywe zbieżności próbkowane są w stałych
punktach FE, więc są bezpośrednio uśrednialne między algorytmami o różnej liczbie
ewaluacji na iterację.

## Wyniki

Po uruchomieniu `compare.py` w katalogu `results/`:

- `report.md` — pełne sprawozdanie (opis modyfikacji, konfiguracja, tabele
  mean±std, wykresy, test Wilcoxona, wnioski),
- `metrics.csv`, `wilcoxon.csv` — dane liczbowe,
- `convergence/F*.png`, `boxplots/F*.png` — wykresy per funkcja,
- `runs/*.npz` — surowe wyniki przebiegów (cache, regenerowalne).

## Źródła

- A. Lin, W. Sun i in., *Global genetic learning particle swarm optimization with
  diversity enhancement by ring topology* (GGL-PSOD).
- Y.-J. Gong i in., *Genetic Learning Particle Swarm Optimization* (GLPSO).
- Y. Fan, J. Shao, G. Sun, X. Shao, *A Self-Adaption Butterfly Optimization
  Algorithm for Numerical Optimization Problems*, IEEE Access 8 (2020).
- S. Arora, S. Singh, *Butterfly optimization algorithm: a novel approach for
  global optimization*, Soft Computing 23 (2019).
