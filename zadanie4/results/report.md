# Zadanie 4 — DESABOA: modyfikacja SABOA zwiększająca różnorodność roju

Propozycja i ocena modyfikacji algorytmu **SABOA** (Self-Adaption Butterfly Optimization Algorithm). Modyfikacja **DESABOA** dodaje uczenie przez przeciwności (OBL) oraz sterowaną różnorodnością roju mutację (lot Lévy'ego / skok Cauchy'ego), aby przeciwdziałać przedwczesnej zbieżności. Badanie na zestawie **CEC2017**, D=10, 51 przebiegów, weryfikacja testem Wilcoxona.

## Opis zaproponowanej modyfikacji

### Punkt wyjścia: SABOA

SABOA upraszcza BOA, usuwając parametry `c`, `a` oraz natężenie bodźca `I`. Współczynnik zapachu jest samoadaptacyjny i maleje w czasie: `f = u·(1 − t/T)`, `u ~ U(0,1)`. Faza globalna przyciąga osobnika ku najlepszemu `x = x + g* + (x − g*)·f`, a faza lokalna miesza najlepszego i najgorszego `x = 0.5·(g* + w*)·f` (zgodnie z publiczną implementacją referencyjną; artykuł źródłowy jest płatny). Mały współczynnik `f ∈ (0,1)` daje krótkie kroki, więc rój szybko traci różnorodność i przedwcześnie zbiega, zwłaszcza na funkcjach multimodalnych i złożonych.

### Modyfikacja: DESABOA

**(a) Opposition-Based Learning (OBL) przy inicjalizacji.** Obok losowej populacji `P` generujemy populację przeciwną `P̆ = lb + ub − P`, oceniamy `2N` punktów i zachowujemy najlepsze `N`. Daje to równomierniejsze, lepsze pokrycie przestrzeni na starcie.

**(b) Mutacja sterowana różnorodnością.** W każdej iteracji mierzymy różnorodność roju jako średnią odległość od centroidu: `D_t = mean‖x_i − x̄‖`. Gdy `D_t < θ·D₀` (kolaps różnorodności) lub najlepsze rozwiązanie stagnuje, perturbujemy `m` najgorszych osobników:
- lotem Lévy'ego wokół najlepszego: `x = g* + α·(ub−lb)·Lévy(β)·(x − g*)`,
- albo skokiem Cauchy'ego: `x = g* + γ·(ub−lb)·Cauchy(0,1)`.

Re-injekcja przywraca różnorodność i pozwala uciec z minimów lokalnych. Wszystkie dodatkowe ewaluacje są wliczane do wspólnego budżetu FE, więc porównanie z SABOA pozostaje uczciwe.

## Konfiguracja badań

- Wymiar problemu **D = 10**
- Liczność populacji **POP = 30**
- Budżet ewaluacji **MAX_FES = 100,000** (standard CEC2017: 10000·D)
- Liczba niezależnych przebiegów na funkcję: **51**
- Ziarno losowe (bazowe): `42` (każdy przebieg: SEED + nr przebiegu)
- Zakres poszukiwań: `[-100, 100]^D` (jednolity dla CEC2017)
- Funkcje testowe (CEC2017): `['F1', 'F4', 'F5', 'F9', 'F10', 'F14', 'F17', 'F20', 'F21', 'F24', 'F27', 'F29']`
- Prawdopodobieństwo przełączania p = `0.8`

Parametry modyfikacji DESABOA:
- próg różnorodności θ = `0.1` (mutacja gdy D_t < θ·D₀)
- frakcja perturbowanych (najgorszych) osobników = `0.2`
- lot Lévy'ego: β = `1.5`, skala α = `0.01`·(ub−lb)
- skok Cauchy'ego: γ = `0.1`·(ub−lb)
- próg stagnacji g*: `5` iteracji; OBL na starcie: `True`

## Wyniki (błąd f(x) − f\*, niżej = lepiej)

| funkcja   | kategoria    | algorytm   | mean ± std           |       min |   mediana |       max |
|:----------|:-------------|:-----------|:---------------------|----------:|----------:|----------:|
| F1        | unimodalna   | DESABOA    | 4.861e+07 ± 2.49e+07 | 5.129e+06 | 4.611e+07 | 1.212e+08 |
| F1        | unimodalna   | SABOA      | 1.878e+10 ± 6.43e+09 | 6.508e+09 | 1.743e+10 | 2.998e+10 |
| F4        | multimodalna | DESABOA    | 1.335e+02 ± 3.29e+01 | 7.974e+01 | 1.251e+02 | 2.480e+02 |
| F4        | multimodalna | SABOA      | 1.989e+04 ± 2.88e+03 | 5.693e+03 | 2.146e+04 | 2.155e+04 |
| F5        | multimodalna | DESABOA    | 8.077e-05 ± 1.51e-04 | 9.959e-08 | 2.321e-05 | 7.812e-04 |
| F5        | multimodalna | SABOA      | 1.801e-02 ± 1.18e-02 | 5.170e-04 | 1.451e-02 | 5.184e-02 |
| F9        | multimodalna | DESABOA    | 8.639e+02 ± 2.84e+02 | 1.268e+02 | 8.513e+02 | 1.622e+03 |
| F9        | multimodalna | SABOA      | 2.168e+03 ± 2.76e+02 | 1.410e+03 | 2.112e+03 | 2.822e+03 |
| F10       | hybrydowa    | DESABOA    | 1.987e+04 ± 1.11e+04 | 5.362e+03 | 1.702e+04 | 5.529e+04 |
| F10       | hybrydowa    | SABOA      | 1.499e+08 ± 1.71e+08 | 5.430e+05 | 5.718e+07 | 4.515e+08 |
| F14       | hybrydowa    | DESABOA    | 2.512e+05 ± 2.59e+05 | 1.146e+04 | 1.329e+05 | 9.984e+05 |
| F14       | hybrydowa    | SABOA      | 6.796e+07 ± 8.96e+07 | 1.388e+05 | 2.143e+07 | 3.993e+08 |
| F17       | hybrydowa    | DESABOA    | 4.803e+04 ± 2.46e+04 | 6.735e+03 | 4.389e+04 | 1.472e+05 |
| F17       | hybrydowa    | SABOA      | 1.308e+08 ± 2.25e+08 | 8.497e+04 | 3.292e+07 | 1.153e+09 |
| F20       | złożona      | DESABOA    | 2.303e+02 ± 7.31e+01 | 5.859e+01 | 2.121e+02 | 4.195e+02 |
| F20       | złożona      | SABOA      | 1.020e+04 ± 5.82e+03 | 2.052e+03 | 9.380e+03 | 1.921e+04 |
| F21       | złożona      | DESABOA    | 1.137e+02 ± 1.91e+00 | 1.098e+02 | 1.133e+02 | 1.201e+02 |
| F21       | złożona      | SABOA      | 2.833e+02 ± 1.77e+02 | 1.259e+02 | 2.228e+02 | 9.670e+02 |
| F24       | złożona      | DESABOA    | 5.047e+02 ± 2.47e+01 | 4.816e+02 | 4.946e+02 | 6.156e+02 |
| F24       | złożona      | SABOA      | 1.428e+03 ± 3.22e+02 | 7.137e+02 | 1.386e+03 | 1.862e+03 |
| F27       | złożona      | DESABOA    | 1.632e+02 ± 1.08e+02 | 5.858e+01 | 1.283e+02 | 4.593e+02 |
| F27       | złożona      | SABOA      | 9.302e+02 ± 3.84e+02 | 2.664e+02 | 8.471e+02 | 2.404e+03 |
| F29       | złożona      | DESABOA    | 4.066e+06 ± 6.28e+06 | 1.482e+04 | 1.691e+06 | 2.941e+07 |
| F29       | złożona      | SABOA      | 4.105e+09 ± 5.12e+09 | 4.920e+05 | 2.020e+09 | 2.257e+10 |

## Wykresy

### F1 (unimodalna)

![conv](convergence/F1.png)
![box](boxplots/F1.png)

### F4 (multimodalna)

![conv](convergence/F4.png)
![box](boxplots/F4.png)

### F5 (multimodalna)

![conv](convergence/F5.png)
![box](boxplots/F5.png)

### F9 (multimodalna)

![conv](convergence/F9.png)
![box](boxplots/F9.png)

### F10 (hybrydowa)

![conv](convergence/F10.png)
![box](boxplots/F10.png)

### F14 (hybrydowa)

![conv](convergence/F14.png)
![box](boxplots/F14.png)

### F17 (hybrydowa)

![conv](convergence/F17.png)
![box](boxplots/F17.png)

### F20 (złożona)

![conv](convergence/F20.png)
![box](boxplots/F20.png)

### F21 (złożona)

![conv](convergence/F21.png)
![box](boxplots/F21.png)

### F24 (złożona)

![conv](convergence/F24.png)
![box](boxplots/F24.png)

### F27 (złożona)

![conv](convergence/F27.png)
![box](boxplots/F27.png)

### F29 (złożona)

![conv](convergence/F29.png)
![box](boxplots/F29.png)

## Test skuteczności — Wilcoxon signed-rank (DESABOA vs SABOA)

Test sparowany po 51 przebiegach, dwustronny, poziom istotności α = 0.05. Zwycięzca wg mediany przy p < 0.05.

| func   | category     |   median_SABOA |   median_DESABOA |   statistic |   p_value | winner   |
|:-------|:-------------|---------------:|-----------------:|------------:|----------:|:---------|
| F1     | unimodalna   |      1.743e+10 |        4.611e+07 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F4     | multimodalna |      2.146e+04 |        1.251e+02 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F5     | multimodalna |      1.451e-02 |        2.321e-05 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F9     | multimodalna |      2.112e+03 |        8.513e+02 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F10    | hybrydowa    |      5.718e+07 |        1.702e+04 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F14    | hybrydowa    |      2.143e+07 |        1.329e+05 |   1.000e+00 | 5.460e-10 | DESABOA  |
| F17    | hybrydowa    |      3.292e+07 |        4.389e+04 |   1.000e+00 | 5.460e-10 | DESABOA  |
| F20    | złożona      |      9.380e+03 |        2.121e+02 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F21    | złożona      |      2.228e+02 |        1.133e+02 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F24    | złożona      |      1.386e+03 |        4.946e+02 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F27    | złożona      |      8.471e+02 |        1.283e+02 |   0.000e+00 | 5.150e-10 | DESABOA  |
| F29    | złożona      |      2.020e+09 |        1.691e+06 |   0.000e+00 | 5.150e-10 | DESABOA  |

**Podsumowanie (z perspektywy DESABOA): 12 zwycięstw / 0 remisów / 0 porażek na 12 funkcji.**

## Wnioski

- Średni błąd: **DESABOA lepszy od SABOA na 12/12 funkcjach** (porównanie średnich z 51 przebiegów).
- Istotność statystyczna (Wilcoxon, α=0.05): **12 zwycięstw, 0 remisów, 0 porażek** dla DESABOA.
- Mechanizmy OBL + mutacja sterowana różnorodnością najsilniej pomagają na funkcjach multimodalnych/złożonych, gdzie bazowy SABOA przedwcześnie zbiega.
- Budżet FE jest identyczny dla obu algorytmów, więc poprawa wynika z lepszej eksploracji, a nie z większej liczby ewaluacji.
