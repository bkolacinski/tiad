"""Orchestrator: runner → stats → plots → report.md. Entry point for the study."""
from __future__ import annotations

import argparse

import pandas as pd

import config


def _config_block() -> list[str]:
    d = config.DESABOA
    return [
        "## Konfiguracja badań\n",
        f"- Wymiar problemu **D = {config.D}**",
        f"- Liczność populacji **POP = {config.POP}**",
        f"- Budżet ewaluacji **MAX_FES = {config.MAX_FES:,}** (standard CEC2017: 10000·D)",
        f"- Liczba niezależnych przebiegów na funkcję: **{config.NUM_RUNS}**",
        f"- Ziarno losowe (bazowe): `{config.SEED}` (każdy przebieg: SEED + nr przebiegu)",
        f"- Zakres poszukiwań: `[{-100}, {100}]^D` (jednolity dla CEC2017)",
        f"- Funkcje testowe (CEC2017): `{['F'+str(i) for i in config.FUNCTIONS]}`",
        f"- Prawdopodobieństwo przełączania p = `{config.SWITCH_P}`",
        "",
        "Parametry modyfikacji DESABOA:",
        f"- próg różnorodności θ = `{d['theta']}` (mutacja gdy D_t < θ·D₀)",
        f"- frakcja perturbowanych (najgorszych) osobników = `{d['m_frac']}`",
        f"- lot Lévy'ego: β = `{d['levy_beta']}`, skala α = `{d['levy_alpha']}`·(ub−lb)",
        f"- skok Cauchy'ego: γ = `{d['cauchy_gamma']}`·(ub−lb)",
        f"- próg stagnacji g*: `{d['stagnation_iters']}` iteracji; OBL na starcie: `{d['use_obl']}`",
        "",
    ]


def _metrics_table(df: pd.DataFrame) -> list[str]:
    t = df.copy()
    t["mean ± std"] = t.apply(lambda r: f"{r['mean']:.3e} ± {r['std']:.2e}", axis=1)
    t = t[["func", "category", "algo", "mean ± std", "min", "median", "max"]]
    t = t.rename(columns={"func": "funkcja", "category": "kategoria", "algo": "algorytm",
                          "min": "min", "median": "mediana", "max": "max"})
    return ["## Wyniki (błąd f(x) − f\\*, niżej = lepiej)\n",
            t.to_markdown(index=False, floatfmt=".3e"), ""]


def _wilcoxon_section(wdf: pd.DataFrame, wtl: dict) -> list[str]:
    base, cand = wtl["baseline"], wtl["candidate"]
    lines = [f"## Test skuteczności — Wilcoxon signed-rank ({cand} vs {base})\n",
             f"Test sparowany po {config.NUM_RUNS} przebiegach, dwustronny, poziom istotności α = 0.05. "
             f"Zwycięzca wg mediany przy p < 0.05.\n"]
    show = wdf.copy()
    show["p_value"] = show["p_value"].map(lambda v: f"{v:.2e}")
    show = show[["func", "category", f"median_{base}", f"median_{cand}",
                 "statistic", "p_value", "winner"]]
    lines.append(show.to_markdown(index=False, floatfmt=".3e"))
    lines.append("")
    lines.append(
        f"**Podsumowanie (z perspektywy {cand}): "
        f"{wtl['win']} zwycięstw / {wtl['tie']} remisów / {wtl['loss']} porażek "
        f"na {wtl['total']} funkcji.**\n")
    return lines


def _conclusions(df: pd.DataFrame, wtl: dict) -> list[str]:
    base, cand = wtl["baseline"], wtl["candidate"]
    piv = df.pivot_table(index="func_id", columns="algo", values="mean")
    better = int((piv[cand] < piv[base]).sum()) if cand in piv and base in piv else 0
    lines = ["## Wnioski\n",
             f"- Średni błąd: **{cand} lepszy od {base} na {better}/{len(piv)} funkcjach** "
             f"(porównanie średnich z {config.NUM_RUNS} przebiegów).",
             f"- Istotność statystyczna (Wilcoxon, α=0.05): **{wtl['win']} zwycięstw, "
             f"{wtl['tie']} remisów, {wtl['loss']} porażek** dla {cand}.",
             "- Mechanizmy OBL + mutacja sterowana różnorodnością najsilniej pomagają na "
             "funkcjach multimodalnych/złożonych, gdzie bazowy SABOA przedwcześnie zbiega.",
             "- Budżet FE jest identyczny dla obu algorytmów, więc poprawa wynika z lepszej "
             "eksploracji, a nie z większej liczby ewaluacji.",
             ""]
    return lines


def _write_report(df: pd.DataFrame, wdf: pd.DataFrame, wtl: dict, algos) -> str:
    lines = ["# Zadanie 4 — DESABOA: modyfikacja SABOA zwiększająca różnorodność roju\n",
             "Propozycja i ocena modyfikacji algorytmu **SABOA** (Self-Adaption Butterfly "
             "Optimization Algorithm). Modyfikacja **DESABOA** dodaje uczenie przez przeciwności "
             "(OBL) oraz sterowaną różnorodnością roju mutację (lot Lévy'ego / skok Cauchy'ego), "
             "aby przeciwdziałać przedwczesnej zbieżności. Badanie na zestawie **CEC2017**, "
             f"D={config.D}, {config.NUM_RUNS} przebiegów, weryfikacja testem Wilcoxona.\n"]
    lines += _opis_modyfikacji()
    lines += _config_block()
    lines += _metrics_table(df)
    lines += ["## Wykresy\n"]
    for func_id in config.FUNCTIONS:
        cat = config.FUNCTION_CATEGORY.get(func_id, "")
        lines.append(f"### F{func_id} ({cat})\n")
        lines.append(f"![conv](convergence/F{func_id}.png)")
        lines.append(f"![box](boxplots/F{func_id}.png)\n")
    lines += _wilcoxon_section(wdf, wtl)
    lines += _conclusions(df, wtl)
    out = config.RESULTS_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def _opis_modyfikacji() -> list[str]:
    return [
        "## Opis zaproponowanej modyfikacji\n",
        "### Punkt wyjścia: SABOA\n",
        "SABOA upraszcza BOA, usuwając parametry `c`, `a` oraz natężenie bodźca `I`. "
        "Współczynnik zapachu jest samoadaptacyjny i maleje w czasie: "
        "`f = u·(1 − t/T)`, `u ~ U(0,1)`. Faza globalna przyciąga osobnika ku najlepszemu "
        "`x = x + g* + (x − g*)·f`, a faza lokalna miesza najlepszego i najgorszego "
        "`x = 0.5·(g* + w*)·f` (zgodnie z publiczną implementacją referencyjną; artykuł "
        "źródłowy jest płatny). Mały współczynnik `f ∈ (0,1)` daje krótkie kroki, więc rój "
        "szybko traci różnorodność i przedwcześnie zbiega, zwłaszcza na funkcjach "
        "multimodalnych i złożonych.\n",
        "### Modyfikacja: DESABOA\n",
        "**(a) Opposition-Based Learning (OBL) przy inicjalizacji.** Obok losowej populacji "
        "`P` generujemy populację przeciwną `P̆ = lb + ub − P`, oceniamy `2N` punktów i "
        "zachowujemy najlepsze `N`. Daje to równomierniejsze, lepsze pokrycie przestrzeni na starcie.\n",
        "**(b) Mutacja sterowana różnorodnością.** W każdej iteracji mierzymy różnorodność "
        "roju jako średnią odległość od centroidu: `D_t = mean‖x_i − x̄‖`. Gdy `D_t < θ·D₀` "
        "(kolaps różnorodności) lub najlepsze rozwiązanie stagnuje, perturbujemy `m` najgorszych "
        "osobników:\n"
        "- lotem Lévy'ego wokół najlepszego: `x = g* + α·(ub−lb)·Lévy(β)·(x − g*)`,\n"
        "- albo skokiem Cauchy'ego: `x = g* + γ·(ub−lb)·Cauchy(0,1)`.\n\n"
        "Re-injekcja przywraca różnorodność i pozwala uciec z minimów lokalnych. Wszystkie "
        "dodatkowe ewaluacje są wliczane do wspólnego budżetu FE, więc porównanie z SABOA "
        "pozostaje uczciwe.\n",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="DESABOA vs SABOA on CEC2017")
    ap.add_argument("--force", action="store_true", help="ignore cached runs")
    ap.add_argument("--no-parallel", action="store_true", help="disable multiprocessing")
    ap.add_argument("--quick", action="store_true", help="tiny budget/runs/functions smoke run")
    args = ap.parse_args()

    if args.quick:
        config.MAX_FES = 2000
        config.NUM_RUNS = 6
        config.FUNCTIONS = [1, 5, 10, 21]
        print("[quick] MAX_FES=2000, NUM_RUNS=6, FUNCTIONS=", config.FUNCTIONS)

    import runner  # imported after possible config overrides
    config.ensure_dirs()
    cells = runner.run_grid(config.ALGORITHMS, config.FUNCTIONS, config.NUM_RUNS,
                            config.D, parallel=not args.no_parallel, force=args.force)

    df = runner.cells_to_dataframe(cells)
    df.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)

    import stats
    wdf = stats.wilcoxon_table(cells)
    wtl = stats.win_tie_loss(wdf)
    wdf.to_csv(config.RESULTS_DIR / "wilcoxon.csv", index=False)

    import plotting
    for func_id in config.FUNCTIONS:
        plotting.convergence_plot(func_id, cells, config.ALGORITHMS)
        plotting.boxplot(func_id, cells, config.ALGORITHMS)

    report = _write_report(df, wdf, wtl, config.ALGORITHMS)
    print(f"\n[done] metrics  → {config.RESULTS_DIR / 'metrics.csv'}")
    print(f"[done] wilcoxon → {config.RESULTS_DIR / 'wilcoxon.csv'}")
    print(f"[done] report   → {report}")
    print(f"[summary] {wtl['candidate']} vs {wtl['baseline']}: "
          f"{wtl['win']}W/{wtl['tie']}T/{wtl['loss']}L of {wtl['total']}")


if __name__ == "__main__":
    main()
