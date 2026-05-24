"""Generate LaTeX table fragments + copy figures for sprawozdanie.tex.

Reads results/metrics.csv and results/wilcoxon.csv (exact, reproducible numbers),
emits sprawozdanie/tab_wyniki.tex and sprawozdanie/tab_wilcoxon.tex, and copies the
per-function PNGs into sprawozdanie/figury/. Run after compare.py.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import config

SPR = config.BASE / "sprawozdanie"
FIG = SPR / "figury"


def _sci(x: float, sig: int = 2) -> str:
    """Format as LaTeX scientific math, e.g. 1.88\\cdot10^{10}."""
    if x == 0 or not np.isfinite(x):
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10.0 ** exp)
    if exp == 0:
        return f"{mant:.{sig}f}"
    return f"{mant:.{sig}f}\\cdot10^{{{exp}}}"


def _table_wyniki(df: pd.DataFrame) -> str:
    piv = {(r.algo, r.func_id): r for r in df.itertuples()}
    lines = [
        "\\begin{table}[H]", "\\centering",
        "\\caption{Uśrednione wartości błędu $f(x)-f^*$ po "
        f"{config.NUM_RUNS} przebiegach (D={config.D}). Niżej = lepiej; "
        "wynik lepszy pogrubiono.}",
        "\\label{tab:wyniki}",
        "\\begin{tabular}{llrr}", "\\toprule",
        "\\textbf{Funkcja} & \\textbf{Kategoria} & "
        "\\textbf{SABOA (śr. $\\pm$ odch.)} & \\textbf{DESABOA (śr. $\\pm$ odch.)} \\\\",
        "\\midrule",
    ]
    for fid in config.FUNCTIONS:
        s, d = piv[("SABOA", fid)], piv[("DESABOA", fid)]
        cat = config.FUNCTION_CATEGORY.get(fid, "")
        s_cell = f"${_sci(s.mean)} \\pm {_sci(s.std)}$"
        d_cell = f"$\\mathbf{{{_sci(d.mean)} \\pm {_sci(d.std)}}}$"
        lines.append(f"F{fid} & {cat} & {s_cell} & {d_cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _table_wilcoxon(wdf: pd.DataFrame) -> str:
    base, cand = "SABOA", "DESABOA"
    lines = [
        "\\begin{table}[H]", "\\centering",
        "\\caption{Test sumy rang Wilcoxona (sparowany, dwustronny, $\\alpha=0{,}05$) "
        f"dla {cand} względem {base}. Zwycięzca wg mediany przy $p<0{{,}}05$.}}",
        "\\label{tab:wilcoxon}",
        "\\begin{tabular}{lrrrl}", "\\toprule",
        "\\textbf{Funkcja} & \\textbf{med. SABOA} & \\textbf{med. DESABOA} & "
        "\\textbf{$p$} & \\textbf{Zwycięzca} \\\\", "\\midrule",
    ]
    for r in wdf.itertuples():
        med_b = getattr(r, f"median_{base}")
        med_c = getattr(r, f"median_{cand}")
        lines.append(
            f"{r.func} & ${_sci(med_b)}$ & ${_sci(med_c)}$ & ${_sci(r.p_value)}$ & {r.winner} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
    wdf = pd.read_csv(config.RESULTS_DIR / "wilcoxon.csv")

    (SPR / "tab_wyniki.tex").write_text(_table_wyniki(df), encoding="utf-8")
    (SPR / "tab_wilcoxon.tex").write_text(_table_wilcoxon(wdf), encoding="utf-8")

    for fid in config.FUNCTIONS:
        for kind in ("convergence", "boxplots"):
            src = config.RESULTS_DIR / kind / f"F{fid}.png"
            if src.exists():
                shutil.copy(src, FIG / f"{kind}_F{fid}.png")

    print(f"[done] tables → {SPR}/tab_wyniki.tex, tab_wilcoxon.tex")
    print(f"[done] figures → {FIG} ({len(list(FIG.glob('*.png')))} plików)")


if __name__ == "__main__":
    main()
