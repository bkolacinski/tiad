"""Convergence curves (mean error vs FEs, log-y) and final-error boxplots."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

_COLORS = plt.cm.tab10.colors


def _fe_axis() -> np.ndarray:
    return np.linspace(
        config.MAX_FES / config.N_CHECKPOINTS, config.MAX_FES, config.N_CHECKPOINTS
    )


def convergence_plot(func_id: int, cells, algorithms_list) -> str:
    fe = _fe_axis()
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, algo in enumerate(algorithms_list):
        mean_curve = cells[(algo, func_id)].curves.mean(axis=0)
        ax.plot(fe, np.maximum(mean_curve, 1e-12), color=_COLORS[i], lw=2, label=algo)
    ax.set_yscale("log")
    ax.set_xlabel("Liczba ewaluacji funkcji (FEs)")
    ax.set_ylabel("Średni błąd  f(x) − f*  (log)")
    ax.set_title(f"Zbieżność — CEC2017 F{func_id} ({config.FUNCTION_CATEGORY.get(func_id,'')}, D={config.D})")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = config.CONV_DIR / f"F{func_id}.png"
    fig.savefig(out, dpi=config.PLOT_DPI)
    plt.close(fig)
    return out.name


def boxplot(func_id: int, cells, algorithms_list) -> str:
    data = [cells[(algo, func_id)].errors for algo in algorithms_list]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=algorithms_list, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], _COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Błąd końcowy  f(x) − f*  (log)")
    ax.set_title(f"Rozkład błędu — CEC2017 F{func_id} ({config.NUM_RUNS} przebiegów)")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = config.BOX_DIR / f"F{func_id}.png"
    fig.savefig(out, dpi=config.PLOT_DPI)
    plt.close(fig)
    return out.name
