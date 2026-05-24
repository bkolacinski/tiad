"""Wilcoxon signed-rank test (baseline vs modified) per function + win/tie/loss."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config


def wilcoxon_table(cells, baseline=None, candidate=None) -> pd.DataFrame:
    """Paired Wilcoxon per function: baseline errors vs candidate errors (51 runs)."""
    baseline = baseline or config.BASELINE
    candidate = candidate or next(a for a in config.ALGORITHMS if a != baseline)

    rows = []
    for func_id in config.FUNCTIONS:
        b = cells[(baseline, func_id)].errors
        c = cells[(candidate, func_id)].errors
        med_b, med_c = float(np.median(b)), float(np.median(c))

        diff = b - c
        if np.allclose(diff, 0.0):
            stat, p = np.nan, 1.0          # identical samples -> no difference
        else:
            stat, p = wilcoxon(b, c, alternative="two-sided", zero_method="wilcox")

        if not np.isnan(p) and p < 0.05:
            winner = candidate if med_c < med_b else baseline
        else:
            winner = "tie"
        rows.append({
            "func": f"F{func_id}",
            "func_id": func_id,
            "category": config.FUNCTION_CATEGORY.get(func_id, ""),
            f"median_{baseline}": med_b,
            f"median_{candidate}": med_c,
            "statistic": float(stat) if not np.isnan(stat) else np.nan,
            "p_value": float(p),
            "winner": winner,
        })
    df = pd.DataFrame(rows).sort_values("func_id").reset_index(drop=True)
    df.attrs["baseline"] = baseline
    df.attrs["candidate"] = candidate
    return df


def win_tie_loss(wilcox_df: pd.DataFrame) -> dict:
    """Summary from the candidate's perspective (+/=/-)."""
    candidate = wilcox_df.attrs["candidate"]
    baseline = wilcox_df.attrs["baseline"]
    w = int((wilcox_df["winner"] == candidate).sum())
    loss = int((wilcox_df["winner"] == baseline).sum())
    tie = int((wilcox_df["winner"] == "tie").sum())
    return {"candidate": candidate, "baseline": baseline,
            "win": w, "tie": tie, "loss": loss, "total": len(wilcox_df)}
