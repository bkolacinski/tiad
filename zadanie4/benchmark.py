"""CEC2017 wrapper with an explicit FE counter, returning error = f(x) - f_global.

The FE counter is the fairness guarantee: every algorithm shares the same MAX_FES
budget and the optimization loop is driven by ``while problem.n_fe < MAX_FES``.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")  # silence opfunu's latex SyntaxWarnings on import


def _cec2017_class(func_id: int):
    import opfunu.cec_based.cec2017 as m
    cls = getattr(m, f"F{func_id}2017", None)
    if cls is None:
        raise ValueError(f"CEC2017 F{func_id} not available in opfunu")
    return cls


class Benchmark:
    """One CEC2017 function instance with a private FE counter."""

    def __init__(self, func_id: int, ndim: int):
        self.func_id = func_id
        self.ndim = ndim
        self._func = _cec2017_class(func_id)(ndim=ndim)
        self.lb = np.asarray(self._func.lb, dtype=float)
        self.ub = np.asarray(self._func.ub, dtype=float)
        self.f_global = float(self._func.f_global)
        self.n_fe = 0

    def reset(self) -> None:
        self.n_fe = 0

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate one solution, count one FE, return error >= 0 (ideal = 0)."""
        self.n_fe += 1
        return float(self._func.evaluate(np.asarray(x, dtype=float))) - self.f_global

    def evaluate_pop(self, X: np.ndarray) -> np.ndarray:
        """Evaluate a (N, D) population; counts N FEs."""
        return np.array([self.evaluate(row) for row in X])

    @property
    def budget_left(self) -> int:
        from config import MAX_FES
        return MAX_FES - self.n_fe


def make(func_id: int, ndim: int) -> Benchmark:
    return Benchmark(func_id, ndim)
