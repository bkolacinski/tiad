"""Shared optimizer scaffolding: population init, clipping, FE-checkpoint tracking.

Convergence is sampled at fixed FE checkpoints (not per-iteration) so that
algorithms with different per-iteration FE counts produce equal-length,
directly-averageable curves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass
class OptResult:
    best_x: np.ndarray
    best_error: float
    convergence: np.ndarray  # length N_CHECKPOINTS, best-so-far error vs FE


class BaseOptimizer:
    name = "Base"

    def __init__(self, problem, rng: np.random.Generator):
        self.problem = problem
        self.rng = rng
        self.lb = problem.lb
        self.ub = problem.ub
        self.dim = problem.ndim
        self.pop = config.POP
        self.max_fes = config.MAX_FES

        # FE checkpoints at which to record best-so-far error
        self._checkpoints = np.linspace(
            self.max_fes / config.N_CHECKPOINTS, self.max_fes, config.N_CHECKPOINTS
        )
        self._curve = np.full(config.N_CHECKPOINTS, np.nan)
        self._next_cp = 0
        self.best_error = np.inf
        self.best_x = None

    # -- helpers ------------------------------------------------------------
    def _init_population(self) -> np.ndarray:
        return self.lb + self.rng.random((self.pop, self.dim)) * (self.ub - self.lb)

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lb, self.ub)

    def _track(self, fitness: np.ndarray, X: np.ndarray) -> None:
        """Update global best and fill any checkpoints passed by the FE counter."""
        i = int(np.argmin(fitness))
        if fitness[i] < self.best_error:
            self.best_error = float(fitness[i])
            self.best_x = X[i].copy()
        fe = self.problem.n_fe
        while self._next_cp < len(self._checkpoints) and fe >= self._checkpoints[self._next_cp]:
            self._curve[self._next_cp] = self.best_error
            self._next_cp += 1

    def _finalize(self) -> OptResult:
        # forward-fill any unreached checkpoints with the final best
        for k in range(config.N_CHECKPOINTS):
            if np.isnan(self._curve[k]):
                self._curve[k] = self.best_error
        return OptResult(self.best_x, self.best_error, self._curve)

    # -- interface ----------------------------------------------------------
    def optimize(self) -> OptResult:
        raise NotImplementedError
