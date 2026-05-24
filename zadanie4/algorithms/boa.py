"""Butterfly Optimization Algorithm (Arora & Singh, Soft Computing 2019).

Faithful to Arora's reference MATLAB code where it diverges from the paper:
- two independent randoms r1*r2 (paper writes r^2),
- ``rand > p`` selects the GLOBAL phase,
- the sensory modality ``c`` (not ``a``) is updated each iteration.
"""
from __future__ import annotations

import numpy as np

import config
from .base import BaseOptimizer, OptResult


class BOA(BaseOptimizer):
    name = "BOA"

    def optimize(self) -> OptResult:
        rng, p = self.rng, config.SWITCH_P
        a, c = config.BOA_A, config.BOA_C0

        X = self._init_population()
        fit = self.problem.evaluate_pop(X)
        self._track(fit, X)

        # MaxIter implied by the FE budget (init + pop FEs/iter)
        max_iter = max(1, (self.max_fes - self.pop) // self.pop)
        t = 0
        while self.problem.n_fe + self.pop <= self.max_fes:
            g = self.best_x
            # stimulus intensity = fitness magnitude; |.|^a keeps fragrance real
            f = (c * np.abs(fit) ** a).reshape(-1, 1)
            r1 = rng.random((self.pop, 1))
            r2 = rng.random((self.pop, 1))
            go_global = rng.random(self.pop) > p

            new = X.copy()
            # global phase: move toward best
            gi = np.where(go_global)[0]
            new[gi] = X[gi] + (r1[gi] * r2[gi] * g - X[gi]) * f[gi]
            # local phase: move along two random butterflies
            li = np.where(~go_global)[0]
            if li.size:
                j = rng.integers(0, self.pop, li.size)
                k = rng.integers(0, self.pop, li.size)
                new[li] = X[li] + (r1[li] * r2[li] * X[j] - X[k]) * f[li]

            new = self._clip(new)
            new_fit = self.problem.evaluate_pop(new)
            improved = new_fit < fit          # greedy acceptance
            X[improved] = new[improved]
            fit[improved] = new_fit[improved]
            self._track(fit, X)

            t += 1
            c = c + 0.025 / (c * max_iter)
        return self._finalize()
