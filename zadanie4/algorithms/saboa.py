"""SABOA — Self-Adaption Butterfly Optimization Algorithm (Fan et al., IEEE Access 2020).

Removes BOA's c, a and stimulus intensity I; the only parameter is the switch p.
- self-adaptive, time-decaying fragrance:  f = u * (1 - t/T),  u ~ U(0,1)
- global phase (toward best):  x = x + g* + (x - g*) * f
- local  phase (best & worst): x = 0.5 * (g* + w*) * f

Note: the SABOA paper (Fan et al. 2020) is paywalled; we follow the public reference
implementation (amineHorseman/butterfly-optimization-algorithms), whose global update
``x = x + g* + (x - g*)*f`` overshoots the [x, g*] segment and so retains some
exploration. A pure contraction ``x = g* + (x-g*)*f`` instead freezes the swarm on the
initial best (it never beats g*), which is not how SABOA behaves.
"""
from __future__ import annotations

import numpy as np

import config
from .base import BaseOptimizer, OptResult


class SABOA(BaseOptimizer):
    name = "SABOA"

    def optimize(self) -> OptResult:
        rng, p = self.rng, config.SWITCH_P
        X = self._init_population()
        fit = self.problem.evaluate_pop(X)
        self._track(fit, X)

        max_iter = max(1, (self.max_fes - self.pop) // self.pop)
        t = 0
        while self.problem.n_fe + self.pop <= self.max_fes:
            g = self.best_x
            w = X[int(np.argmax(fit))]                  # current global worst
            decay = 1.0 - t / max_iter
            f = (rng.random((self.pop, 1)) * decay)     # self-adaptive fragrance
            go_global = rng.random(self.pop) > p

            new = X.copy()
            gi = np.where(go_global)[0]
            new[gi] = X[gi] + g + (X[gi] - g) * f[gi]   # global: toward best (reference form)
            li = np.where(~go_global)[0]
            if li.size:
                new[li] = 0.5 * (g + w) * f[li]          # local: best & worst midpoint

            new = self._clip(new)
            new_fit = self.problem.evaluate_pop(new)
            improved = new_fit < fit
            X[improved] = new[improved]
            fit[improved] = new_fit[improved]
            self._track(fit, X)

            t += 1
        return self._finalize()
