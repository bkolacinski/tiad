"""DESABOA — Diversity-Enhanced SABOA (proposed modification).

Two mechanisms are added to SABOA to fight premature convergence:
  (a) Opposition-Based Learning (OBL) at initialization — wider, better-spread start;
  (b) diversity-driven re-injection — when swarm diversity collapses below
      theta * D0 (or the global best stagnates), the worst m individuals are
      perturbed with a Levy flight around the best or a Cauchy long jump.
All extra evaluations are counted against the shared FE budget.
"""
from __future__ import annotations

import numpy as np

import config
import diversity as dv
from .base import BaseOptimizer, OptResult


class DESABOA(BaseOptimizer):
    name = "DESABOA"

    def optimize(self) -> OptResult:
        rng, p = self.rng, config.SWITCH_P
        cfg = config.DESABOA
        span = self.ub - self.lb

        # (a) OBL initialization: keep best POP of {P, opposite(P)}
        X = self._init_population()
        if cfg["use_obl"]:
            Xo = self._clip(dv.opposition(X, self.lb, self.ub))
            both = np.vstack([X, Xo])
            both_fit = self.problem.evaluate_pop(both)
            order = np.argsort(both_fit)[: self.pop]
            X, fit = both[order], both_fit[order]
        else:
            fit = self.problem.evaluate_pop(X)
        self._track(fit, X)

        D0 = max(dv.swarm_diversity(X), 1e-12)
        m = max(1, round(cfg["m_frac"] * self.pop))
        max_iter = max(1, (self.max_fes - self.pop) // self.pop)
        t, stagn = 0, 0
        prev_best = self.best_error

        while self.problem.n_fe + self.pop <= self.max_fes:
            g = self.best_x
            w = X[int(np.argmax(fit))]
            decay = 1.0 - t / max_iter
            f = (rng.random((self.pop, 1)) * decay)
            go_global = rng.random(self.pop) > p

            new = X.copy()
            gi = np.where(go_global)[0]
            new[gi] = X[gi] + g + (X[gi] - g) * f[gi]   # SABOA global (reference form)
            li = np.where(~go_global)[0]
            if li.size:
                new[li] = 0.5 * (g + w) * f[li]

            new = self._clip(new)
            new_fit = self.problem.evaluate_pop(new)
            improved = new_fit < fit
            X[improved] = new[improved]
            fit[improved] = new_fit[improved]
            self._track(fit, X)

            # (b) diversity-driven re-injection
            stagn = stagn + 1 if self.best_error >= prev_best - 1e-12 else 0
            prev_best = self.best_error
            collapsed = dv.swarm_diversity(X) < cfg["theta"] * D0
            if (collapsed or stagn >= cfg["stagnation_iters"]) and self.problem.n_fe + m <= self.max_fes:
                worst = np.argsort(fit)[-m:]
                g = self.best_x
                mut = X[worst].copy()
                for idx, row in enumerate(worst):
                    if rng.random() < cfg["mut_mix"]:
                        step = dv.levy_flight(rng, self.dim, cfg["levy_beta"])
                        mut[idx] = g + cfg["levy_alpha"] * span * step * (X[row] - g)
                    else:
                        step = dv.cauchy_perturb(rng, self.dim)
                        mut[idx] = g + cfg["cauchy_gamma"] * span * step
                mut = self._clip(mut)
                mut_fit = self.problem.evaluate_pop(mut)
                X[worst], fit[worst] = mut, mut_fit   # accept to restore diversity
                self._track(fit, X)
                stagn = 0

            t += 1
        return self._finalize()
