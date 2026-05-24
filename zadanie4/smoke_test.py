"""Fast sanity check: tiny budget, few runs — verifies the pipeline end to end."""
from __future__ import annotations

import numpy as np

import algorithms
import benchmark
import config


def main() -> None:
    config.MAX_FES = 1500
    config.N_CHECKPOINTS = 50

    for func_id in (1, 21):
        for algo_name in config.ALGORITHMS:
            prob = benchmark.make(func_id, config.D)
            rng = np.random.default_rng(config.SEED)
            res = algorithms.get(algo_name)(prob, rng).optimize()
            curve = res.convergence
            assert prob.n_fe <= config.MAX_FES, f"{algo_name} overran budget: {prob.n_fe}"
            assert res.best_error >= 0, "CEC error must be >= 0"
            assert np.all(np.diff(curve) <= 1e-9), "convergence curve must be non-increasing"
            assert len(curve) == config.N_CHECKPOINTS
            print(f"[ok] {algo_name:8s} F{func_id}: best={res.best_error:.4e} "
                  f"FEs={prob.n_fe} curve[0]={curve[0]:.2e} curve[-1]={curve[-1]:.2e}")

    # determinism: same seed -> same result
    p1 = benchmark.make(5, config.D)
    p2 = benchmark.make(5, config.D)
    r1 = algorithms.DESABOA(p1, np.random.default_rng(7)).optimize().best_error
    r2 = algorithms.DESABOA(p2, np.random.default_rng(7)).optimize().best_error
    assert r1 == r2, f"non-deterministic: {r1} != {r2}"
    print(f"[ok] determinism: DESABOA F5 reproducible ({r1:.4e})")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
