"""Grid runner: ALGORITHMS × FUNCTIONS × NUM_RUNS, parallelized, cached to disk.

Each (algorithm, function) cell is cached to results/runs/{algo}_F{id}.npz the moment
its runs finish, so re-running skips completed cells and a crash loses only the
in-flight cell. Determinism: every run uses np.random.default_rng(SEED + run_idx).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import algorithms
import benchmark
import config


@dataclass
class Cell:
    algo: str
    func_id: int
    errors: np.ndarray      # (n_runs,)
    curves: np.ndarray      # (n_runs, N_CHECKPOINTS)


def _run_single(task: tuple[str, int, int, int]) -> tuple[float, np.ndarray]:
    """One independent run; defined at module top level for multiprocessing."""
    algo_name, func_id, ndim, run_idx = task
    prob = benchmark.make(func_id, ndim)
    rng = np.random.default_rng(config.SEED + run_idx)
    opt = algorithms.get(algo_name)(prob, rng)
    res = opt.optimize()
    return res.best_error, res.convergence


def _cache_path(algo: str, func_id: int):
    return config.RUNS_DIR / f"{algo}_F{func_id}.npz"


def run_cell(algo: str, func_id: int, n_runs: int, ndim: int, pool, force: bool) -> Cell:
    path = _cache_path(algo, func_id)
    if path.exists() and not force:
        d = np.load(path)
        if len(d["errors"]) >= n_runs:
            print(f"[skip] {algo} F{func_id} (cached, {len(d['errors'])} runs)")
            return Cell(algo, func_id, d["errors"][:n_runs], d["curves"][:n_runs])

    tasks = [(algo, func_id, ndim, r) for r in range(n_runs)]
    mapper = pool.map if pool is not None else map
    results = list(mapper(_run_single, tasks))
    errors = np.array([e for e, _ in results])
    curves = np.vstack([c for _, c in results])
    np.savez(path, errors=errors, curves=curves)
    print(f"[done] {algo} F{func_id}: mean={errors.mean():.3e} std={errors.std():.3e}")
    return Cell(algo, func_id, errors, curves)


def run_grid(algorithms_list, functions, n_runs, ndim, parallel=True, force=False):
    config.ensure_dirs()
    pool = None
    if parallel:
        import multiprocessing as mp
        pool = mp.Pool(min(os.cpu_count() or 1, n_runs))
    cells: dict[tuple[str, int], Cell] = {}
    try:
        for func_id in functions:
            for algo in algorithms_list:
                cells[(algo, func_id)] = run_cell(algo, func_id, n_runs, ndim, pool, force)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return cells


def cells_to_dataframe(cells: dict[tuple[str, int], Cell]) -> pd.DataFrame:
    rows = []
    for (algo, func_id), cell in cells.items():
        e = cell.errors
        rows.append({
            "func": f"F{func_id}",
            "func_id": func_id,
            "category": config.FUNCTION_CATEGORY.get(func_id, ""),
            "algo": algo,
            "mean": e.mean(), "std": e.std(),
            "min": e.min(), "max": e.max(), "median": np.median(e),
        })
    return pd.DataFrame(rows).sort_values(["func_id", "algo"]).reset_index(drop=True)
