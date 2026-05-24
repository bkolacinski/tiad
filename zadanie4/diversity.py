"""Diversity measurement and perturbation operators used by DESABOA."""
from __future__ import annotations

import numpy as np
from scipy.special import gamma


def swarm_diversity(X: np.ndarray) -> float:
    """Mean Euclidean distance of individuals to the population centroid."""
    centroid = X.mean(axis=0)
    return float(np.mean(np.linalg.norm(X - centroid, axis=1)))


def opposition(X: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Opposition-Based Learning: reflected points P̆ = lb + ub - P."""
    return lb + ub - X


def _levy_sigma(beta: float) -> float:
    """Mantegna's sigma_u for a Levy step with stability index beta."""
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2.0)
    den = gamma((1 + beta) / 2.0) * beta * (2.0 ** ((beta - 1) / 2.0))
    return (num / den) ** (1.0 / beta)


def levy_flight(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    """One Mantegna Levy step vector of length ``dim``."""
    sigma = _levy_sigma(beta)
    u = rng.normal(0.0, sigma, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    return u / (np.abs(v) ** (1.0 / beta))


def cauchy_perturb(rng: np.random.Generator, dim: int) -> np.ndarray:
    """One standard-Cauchy long-jump vector of length ``dim``."""
    return rng.standard_cauchy(size=dim)
