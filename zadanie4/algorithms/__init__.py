"""Optimizer implementations: BOA, SABOA, and the proposed DESABOA."""
from .base import BaseOptimizer, OptResult
from .boa import BOA
from .saboa import SABOA
from .desaboa import DESABOA

REGISTRY = {"BOA": BOA, "SABOA": SABOA, "DESABOA": DESABOA}


def get(name: str) -> type[BaseOptimizer]:
    return REGISTRY[name]
