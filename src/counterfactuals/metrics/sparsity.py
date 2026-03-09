"""Sparsity metrics for counterfactual explanations."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.interfaces import MetricInterface


class SparsityMetric(MetricInterface):
    """Fraction of changed features (lower is better)."""

    name = "sparsity"

    def __init__(self, atol: float = 1e-6):
        self.atol = atol

    def evaluate(self, x_orig: np.ndarray, x_cf: np.ndarray, context=None) -> float:
        del context
        x_orig = np.asarray(x_orig, dtype=np.float32)
        x_cf = np.asarray(x_cf, dtype=np.float32)
        changed = np.abs(x_cf - x_orig) > self.atol
        return float(np.mean(changed))
