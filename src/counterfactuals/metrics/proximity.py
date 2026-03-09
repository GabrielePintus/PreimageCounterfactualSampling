"""Proximity metrics for factual/counterfactual pairs."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.interfaces import MetricInterface


class L2Proximity(MetricInterface):
    """Euclidean distance between factual and counterfactual points."""

    name = "proximity_l2"

    def evaluate(self, x_orig: np.ndarray, x_cf: np.ndarray, context=None) -> float:
        del context
        x_orig = np.asarray(x_orig, dtype=np.float32)
        x_cf = np.asarray(x_cf, dtype=np.float32)
        return float(np.linalg.norm(x_cf - x_orig, ord=2))


class L1Proximity(MetricInterface):
    """L1 distance, useful for sparse tabular changes."""

    name = "proximity_l1"

    def evaluate(self, x_orig: np.ndarray, x_cf: np.ndarray, context=None) -> float:
        del context
        x_orig = np.asarray(x_orig, dtype=np.float32)
        x_cf = np.asarray(x_cf, dtype=np.float32)
        return float(np.linalg.norm(x_cf - x_orig, ord=1))
