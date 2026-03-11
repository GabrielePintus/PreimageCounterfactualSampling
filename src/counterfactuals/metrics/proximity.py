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


class MADWeightedL1Proximity(MetricInterface):
    """MAD-normalized L1 distance for mixed tabular data.

    Continuous features are divided by their per-feature MAD (median absolute
    deviation) computed on the training set.  Categorical features contribute
    a binary mismatch term (0 or 1).  The final score is the mean over all
    features, giving a roughly [0, 1] range and making features comparable
    regardless of their original scale.
    """

    name = "proximity_mad_l1"

    def __init__(
        self,
        mad_weights: np.ndarray,
        input_types: list[str],
        atol: float = 1e-6,
    ) -> None:
        self.mad_weights = np.asarray(mad_weights, dtype=np.float64)
        self.input_types = input_types
        self.atol = atol

    def evaluate(self, x_orig: np.ndarray, x_cf: np.ndarray, context=None) -> float:
        del context
        x_orig = np.asarray(x_orig, dtype=np.float64)
        x_cf = np.asarray(x_cf, dtype=np.float64)
        per_feature = np.empty(len(x_orig))
        for i, t in enumerate(self.input_types):
            if t == "numerical":
                per_feature[i] = abs(x_cf[i] - x_orig[i]) / self.mad_weights[i]
            else:
                per_feature[i] = float(abs(x_cf[i] - x_orig[i]) > self.atol)
        return float(per_feature.mean())
