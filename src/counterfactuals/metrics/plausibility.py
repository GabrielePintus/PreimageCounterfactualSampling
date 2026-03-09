"""Plausibility metrics for manifold adherence."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from counterfactuals.core.interfaces import MetricInterface


class KNNPlausibility(MetricInterface):
    """Average distance to nearest train neighbors (lower is more plausible)."""

    name = "plausibility"

    def __init__(self, x_train: np.ndarray, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.nn.fit(np.asarray(x_train, dtype=np.float32))

    def evaluate(self, x_orig: np.ndarray, x_cf: np.ndarray, context=None) -> float:
        del x_orig, context
        distances, _ = self.nn.kneighbors(np.asarray(x_cf, dtype=np.float32)[None, :])
        return float(np.mean(distances))
