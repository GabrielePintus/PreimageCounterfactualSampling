from __future__ import annotations

from math import exp, lgamma, log, pi

import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors, radius_neighbors_graph

from .base import BaseDensityEstimator


def unit_ball_volume(d: int) -> float:
    """Volume of the unit ball in R^d."""
    log_volume = log_unit_ball_volume(d)
    if log_volume < np.log(np.finfo(np.float64).tiny):
        return 0.0
    return exp(log_volume)


def log_unit_ball_volume(d: int) -> float:
    """Log-volume of the unit ball in R^d, stable in high dimension."""
    return (d / 2.0) * log(pi) - lgamma(d / 2.0 + 1.0)


class KDEEstimator(BaseDensityEstimator):
    """KDE density estimator (Eq. 2 in the paper).

    Graph connectivity: ε-radius graph, controlled by FACEMethod.epsilon.
    Edge weights: -log(p̂(midpoint)) · dist.
    """

    def __init__(self, bandwidth: float | str = "scott", kernel: str = "gaussian") -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self._kde: KernelDensity | None = None

    def fit(self, z_train: np.ndarray) -> None:
        self._kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self._kde.fit(z_train)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        assert self._kde is not None, "Call fit() before evaluating."
        z = np.asarray(z, dtype=np.float32)
        return np.exp(self._kde.score_samples(z)).astype(np.float32)


class KNNEstimator(BaseDensityEstimator):
    """k-NN density estimator (Eq. 3 in the paper).

    Graph connectivity: k-NN (kneighbors_graph).
    Edge weights: -log(k / (N · ηd · dist_k(midpoint))) · dist.

    Args:
        n_neighbors: k — controls both graph connectivity and the density formula.
        epsilon: optional distance threshold to prune long-range kNN edges (default: no pruning).
    """

    def __init__(self, n_neighbors: int = 15) -> None:
        self.n_neighbors = n_neighbors
        self._nn: NearestNeighbors | None = None
        self._log_eta_d: float = 0.0
        self._n_samples: int = 1

    def fit(self, z_train: np.ndarray) -> None:
        self._n_samples = len(z_train)
        self._log_eta_d = log_unit_ball_volume(z_train.shape[1])
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self._nn.fit(z_train)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        assert self._nn is not None, "Call fit() before evaluating."
        z = np.asarray(z, dtype=np.float32)
        distances, _ = self._nn.kneighbors(z)
        dist_k = distances[:, -1]
        log_density = (
            log(float(self.n_neighbors))
            - log(float(self._n_samples))
            - self._log_eta_d
            - np.log(np.maximum(dist_k, 1e-300))
        )
        # FACE weights use -log(density), so keep densities in (0, 1].
        log_density = np.clip(log_density, np.log(np.finfo(np.float32).tiny), 0.0)
        return np.exp(log_density).astype(np.float32)


class EpsilonBallEstimator(BaseDensityEstimator):
    """Epsilon-ball density estimator (Eq. 4 in the paper).

    Graph connectivity: ε-radius (radius_neighbors_graph).
    Edge weights: -log(count_ε(midpoint) / (N · ηd · ε^d)) · dist.
    """

    def __init__(self, epsilon: float = 0.5) -> None:
        self.epsilon = epsilon
        self._z_train: np.ndarray | None = None
        self._normalizer: float = 1.0

    def fit(self, z_train: np.ndarray) -> None:
        self._z_train = z_train
        n, d = z_train.shape
        self._normalizer = n * unit_ball_volume(d) * (self.epsilon ** d)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        assert self._z_train is not None, "Call fit() before evaluating."
        z = np.asarray(z, dtype=np.float32)
        combined = np.vstack([self._z_train, z])
        n_train = len(self._z_train)
        adj = radius_neighbors_graph(combined, radius=self.epsilon, mode="connectivity", include_self=False)
        counts = np.asarray(adj[n_train:, :n_train].sum(axis=1)).flatten()
        return (counts / max(self._normalizer, 1e-300)).astype(np.float32)
