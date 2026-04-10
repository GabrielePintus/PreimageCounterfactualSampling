"""
Pluggable strategies for computing per-sample epsilon values during atlas construction.

Each strategy exposes a single method::

    eps = strategy.compute_eps(X, y)   # shape (N,)

The returned array is aligned with the dataset rows (all classes combined).
``CertCFAtlas.build()`` splits it by class before passing it downstream.
"""

from abc import ABC, abstractmethod

import numpy as np


class EpsStrategy(ABC):
    """Abstract base class for per-sample epsilon strategies."""

    @abstractmethod
    def compute_eps(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute per-sample epsilon values for the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Input samples from all classes combined.
        y : np.ndarray, shape (N,)
            Corresponding class labels (integer).

        Returns
        -------
        eps : np.ndarray, shape (N,)
            Non-negative per-sample epsilon values, one per row of X.
        """


class ConstantEpsStrategy(EpsStrategy):
    """
    Returns the same epsilon for every data point.

    This is the original behaviour of ``CertCFAtlas.build(eps=...)``.

    Parameters
    ----------
    eps : float
        Perturbation radius used for every sample.
    """

    def __init__(self, eps: float):
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps

    def compute_eps(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.eps)

    def __repr__(self) -> str:
        return f"ConstantEpsStrategy(eps={self.eps})"


class NearestOppositeClassClearanceStrategy(EpsStrategy):
    """
    Per-point epsilon set to a fraction of the L-infinity clearance to the
    nearest opposite-class sample.

    For each point x_i with label c_i:

        d_i = min_{j: y_j != c_i}  ||x_i - x_j||_inf
        eps_i = alpha * d_i

    **Intuition**: points close to a class boundary receive a small epsilon
    (tight certification ball, better feasibility); isolated points receive a
    larger epsilon (more coverage, potentially more non-trivial polytopes).

    **Overlap**: same-class overlap is allowed; only opposite-class distance
    matters.

    Parameters
    ----------
    alpha : float
        Clearance scaling factor in (0, 1]. A value of 0.25 is a reasonable
        default. At alpha < 0.5 the eps-ball around x_i cannot reach any
        opposite-class center (it stays strictly within the clearance zone).
        Larger values are allowed for experimentation but lose that guarantee.

    Notes
    -----
    ``compute_eps`` runs once offline (before LiRPA), with cost O(N^2) L-inf
    distance computations implemented via ``scipy.spatial.distance.cdist`` with
    ``metric='chebyshev'``.  For N <= 10 000 and moderate d this is negligible
    compared with the LiRPA calls that follow.
    """

    def __init__(self, alpha: float = 0.25):
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def compute_eps(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import cdist

        X = np.asarray(X)
        y = np.asarray(y)
        N = len(X)
        eps = np.zeros(N)

        for c in np.unique(y):
            mask_c = y == c
            mask_other = ~mask_c

            if not np.any(mask_other):
                # Only one class exists — clearance is undefined; leave eps=0.
                continue

            X_c = X[mask_c]        # (N_c, d)
            X_other = X[mask_other]  # (N_other, d)

            # Pairwise L-inf (Chebyshev) distances: (N_c, N_other)
            dists = cdist(X_c, X_other, metric='chebyshev')

            # Per-point clearance = distance to nearest opposite-class neighbor
            clearance = dists.min(axis=1)  # (N_c,)
            eps[mask_c] = self.alpha * clearance

        return eps

    def __repr__(self) -> str:
        return f"NearestOppositeClassClearanceStrategy(alpha={self.alpha})"
