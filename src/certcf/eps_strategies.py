"""
Pluggable strategies for computing per-sample epsilon values during atlas construction.

Each strategy exposes a single method::

    eps = strategy.compute_eps(X, y)   # shape (N,)

The returned array is aligned with the dataset rows (all classes combined).
``CertCFAtlas.build()`` splits it by class before passing it downstream.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class EpsStrategy(ABC):
    """Abstract base class for per-sample epsilon strategies."""

    @abstractmethod
    def compute_eps(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm: int | float = np.inf,
        parallelism: int = 1,
    ) -> np.ndarray:
        """
        Compute per-sample epsilon values for the full dataset.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Input samples from all classes combined.
        y : np.ndarray, shape (N,)
            Corresponding class labels (integer).
        norm : int or float, optional
            Lp norm used to define the opposite-class clearance. Defaults to
            ``np.inf`` for backward compatibility.
        parallelism : int, optional
            Maximum number of independent distance chunks evaluated at once.

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

    def compute_eps(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm: int | float = np.inf,
        parallelism: int = 1,
    ) -> np.ndarray:
        if int(parallelism) <= 0:
            raise ValueError("parallelism must be positive")
        return np.full(len(X), self.eps)

    def __repr__(self) -> str:
        return f"ConstantEpsStrategy(eps={self.eps})"


class NearestOppositeClassClearanceStrategy(EpsStrategy):
    """
    Per-point epsilon set to a fraction of the Lp clearance to the nearest
    opposite-class sample.

    For each point x_i with label c_i:

        d_i = min_{j: y_j != c_i}  ||x_i - x_j||_p
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
    ``compute_eps`` runs once offline (before LiRPA), with cost O(N^2)
    pairwise distance computations implemented via
    ``scipy.spatial.distance.cdist``. Distance chunks can be evaluated by
    multiple workers through the ``parallelism`` argument; this is especially
    useful when both the number of support points and input dimension are high.
    """

    def __init__(self, alpha: float = 0.25, *, chunk_size: int = 128):
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._reference_X: np.ndarray | None = None
        self._reference_y: np.ndarray | None = None

    def set_reference(self, X: np.ndarray, y: np.ndarray) -> None:
        """Use a separate observed support pool for Eq. (1) clearances."""
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError("reference X must be 2D and aligned with reference y")
        self._reference_X = X
        self._reference_y = y

    def clear_reference(self) -> None:
        self._reference_X = None
        self._reference_y = None

    @staticmethod
    def _resolve_cdist_metric(norm: int | float) -> tuple[str, dict]:
        if norm == np.inf:
            return "chebyshev", {}
        p = float(norm)
        if p <= 0:
            raise ValueError(f"norm must be positive, got {norm}")
        if p == 1.0:
            return "cityblock", {}
        if p == 2.0:
            return "euclidean", {}
        return "minkowski", {"p": p}

    def compute_eps(
        self,
        X: np.ndarray,
        y: np.ndarray,
        norm: int | float = np.inf,
        parallelism: int = 1,
    ) -> np.ndarray:
        from scipy.spatial.distance import cdist

        X = np.asarray(X)
        y = np.asarray(y)
        N = len(X)
        eps = np.zeros(N)
        metric, metric_kwargs = self._resolve_cdist_metric(norm)
        parallelism = int(parallelism)
        if parallelism <= 0:
            raise ValueError("parallelism must be positive")

        reference_X = X if self._reference_X is None else self._reference_X
        reference_y = y if self._reference_y is None else self._reference_y
        if reference_X.shape[1] != X.shape[1]:
            raise ValueError("reference and anchor feature dimensions do not match")

        def chunk_minimum(
            X_chunk: np.ndarray,
            X_other: np.ndarray,
        ) -> np.ndarray:
            distances = cdist(X_chunk, X_other, metric=metric, **metric_kwargs)
            return distances.min(axis=1)

        executor = (
            ThreadPoolExecutor(max_workers=parallelism)
            if parallelism > 1
            else None
        )
        try:
            for c in np.unique(y):
                mask_c = y == c
                mask_other = reference_y != c

                if not np.any(mask_other):
                    # Only one class exists — clearance is undefined; leave eps=0.
                    continue

                X_c = X[mask_c]
                X_other = reference_X[mask_other]
                clearance = np.full(len(X_c), np.inf, dtype=np.float64)
                chunks = []
                for start in range(0, len(X_c), self.chunk_size):
                    stop = min(start + self.chunk_size, len(X_c))
                    if executor is None:
                        clearance[start:stop] = chunk_minimum(
                            X_c[start:stop], X_other
                        )
                    else:
                        future = executor.submit(
                            chunk_minimum, X_c[start:stop], X_other
                        )
                        chunks.append((start, stop, future))
                for start, stop, future in chunks:
                    clearance[start:stop] = future.result()
                eps[mask_c] = self.alpha * clearance
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        return eps

    def __repr__(self) -> str:
        return f"NearestOppositeClassClearanceStrategy(alpha={self.alpha})"
