"""Shared prototype-selection utilities for counterfactual methods."""

from __future__ import annotations

import numpy as np


def _fps_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Greedy Farthest Point Sampling: return indices of k maximally spread points.

    Initialization: the point closest to the class mean (deterministic, no
    random seed needed).  Each subsequent step picks the point with the largest
    minimum distance to the already-selected subset.  Uses scipy.cdist for
    O(N·k) pairwise distance computation with no extra dependencies.
    """
    from scipy.spatial.distance import cdist

    k = min(k, len(X))
    if k == len(X):
        return np.arange(len(X))

    mean = X.mean(axis=0, keepdims=True)
    first = int(cdist(mean, X, metric="euclidean").argmin())

    selected = [first]
    min_dists = cdist(X[first : first + 1], X, metric="euclidean")[0]

    for _ in range(k - 1):
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        new_dists = cdist(X[farthest : farthest + 1], X, metric="euclidean")[0]
        np.minimum(min_dists, new_dists, out=min_dists)

    return np.array(selected)


def select_prototype_indices(
    X: np.ndarray,
    k: int,
    method: str = "kmedoids",
    random_state: int = 42,
) -> np.ndarray:
    """Return indices of k prototype points from X.

    Parameters
    ----------
    X : (N, d) array
    k : number of prototypes to select (capped at len(X))
    method : one of {"kmedoids", "kmeans", "fps"}
    random_state : RNG seed (ignored by fps)
    """
    k = min(k, len(X))
    if method == "fps":
        return _fps_indices(X, k)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances as _pw

        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X)
        return _pw(km.cluster_centers_, X).argmin(axis=1)
    # kmedoids (default)
    try:
        from sklearn_extra.cluster import KMedoids

        import warnings
        km = KMedoids(n_clusters=k, metric="euclidean", method="alternate", random_state=random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cluster .* is empty", category=UserWarning)
            km.fit(X)
        return km.medoid_indices_
    except ImportError:
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances as _pw

        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        km.fit(X)
        return _pw(km.cluster_centers_, X).argmin(axis=1)
