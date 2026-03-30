"""Shared prototype-selection utilities for counterfactual methods."""

from __future__ import annotations

import numpy as np


def _fps_indices(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Greedy Farthest Point Sampling: return indices of k maximally spread points.

    Initialization: the point closest to the class mean (deterministic, no
    random seed needed).  Each subsequent step picks the point with the largest
    minimum distance to the already-selected subset.  Uses scipy.cdist for
    O(N·k) pairwise distance computation with no extra dependencies.
    """
    del random_state
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


def _kmeans_indices(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Select prototypes by assigning each KMeans centroid to its nearest sample."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances as _pw

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(X)
    return _pw(km.cluster_centers_, X).argmin(axis=1)


def _kmedoids_indices(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Select prototypes with KMedoids, falling back to KMeans if unavailable."""
    try:
        import warnings

        from sklearn_extra.cluster import KMedoids

        km = KMedoids(n_clusters=k, metric="euclidean", method="alternate", random_state=random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Cluster .* is empty", category=UserWarning)
            km.fit(X)
        return km.medoid_indices_
    except ImportError:
        return _kmeans_indices(X, k, random_state=random_state)


def _bandit_kmedoids_indices(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Compatibility alias for the current bandit-kmedoids selection path."""
    return _kmedoids_indices(X, k, random_state=random_state)


def _density_flat_kmedoids_indices(
    X: np.ndarray,
    k: int,
    subset_size: int = 30_000,
    random_state: int = 42,
) -> np.ndarray:
    import warnings
    from sklearn.neighbors import NearestNeighbors
    from sklearn_extra.cluster import KMedoids

    rng = np.random.default_rng(random_state)
    N = len(X)
    subset_size = min(N, subset_size)

    print(f"Selecting {k} prototypes with density-flattened KMedoids on a subset of {subset_size} points...")

    # --- Simple uniform subsampling ---
    print("Number of points before subsampling:", N)
    sample_idx = rng.choice(N, size=subset_size, replace=False)
    X_subset = X[sample_idx]
    print("Number of points after subsampling:", len(X_subset))

    # --- PAM on subset ---
    km = KMedoids(
        n_clusters=k, metric="euclidean", method="pam",
        init="k-medoids++", random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cluster .* is empty", category=UserWarning)
        km.fit(X_subset)

    return sample_idx[km.medoid_indices_]


_PROTOTYPE_SELECTION_METHODS = {
    "fps": _fps_indices,
    "kmeans": _kmeans_indices,
    "kmedoids": _kmedoids_indices,
    "bandit_kmedoids": _bandit_kmedoids_indices,
    "density_flat_kmedoids": _density_flat_kmedoids_indices,
}


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
    method : one of {"kmedoids", "bandit_kmedoids", "kmeans", "fps", "density_flat_kmedoids"}
    random_state : RNG seed (ignored by fps)
    """
    k = min(k, len(X))
    try:
        selector = _PROTOTYPE_SELECTION_METHODS[method]
    except KeyError as exc:
        available = ", ".join(sorted(_PROTOTYPE_SELECTION_METHODS))
        raise ValueError(f"Unknown prototype selection method '{method}'. Expected one of: {available}") from exc
    return selector(X, k, random_state=random_state)
