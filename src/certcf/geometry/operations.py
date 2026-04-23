"""Geometric operations on polytope collections."""

from __future__ import annotations

try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover - optional dependency for 2D visualization only
    Polygon = None
    MultiPolygon = None
    unary_union = None

from .polytopes import make_polygon


def _require_shapely() -> None:
    if Polygon is None or MultiPolygon is None or unary_union is None:
        raise ImportError("shapely is required for polygon union utilities.")


def build_class_union(
    label: int,
    bounds: dict,
    eps,
    norm: int | float,
    tol: float = 1e-7
) -> Polygon | MultiPolygon:
    """
    Union of all certified polytopes for one class.

    Takes the LiRPA bounds for all samples in a class and constructs
    the union of their individual certified polytopes.

    Parameters
    ----------
    label : int
        The class label.
    bounds : dict
        Dictionary mapping labels to bound dictionaries with keys:
        'lA', 'lbias', 'X' (from PreimageApproximation.compute_all_bounds).
    eps : float or np.ndarray
        Perturbation radius used for clipping polytopes.  Can be a scalar
        (same for all samples) or a 1-D array of shape ``(N,)`` for
        per-sample radii.
    norm : int or float
        Lp norm used to define the exact certified trust region.
    tol : float, optional
        Tolerance for polytope construction (default: 1e-7).

    Returns
    -------
    Polygon or MultiPolygon
        The union of all valid polytopes for this class.
        Returns an empty Polygon if no valid polytopes exist.
    """
    _require_shapely()
    import numpy as np

    bd = bounds[label]
    polys = []

    for i in range(bd['lA'].shape[0]):
        eps_i = float(eps[i]) if isinstance(eps, np.ndarray) else eps
        p = make_polygon(bd['lA'][i], bd['lbias'][i], bd['X'][i], eps_i, norm=norm, tol=tol)
        if p is not None:
            polys.append(p)

    if not polys:
        return Polygon()

    return unary_union(polys)


def compute_overlap_matrix(unions: dict[int, Polygon | MultiPolygon]) -> dict:
    """
    Compute pairwise overlap areas between class unions.

    Parameters
    ----------
    unions : dict[int, Polygon | MultiPolygon]
        Dictionary mapping class labels to their union geometries.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'overlap': 2D array of pairwise overlap areas
        - 'labels': list of class labels in order
    """
    _require_shapely()
    labels = sorted(unions.keys())
    n = len(labels)
    overlap = [[0.0] * n for _ in range(n)]

    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i <= j:
                area = unions[label_i].intersection(unions[label_j]).area
                overlap[i][j] = area
                overlap[j][i] = area

    return {'overlap': overlap, 'labels': labels}


def assert_no_cross_class_overlap(
    unions: dict[int, Polygon | MultiPolygon],
    tol: float = 1e-9,
) -> None:
    """
    Raise if any certified class unions have positive-area overlap.

    Certified regions of different classes must be disjoint. Any positive-area
    overlap indicates a bug in the geometric reconstruction path.
    """
    _require_shapely()
    labels = sorted(unions.keys())
    for i, label_i in enumerate(labels):
        for label_j in labels[i + 1:]:
            area = unions[label_i].intersection(unions[label_j]).area
            if area > tol:
                raise ValueError(
                    "Certified class unions must be disjoint, "
                    f"but classes {label_i} and {label_j} overlap with area {area:.12g}."
                )
