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
        p = make_polygon(bd['lA'][i], bd['lbias'][i], bd['X'][i], eps_i, tol=tol)
        if p is not None:
            polys.append(p)

    if not polys:
        return Polygon()

    return unary_union(polys)


def refine_unions_by_priority(
    unions: dict[int, Polygon | MultiPolygon],
    priority_order: list[int] | None = None
) -> dict[int, Polygon | MultiPolygon]:
    """
    Subtract higher-priority class regions from lower-priority ones.

    Since LiRPA inner approximations can overlap between classes,
    we resolve conflicts by assigning priorities: higher-priority
    classes keep their full footprint, while lower-priority classes
    have overlapping regions removed.

    U'_y = U_y minus (union of all U_y' for y' with higher priority)

    Parameters
    ----------
    unions : dict[int, Polygon | MultiPolygon]
        Dictionary mapping class labels to their union geometries.
    priority_order : list[int], optional
        List of labels in descending priority order (highest first).
        If None, uses sorted label order.

    Returns
    -------
    dict[int, Polygon | MultiPolygon]
        Refined unions with overlaps removed according to priority.
    """
    _require_shapely()
    if priority_order is None:
        priority_order = sorted(unions.keys())

    refined = {}
    placed = Polygon()  # Accumulator of all previously placed regions

    for label in priority_order:
        region = unions[label]

        if region.is_empty:
            refined[label] = region
            continue

        # Remove all higher-priority regions from this class
        refined[label] = region.difference(placed)

        # Add this class's original footprint to the accumulator
        placed = placed.union(region)

    return refined


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
