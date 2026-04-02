"""Conversion between halfspace and vertex representations."""

import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull


def halfspace_to_vertices(
    A: np.ndarray,
    b: np.ndarray,
    interior_point: np.ndarray,
    tol: float = 1e-7
) -> np.ndarray | None:
    """
    Convert halfspace representation to ordered vertex array.

    Given halfspaces {x : A x + b >= 0}, computes the vertices of the
    resulting convex polytope and orders them by their convex hull.

    Parameters
    ----------
    A : np.ndarray
        Constraint matrix, shape (m, d).
    b : np.ndarray
        Bias vector, shape (m,).
    interior_point : np.ndarray
        A point strictly inside the polytope, shape (d,).
        Required by scipy's HalfspaceIntersection algorithm.
    tol : float, optional
        Tolerance for feasibility check (default: 1e-7).

    Returns
    -------
    np.ndarray or None
        Array of vertices ordered by convex hull, shape (n_vertices, d).
        Returns None if:
        - The interior point is not feasible
        - The polytope is empty
        - The polytope is degenerate (< 3 vertices in 2D)
    """
    # Check if interior point is feasible
    if np.min(A @ interior_point + b) < tol:
        return None

    try:
        # scipy wants halfspaces in form: A x <= b
        # We have: A x + b >= 0, i.e., -A x <= b
        # So we pass: -A and -b
        hs = HalfspaceIntersection(
            np.hstack([-A, -b[:, None]]),
            interior_point,
        )
    except Exception:
        # Infeasible or numerical issues
        return None

    V = hs.intersections

    # Check for degeneracy
    if V.shape[0] < 3:
        return None

    try:
        # Order vertices by convex hull
        hull = ConvexHull(V)
        return V[hull.vertices]
    except Exception:
        # Degenerate hull
        return None
