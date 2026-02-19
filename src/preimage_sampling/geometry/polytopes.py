"""Polytope construction and basic operations."""

import numpy as np
from shapely.geometry import Polygon

from .conversion import halfspace_to_vertices


def ball_box_constraints(center: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return halfspace constraints for axis-aligned box [center-eps, center+eps].

    This box is a conservative (larger) approximation of the Lp ball
    B(center, eps) and is used to clip certified polytopes to their
    validity region.

    Convention: A x + b >= 0

    For each dimension i:
      x_i >= center_i - eps   →   row  e_i,  bias -(center_i - eps)
      x_i <= center_i + eps   →   row -e_i,  bias  (center_i + eps)

    Parameters
    ----------
    center : np.ndarray
        Center point of the box, shape (d,).
    eps : float
        Half-width of the box (perturbation radius).

    Returns
    -------
    A_box : np.ndarray
        Constraint matrix of shape (2d, d).
    b_box : np.ndarray
        Bias vector of shape (2d,).
    """
    d = len(center)
    A_box = np.vstack([np.eye(d), -np.eye(d)])  # (2d, d)
    b_box = np.concatenate([-(center - eps), center + eps])  # (2d,)
    return A_box, b_box


def make_polygon(
    A_i: np.ndarray,
    b_i: np.ndarray,
    x0: np.ndarray,
    eps: float,
    tol: float = 1e-7
) -> Polygon | None:
    """
    Build a certified inner polytope for one sample.

    Intersects the LiRPA halfspaces {A_i x + b_i >= 0} with the
    epsilon-ball bounding box around x0. This ensures the polytope
    is only valid within the region where the LiRPA bounds hold.

    Parameters
    ----------
    A_i : np.ndarray
        Lower-bound A matrix for this sample, shape (k-1, d).
    b_i : np.ndarray
        Lower-bound bias vector, shape (k-1,).
    x0 : np.ndarray
        Nominal point (center of the eps-ball), shape (d,).
    eps : float
        Perturbation radius defining the validity region.
    tol : float, optional
        Tolerance for feasibility check (default: 1e-7).

    Returns
    -------
    Polygon or None
        A shapely Polygon if the intersection is valid and non-degenerate,
        None otherwise.
    """
    # Get box constraints for the epsilon ball
    A_box, b_box = ball_box_constraints(x0, eps)

    # Combine LiRPA constraints with box constraints
    A_full = np.vstack([A_i, A_box])
    b_full = np.concatenate([b_i, b_box])

    # Convert halfspace representation to vertices
    verts = halfspace_to_vertices(A_full, b_full, x0, tol=tol)
    if verts is None:
        return None

    # Create polygon
    poly = Polygon(verts)

    # Return only if valid and non-degenerate
    if poly.is_valid and poly.area > 1e-12:
        return poly

    return None
