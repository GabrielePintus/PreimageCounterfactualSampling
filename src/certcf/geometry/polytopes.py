"""Polytope construction and basic operations."""

from __future__ import annotations

import numpy as np

try:
    from shapely.geometry import Polygon
except ImportError:  # pragma: no cover - optional dependency for 2D visualization only
    Polygon = None

from .conversion import halfspace_to_vertices


def ball_box_constraints(center: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return halfspace constraints for axis-aligned box [center-eps, center+eps].

    This helper is used by the projection/search code as an outer box around
    the true trust region. It must not be used to define certified 2D
    visualization geometry for norms whose balls are not axis-aligned boxes.

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


def exact_2d_ball_constraints(
    center: np.ndarray,
    eps: float,
    norm: int | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the exact 2D halfspace representation of the certified trust region.

    Supported norms are:
    - L-infinity: axis-aligned box
    - L1: diamond defined by the four sign combinations

    Parameters
    ----------
    center : np.ndarray
        Center point of the trust region, shape (2,).
    eps : float
        Radius of the trust region.
    norm : int or float
        Lp norm defining the certified trust region.

    Returns
    -------
    A_ball : np.ndarray
        Constraint matrix in the convention A x + b >= 0.
    b_ball : np.ndarray
        Bias vector in the convention A x + b >= 0.

    Raises
    ------
    ValueError
        If the input is not 2D or the norm is unsupported for exact 2D
        polygonization.
    """
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    if center.shape[0] != 2:
        raise ValueError(
            f"Exact certified 2D trust-region constraints require a 2D center, got dim={center.shape[0]}."
        )

    if norm == np.inf:
        return ball_box_constraints(center, eps)

    if int(norm) == 1:
        sign_rows = np.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=np.float64,
        )
        A_ball = -sign_rows
        b_ball = eps + sign_rows @ center
        return A_ball, b_ball

    raise ValueError(
        f"Exact certified 2D polygon construction only supports L1 and L-inf norms, got norm={norm}."
    )


def make_polygon(
    A_i: np.ndarray,
    b_i: np.ndarray,
    x0: np.ndarray,
    eps: float,
    norm: int | float,
    tol: float = 1e-7
) -> Polygon | None:
    """
    Build a certified inner polytope for one sample.

    Intersects the LiRPA halfspaces {A_i x + b_i >= 0} with the exact 2D
    trust region for the chosen norm. This ensures the polygon is only valid
    within the region where the LiRPA bounds hold.

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
    norm : int or float
        Lp norm defining the certified trust region.
    tol : float, optional
        Tolerance for feasibility check (default: 1e-7).

    Returns
    -------
    Polygon or None
        A shapely Polygon if the intersection is valid and non-degenerate,
        None otherwise.
    """
    if Polygon is None:
        raise ImportError("shapely is required for polygon construction utilities.")

    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0.shape[0] != 2:
        raise ValueError(
            f"Certified polygon construction only supports 2D inputs, got dim={x0.shape[0]}."
        )

    A_ball, b_ball = exact_2d_ball_constraints(x0, eps, norm)

    # Combine LiRPA constraints with exact trust-region constraints
    A_full = np.vstack([A_i, A_ball])
    b_full = np.concatenate([b_i, b_ball])

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
