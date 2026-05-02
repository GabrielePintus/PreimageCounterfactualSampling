"""Shared helpers for coordinate-level counterfactual constraints."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def normalize_fixed_dims(fixed_dims: Optional[Sequence[int]]) -> Optional[np.ndarray]:
    """Return sorted unique non-negative fixed coordinates, or ``None``."""
    if fixed_dims is None:
        return None
    arr = np.asarray(fixed_dims)
    if arr.size == 0:
        return None
    if arr.ndim != 1:
        raise ValueError("fixed_dims must be a one-dimensional sequence of non-negative integers")
    int_arr = arr.astype(np.int64)
    if not np.all(arr == int_arr):
        raise ValueError("fixed_dims must contain only integers")
    if np.any(int_arr < 0):
        raise ValueError("fixed_dims must contain only non-negative integers")
    return np.unique(int_arr)


def normalize_directional_dims(
    dims: Optional[Sequence[int]],
    *,
    name: str,
) -> Optional[np.ndarray]:
    """Return sorted unique non-negative directional coordinates, or ``None``."""
    try:
        return normalize_fixed_dims(dims)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a one-dimensional sequence of non-negative integers"
        ) from exc


def validate_disjoint_directional_dims(
    nondecreasing_dims: Optional[np.ndarray],
    nonincreasing_dims: Optional[np.ndarray],
) -> None:
    """Reject dimensions constrained in both monotonic directions."""
    if nondecreasing_dims is None or nonincreasing_dims is None:
        return
    overlap = np.intersect1d(nondecreasing_dims, nonincreasing_dims)
    if overlap.size:
        dims = ", ".join(str(int(i)) for i in overlap)
        raise ValueError(
            "Directional constraints cannot require the same dimension to be "
            f"both nondecreasing and nonincreasing: {dims}."
        )


def immutable_metadata(
    fixed_dims: Optional[np.ndarray],
    immutable_features: Sequence[str],
) -> dict[str, int | str]:
    """Metadata fields shared by constrained methods."""
    return {
        "fixed_dims_count": 0 if fixed_dims is None else int(len(fixed_dims)),
        "immutable_features": ",".join(immutable_features),
    }


def directional_metadata(
    nondecreasing_dims: Optional[np.ndarray],
    nonincreasing_dims: Optional[np.ndarray],
    nondecreasing_features: Sequence[str],
    nonincreasing_features: Sequence[str],
) -> dict[str, int | str]:
    """Metadata fields shared by directional-actionability methods."""
    return {
        "nondecreasing_dims_count": (
            0 if nondecreasing_dims is None else int(len(nondecreasing_dims))
        ),
        "nonincreasing_dims_count": (
            0 if nonincreasing_dims is None else int(len(nonincreasing_dims))
        ),
        "nondecreasing_features": ",".join(nondecreasing_features),
        "nonincreasing_features": ",".join(nonincreasing_features),
    }


def constraint_metadata(
    fixed_dims: Optional[np.ndarray],
    immutable_features: Sequence[str],
    nondecreasing_dims: Optional[np.ndarray],
    nonincreasing_dims: Optional[np.ndarray],
    nondecreasing_features: Sequence[str],
    nonincreasing_features: Sequence[str],
) -> dict[str, int | str]:
    """Metadata fields shared by immutable and directional constraints."""
    metadata = immutable_metadata(fixed_dims, immutable_features)
    metadata.update(
        directional_metadata(
            nondecreasing_dims,
            nonincreasing_dims,
            nondecreasing_features,
            nonincreasing_features,
        )
    )
    return metadata


def directional_compatibility_mask(
    candidates: np.ndarray,
    x_query: np.ndarray,
    nondecreasing_dims: Optional[np.ndarray],
    nonincreasing_dims: Optional[np.ndarray],
    *,
    tol: float = 1e-6,
) -> np.ndarray:
    """Return candidate mask satisfying query-relative directional constraints."""
    candidates_2d = np.asarray(candidates)
    if candidates_2d.ndim == 1:
        candidates_2d = candidates_2d.reshape(1, -1)
    x_query_1d = np.asarray(x_query).reshape(-1)
    mask = np.ones(candidates_2d.shape[0], dtype=bool)
    if nondecreasing_dims is not None and len(nondecreasing_dims) > 0:
        mask &= np.all(
            candidates_2d[:, nondecreasing_dims]
            >= x_query_1d[nondecreasing_dims][None, :] - tol,
            axis=1,
        )
    if nonincreasing_dims is not None and len(nonincreasing_dims) > 0:
        mask &= np.all(
            candidates_2d[:, nonincreasing_dims]
            <= x_query_1d[nonincreasing_dims][None, :] + tol,
            axis=1,
        )
    return mask


def satisfies_directional_constraints(
    x_candidate: np.ndarray,
    x_query: np.ndarray,
    nondecreasing_dims: Optional[np.ndarray],
    nonincreasing_dims: Optional[np.ndarray],
    *,
    tol: float = 1e-6,
) -> bool:
    """Return whether one candidate satisfies directional constraints."""
    return bool(
        directional_compatibility_mask(
            np.asarray(x_candidate).reshape(1, -1),
            x_query,
            nondecreasing_dims,
            nonincreasing_dims,
            tol=tol,
        )[0]
    )


def apply_directional_constraints(
    x: np.ndarray,
    x_query: np.ndarray,
    nondecreasing_dims: Optional[np.ndarray],
    nonincreasing_dims: Optional[np.ndarray],
) -> np.ndarray:
    """Project coordinates onto query-relative directional halfspaces by clipping."""
    if nondecreasing_dims is None and nonincreasing_dims is None:
        return x
    constrained = np.asarray(x).copy()
    x_query_1d = np.asarray(x_query).reshape(-1)
    if nondecreasing_dims is not None and len(nondecreasing_dims) > 0:
        constrained[..., nondecreasing_dims] = np.maximum(
            constrained[..., nondecreasing_dims],
            x_query_1d[nondecreasing_dims],
        )
    if nonincreasing_dims is not None and len(nonincreasing_dims) > 0:
        constrained[..., nonincreasing_dims] = np.minimum(
            constrained[..., nonincreasing_dims],
            x_query_1d[nonincreasing_dims],
        )
    return constrained
