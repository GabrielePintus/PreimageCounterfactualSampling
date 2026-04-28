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


def immutable_metadata(
    fixed_dims: Optional[np.ndarray],
    immutable_features: Sequence[str],
) -> dict[str, int | str]:
    """Metadata fields shared by constrained methods."""
    return {
        "fixed_dims_count": 0 if fixed_dims is None else int(len(fixed_dims)),
        "immutable_features": ",".join(immutable_features),
    }
