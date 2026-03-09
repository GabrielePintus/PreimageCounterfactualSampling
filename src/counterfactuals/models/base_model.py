"""Base model wrapper types."""

from __future__ import annotations

from typing import Optional

import numpy as np

from counterfactuals.core.interfaces import ModelInterface


class BaseModelWrapper(ModelInterface):
    """Shared functionality for wrapped predictive models."""

    def __init__(self, n_classes: Optional[int] = None):
        self.n_classes = n_classes

    def _ensure_2d(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr
