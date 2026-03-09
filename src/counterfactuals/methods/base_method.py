"""Shared method utilities for counterfactual generators."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.base_classes import BaseCounterfactualMethod


class ProbabilisticMethod(BaseCounterfactualMethod):
    """Base class for methods that rely on random candidate sampling."""

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self.rng = np.random.default_rng(self.random_seed)

    @staticmethod
    def _as_1d(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Expected a 1D sample")
        return arr
