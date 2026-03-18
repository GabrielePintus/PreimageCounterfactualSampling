"""Shared method utilities for counterfactual generators."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.base_classes import BaseCounterfactualMethod


class ProbabilisticMethod(BaseCounterfactualMethod):
    """Base class for methods that rely on random candidate sampling."""

    def __init__(
        self,
        model,
        random_seed: int = 42,
        k_per_class: int | None = None,
        subsample_method: str = "kmedoids",
    ):
        super().__init__(model=model, random_seed=random_seed, k_per_class=k_per_class, subsample_method=subsample_method)
        # Each method instance owns a dedicated RNG so repeated benchmark runs are
        # reproducible and do not share global numpy state implicitly.
        self.rng = np.random.default_rng(self.random_seed)

    @staticmethod
    def _as_1d(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Expected a 1D sample")
        return arr
