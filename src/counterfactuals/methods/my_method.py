"""Adapter for the project's CertifiedAtlas method."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class CertifiedAtlasMethod(ProbabilisticMethod):
    """Wrap ``preimage_sampling.CertifiedAtlas`` into the common method interface."""

    def __init__(self, atlas, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self.atlas = atlas

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, model: ModelInterface) -> None:
        del x_train, y_train, model
        self._is_fitted = True

    def generate(self, example: CounterfactualExample, model: ModelInterface) -> CounterfactualResult:
        del model
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")
        if example.target_class is None:
            raise ValueError("CertifiedAtlasMethod requires target_class.")

        result = self.atlas.find_counterfactual(
            x_query=np.asarray(example.x, dtype=np.float32),
            target_class=int(example.target_class),
        )

        x_cf = np.asarray(result.x_cf, dtype=np.float32)
        distance = float(result.distance)
        success = bool(result.success)
        metadata = dict(getattr(result, "metadata", {}))
        return CounterfactualResult(x_cf=x_cf, success=success, distance=distance, metadata=metadata)
