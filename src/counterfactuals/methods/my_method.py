"""Adapter for the project's CertifiedAtlas method."""

from __future__ import annotations

from typing import Optional

import numpy as np

from counterfactuals.core.base_classes import CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from counterfactuals.core.base_classes import BaseCounterfactualMethod


class CertifiedAtlasMethod(BaseCounterfactualMethod):
    """Wrap ``preimage_sampling.CertifiedAtlas`` into the common method interface."""

    def __init__(self, model: ModelInterface, atlas, random_seed: int = 42):
        super().__init__(model=model, random_seed=random_seed)
        self.atlas = atlas

    def _fit(self) -> None:
        # Atlas construction happens outside this adapter. The common interface still
        # expects a fit() phase, so we mark the wrapper as ready here.
        pass

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")
        if target_class is None:
            raise ValueError("CertifiedAtlasMethod requires target_class.")

        # The atlas already owns all search logic; this adapter only converts the
        # project-specific result object into the shared benchmark result schema.
        result = self.atlas.find_counterfactual(
            x_query=np.asarray(x, dtype=np.float32),
            target_class=int(target_class),
        )

        x_cf = np.asarray(result.x_cf, dtype=np.float32)
        distance = float(result.distance)
        success = bool(result.success)
        metadata = dict(getattr(result, "metadata", {}))
        return CounterfactualResult(x_cf=x_cf, success=success, distance=distance, metadata=metadata)
