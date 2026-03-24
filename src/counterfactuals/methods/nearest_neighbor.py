"""Nearest-neighbour baseline counterfactual method.

Returns the closest opposite-class training point as a counterfactual.
Used as a lower-bound baseline for distance metrics.
"""

from __future__ import annotations

import numpy as np

from counterfactuals.core.base_classes import BaseCounterfactualMethod, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface


class NearestNeighborMethod(BaseCounterfactualMethod):
    """Return the closest training sample from the requested target class."""

    def __init__(
        self,
        # The trained classifier
        model: ModelInterface,

        # Norm used to measure distance to candidates (e.g. 1, 2, "inf")
        norm: int | float | str = 2,

        # Custom downsampling strategy
        subsample_method: str = "kmedoids",
        k_per_class: int | None = None,

        # Random seed for reproducibility (e.g., in subsampling)
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed, k_per_class=k_per_class, subsample_method=subsample_method)
        self.norm = float(norm) if isinstance(norm, str) else norm

    def _fit(self) -> None:
        self.train_pred = self.model.predict(self._x_train)

    def generate(self, x: np.ndarray, target_class: int) -> CounterfactualResult:
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x_query = np.asarray(x, dtype=np.float32).reshape(-1)

        target_mask = self.train_pred == target_class
        if not np.any(target_mask):
            raise ValueError(f"No training samples predicted as target_class={target_class}")

        # Restrict the search to samples the model predicts as target_class, then
        # do a plain L2 nearest-neighbor query in the method's active feature space.
        candidates = self._x_train[target_mask]
        candidate_indices = np.flatnonzero(target_mask)

        distances = np.linalg.norm(candidates - x_query[None, :], ord=self.norm, axis=1)
        best_idx = int(np.argmin(distances))
        x_cf = candidates[best_idx].astype(np.float32)

        return CounterfactualResult(
            x_cf=x_cf,
            success=True,
            distance=float(distances[best_idx]),
            metadata={
                "target_class": int(target_class),
                "nearest_train_index": int(candidate_indices[best_idx]),
                "nearest_train_distance": float(distances[best_idx]),
            },
        )

