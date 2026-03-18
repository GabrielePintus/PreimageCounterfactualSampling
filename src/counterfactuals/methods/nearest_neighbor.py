"""1-NN baseline counterfactual method."""

from __future__ import annotations

from typing import Optional

import numpy as np

from counterfactuals.core.base_classes import CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class NearestNeighborMethod(ProbabilisticMethod):
    """Return the closest training sample from the requested target class."""

    def __init__(
        self,
        model: ModelInterface,
        random_seed: int = 42,
        k_per_class: int | None = None,
        subsample_method: str = "kmedoids",
    ):
        super().__init__(model=model, random_seed=random_seed, k_per_class=k_per_class, subsample_method=subsample_method)

    def _fit(self) -> None:
        pass

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        if not self._is_fitted or self._x_train is None or self._y_train is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x0 = self._as_1d(x)
        target_class = self._resolve_target_class(x=x0, target_class=target_class)

        target_mask = self._y_train == target_class
        if not np.any(target_mask):
            raise ValueError(f"No training samples available for target_class={target_class}")

        # Restrict the search to the requested target class, then do a plain L2
        # nearest-neighbor query in the method's active feature space.
        x_target = self._x_train[target_mask]
        candidate_indices = np.flatnonzero(target_mask)

        distances = np.linalg.norm(x_target - x0[None, :], ord=2, axis=1)
        best_local_idx = int(np.argmin(distances))
        best_global_idx = int(candidate_indices[best_local_idx])
        x_cf = x_target[best_local_idx].astype(np.float32)

        pred = int(self.model.predict(x_cf)[0])
        success = pred == target_class

        return CounterfactualResult(
            x_cf=x_cf,
            success=success,
            distance=float(distances[best_local_idx]),
            metadata={
                "target_class": int(target_class),
                "nearest_train_index": best_global_idx,
                "nearest_train_distance": float(distances[best_local_idx]),
            },
        )

    def _resolve_target_class(self, x: np.ndarray, target_class: Optional[int]) -> int:
        if target_class is not None:
            return int(target_class)
        pred = int(self.model.predict(x)[0])
        if self.model.predict_proba(x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
