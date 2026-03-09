"""1-NN baseline counterfactual method."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class NearestNeighborMethod(ProbabilisticMethod):
    """Return the closest training sample from the requested target class."""

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed=random_seed)
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, model: ModelInterface) -> None:
        del model
        self._x_train = np.asarray(x_train, dtype=np.float32)
        self._y_train = np.asarray(y_train, dtype=np.int64)
        if self._x_train.shape[0] != self._y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of rows")
        self._is_fitted = True

    def generate(self, example: CounterfactualExample, model: ModelInterface) -> CounterfactualResult:
        if not self._is_fitted or self._x_train is None or self._y_train is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x0 = self._as_1d(example.x)
        target_class = self._resolve_target_class(example=example, model=model)

        target_mask = self._y_train == target_class
        if not np.any(target_mask):
            raise ValueError(f"No training samples available for target_class={target_class}")

        # NICE requires the nearest unlike neighbour to be correctly classified by
        # the model (i.e. f(xn) == yn), as stated in Algorithm 1 line 8 of the paper:
        # "FIND-NEAREST-UNLIKE-NEIGHBOUR(x0)" with yn == ŷn.
        x_candidate = self._x_train[target_mask]
        candidate_indices = np.flatnonzero(target_mask)
        predicted = model.predict(x_candidate)
        correct_mask = predicted == target_class
        if np.any(correct_mask):
            x_target = x_candidate[correct_mask]
            candidate_indices = candidate_indices[correct_mask]
        else:
            # Fall back to all target-label instances if none are correctly classified.
            x_target = x_candidate

        distances = np.linalg.norm(x_target - x0[None, :], ord=2, axis=1)
        best_local_idx = int(np.argmin(distances))
        best_global_idx = int(candidate_indices[best_local_idx])
        x_cf = x_target[best_local_idx].astype(np.float32)

        pred = int(model.predict(x_cf)[0])
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

    @staticmethod
    def _resolve_target_class(example: CounterfactualExample, model: ModelInterface) -> int:
        if example.target_class is not None:
            return int(example.target_class)
        pred = int(model.predict(example.x)[0])
        if model.predict_proba(example.x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
