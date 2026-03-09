"""Growing Spheres baseline for model-agnostic counterfactual search."""

from __future__ import annotations

from typing import Optional

import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class GrowingSpheresMethod(ProbabilisticMethod):
    """Expand spherical shells until a target-class point is found."""

    def __init__(
        self,
        n_in_layer: int = 512,
        max_radius: float = 3.0,
        radius_step: float = 0.15,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.n_in_layer = n_in_layer
        self.max_radius = max_radius
        self.radius_step = radius_step
        self._feature_scale: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, model: ModelInterface) -> None:
        del y_train, model
        std = np.std(np.asarray(x_train, dtype=np.float32), axis=0)
        std[std == 0.0] = 1.0
        self._feature_scale = std
        self._is_fitted = True

    def generate(self, example: CounterfactualExample, model: ModelInterface) -> CounterfactualResult:
        if not self._is_fitted or self._feature_scale is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x0 = self._as_1d(example.x)
        target_class = self._resolve_target_class(example, model)

        radius = self.radius_step
        best = None
        best_dist = float("inf")

        while radius <= self.max_radius:
            candidates = self._sample_shell(x0=x0, radius=radius)
            preds = model.predict(candidates)
            valid = candidates[preds == target_class]
            if len(valid) > 0:
                dists = np.linalg.norm(valid - x0[None, :], axis=1)
                idx = int(np.argmin(dists))
                best = valid[idx]
                best_dist = float(dists[idx])
                break
            radius += self.radius_step

        if best is None:
            return CounterfactualResult(
                x_cf=x0.copy(),
                success=False,
                distance=0.0,
                metadata={"target_class": target_class, "searched_radius": self.max_radius},
            )

        return CounterfactualResult(
            x_cf=best.astype(np.float32),
            success=True,
            distance=best_dist,
            metadata={"target_class": target_class, "searched_radius": radius},
        )

    def _sample_shell(self, x0: np.ndarray, radius: float) -> np.ndarray:
        directions = self.rng.normal(size=(self.n_in_layer, x0.shape[0]))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit = directions / norms
        scaled = unit * radius * self._feature_scale[None, :]
        return x0[None, :] + scaled

    @staticmethod
    def _resolve_target_class(example: CounterfactualExample, model: ModelInterface) -> int:
        if example.target_class is not None:
            return int(example.target_class)
        pred = int(model.predict(example.x)[0])
        if model.predict_proba(example.x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
