"""DiCE-inspired diverse counterfactual generation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class DiceMethod(ProbabilisticMethod):
    """Generate one diverse counterfactual by maximizing validity and diversity."""

    def __init__(
        self,
        num_candidates: int = 1024,
        diversity_weight: float = 0.2,
        noise_scale: float = 0.15,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.num_candidates = num_candidates
        self.diversity_weight = diversity_weight
        self.noise_scale = noise_scale
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
        raw_candidates = self._sample_candidates(x0)
        valid, distances = self._collect_valid(raw_candidates, model, target_class, x0)

        if len(valid) == 0:
            return CounterfactualResult(
                x_cf=x0.copy(),
                success=False,
                distance=0.0,
                metadata={"target_class": target_class, "reason": "no_valid_candidate"},
            )

        selected = self._select_by_diversity(valid, distances, x0)
        distance = float(np.linalg.norm(selected - x0, ord=2))
        return CounterfactualResult(
            x_cf=selected.astype(np.float32),
            success=True,
            distance=distance,
            metadata={"target_class": target_class, "n_valid": int(len(valid))},
        )

    def _sample_candidates(self, x0: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(
            loc=0.0,
            scale=self.noise_scale,
            size=(self.num_candidates, x0.shape[0]),
        )
        return x0[None, :] + noise * self._feature_scale[None, :]

    def _collect_valid(
        self,
        candidates: np.ndarray,
        model: ModelInterface,
        target_class: int,
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = model.predict(candidates)
        mask = preds == target_class
        valid = candidates[mask]
        distances = np.linalg.norm((valid - x0[None, :]) / self._feature_scale[None, :], axis=1)
        return valid, distances

    def _select_by_diversity(
        self,
        valid: np.ndarray,
        distances: np.ndarray,
        x0: np.ndarray,
    ) -> np.ndarray:
        centroid = np.mean(valid, axis=0)
        diversity = np.linalg.norm((valid - centroid[None, :]) / self._feature_scale[None, :], axis=1)
        score = distances - self.diversity_weight * diversity
        idx = int(np.argmin(score))
        best = valid[idx]
        return np.where(np.isnan(best), x0, best)

    @staticmethod
    def _resolve_target_class(example: CounterfactualExample, model: ModelInterface) -> int:
        if example.target_class is not None:
            return int(example.target_class)
        pred = int(model.predict(example.x)[0])
        if model.predict_proba(example.x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
