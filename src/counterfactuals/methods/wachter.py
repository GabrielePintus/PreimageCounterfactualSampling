"""Approximate Wachter-style counterfactual generation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class WachterMethod(ProbabilisticMethod):
    """Black-box approximation of Wachter objective via local random search."""

    def __init__(
        self,
        max_iter: int = 800,
        step_scale: float = 0.05,
        lambda_validity: float = 2.0,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.max_iter = max_iter
        self.step_scale = step_scale
        self.lambda_validity = lambda_validity
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
        best = x0.copy()
        best_score = self._objective(x0, best, model, target_class)
        best_success = False

        for _ in range(self.max_iter):
            noise = self.rng.normal(loc=0.0, scale=self.step_scale, size=x0.shape)
            proposal = best + noise * self._feature_scale
            score = self._objective(x0, proposal, model, target_class)
            pred = int(model.predict(proposal)[0])
            success = pred == target_class
            if score < best_score:
                best = proposal
                best_score = score
                best_success = success

        return CounterfactualResult(
            x_cf=best.astype(np.float32),
            success=best_success,
            distance=float(np.linalg.norm(best - x0, ord=2)),
            metadata={"target_class": target_class, "score": float(best_score)},
        )

    def _objective(
        self,
        x_orig: np.ndarray,
        x_candidate: np.ndarray,
        model: ModelInterface,
        target_class: int,
    ) -> float:
        probs = model.predict_proba(x_candidate)[0]
        p_target = float(probs[target_class])
        validity_penalty = 1.0 - p_target
        distance = np.linalg.norm((x_candidate - x_orig) / self._feature_scale, ord=2)
        return float(distance + self.lambda_validity * validity_penalty)

    @staticmethod
    def _resolve_target_class(example: CounterfactualExample, model: ModelInterface) -> int:
        if example.target_class is not None:
            return int(example.target_class)
        pred = int(model.predict(example.x)[0])
        if model.predict_proba(example.x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
