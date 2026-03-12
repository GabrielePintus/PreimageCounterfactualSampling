"""Scikit-learn compatible model wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from .base_model import BaseModelWrapper


class SklearnModelWrapper(BaseModelWrapper):
    """Adapter for estimators with sklearn-like prediction API."""

    def __init__(self, estimator):
        super().__init__(n_classes=None)
        self.estimator = estimator

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_2d = self._ensure_2d(x)
        return np.asarray(self.estimator.predict(x_2d), dtype=np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_2d = self._ensure_2d(x)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(x_2d), dtype=np.float32)
        logits = np.asarray(self.estimator.decision_function(x_2d), dtype=np.float32)
        if logits.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-logits))
            return np.stack([1.0 - probs_pos, probs_pos], axis=1)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def build_sklearn_mlp(random_seed: int = 42) -> SklearnModelWrapper:
    """Create a deterministic sklearn MLP baseline wrapper."""
    estimator = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        random_state=random_seed,
    )
    return SklearnModelWrapper(estimator=estimator)
