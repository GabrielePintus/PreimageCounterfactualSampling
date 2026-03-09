"""Wachter-style counterfactual generation via gradient-based optimisation.

The Wachter et al. (2017) objective is::

    L(x') = ||x' - x0||^2_sigma + lambda * (f_target(x') - 0.5)^2

where ``f_target`` is the model's probability for the target class and
``sigma`` is per-feature standard deviation (MAD scaling from the paper is
approximated by std here).

We solve this with L-BFGS-B and finite-difference gradients (model-agnostic,
works with any sklearn-style ``predict_proba``).  A lambda schedule increases
the validity weight until a valid counterfactual is found, matching the
binary-search spirit of the original paper.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.optimize

from counterfactuals.core.base_classes import CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class WachterMethod(ProbabilisticMethod):
    """Wachter et al. (2017) counterfactual via L-BFGS-B + lambda schedule."""

    def __init__(
        self,
        lambda_schedule: Optional[List[float]] = None,
        max_iter: int = 200,
        n_restarts: int = 2,
        restart_scale: float = 1.0,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        self.lambda_schedule = lambda_schedule or [1.0, 5.0, 20.0, 100.0, 500.0]
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.restart_scale = restart_scale
        self._feature_scale: Optional[np.ndarray] = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, model: ModelInterface) -> None:
        del y_train, model
        std = np.std(np.asarray(x_train, dtype=np.float64), axis=0)
        std[std == 0.0] = 1.0
        self._feature_scale = std
        self._is_fitted = True

    def generate(self, example: CounterfactualExample, model: ModelInterface) -> CounterfactualResult:
        if not self._is_fitted or self._feature_scale is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")
        x0 = self._as_1d(example.x).astype(np.float64)
        target_class = self._resolve_target_class(example, model)
        scale = self._feature_scale

        best_x: Optional[np.ndarray] = None
        best_dist = np.inf
        best_success = False
        used_lambda = float(self.lambda_schedule[-1])

        for lam in self.lambda_schedule:
            # Starting points: original + n_restarts random perturbations
            starts = [x0.copy()]
            for _ in range(self.n_restarts):
                noise = self.rng.normal(0.0, scale * self.restart_scale, size=x0.shape)
                starts.append(x0 + noise)

            for x_start in starts:
                def _obj(x: np.ndarray) -> float:
                    dist_sq = float(np.sum(((x - x0) / scale) ** 2))
                    probs = model.predict_proba(x[None, :])[0]
                    p_target = float(np.clip(probs[target_class], 1e-9, 1 - 1e-9))
                    validity_loss = (p_target - 0.5) ** 2  # 0 when p_target == 0.5 (boundary)
                    return dist_sq + lam * validity_loss

                res = scipy.optimize.minimize(
                    _obj,
                    x_start,
                    method="L-BFGS-B",
                    jac=None,          # forward-difference numerical gradient
                    options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-7},
                )
                x_cand = res.x.astype(np.float64)
                y_cand = int(model.predict(x_cand[None, :])[0])
                if y_cand == target_class:
                    dist = float(np.linalg.norm(x_cand - x0, ord=2))
                    if dist < best_dist:
                        best_dist = dist
                        best_x = x_cand
                        best_success = True
                        used_lambda = lam

            if best_success:
                break  # found a valid CF — no need for larger lambda

        x_cf = (best_x if best_x is not None else x0).astype(np.float32)
        return CounterfactualResult(
            x_cf=x_cf,
            success=best_success,
            distance=best_dist if best_success else 0.0,
            metadata={"target_class": target_class, "lambda": used_lambda},
        )

    @staticmethod
    def _resolve_target_class(example: CounterfactualExample, model: ModelInterface) -> int:
        if example.target_class is not None:
            return int(example.target_class)
        pred = int(model.predict(example.x)[0])
        if model.predict_proba(example.x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
