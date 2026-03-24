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

from counterfactuals.core.base_classes import CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from counterfactuals.core.base_classes import BaseCounterfactualMethod


class WachterMethod(BaseCounterfactualMethod):
    """Wachter et al. (2017) counterfactual via L-BFGS-B + lambda schedule."""

    def __init__(
        self,
        model: ModelInterface,
        lambda_schedule: Optional[List[float]] = None,
        max_iter: int = 200,
        n_restarts: int = 2,
        restart_scale: float = 1.0,
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)
        self.lambda_schedule = lambda_schedule or [1.0, 5.0, 20.0, 100.0, 500.0]
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.restart_scale = restart_scale
        self._feature_scale: Optional[np.ndarray] = None

    def _fit(self) -> None:
        assert self._x_train is not None
        std = np.std(self._x_train.astype(np.float64), axis=0)
        std[std == 0.0] = 1.0
        # Distances are normalised feature-wise so large-scale coordinates do not
        # dominate the optimisation objective.
        self._feature_scale = std

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        if not self._is_fitted or self._feature_scale is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")
        x_query = np.asarray(x, dtype=np.float32).reshape(-1).astype(np.float64)
        target_class = self.resolve_target_class(x=x_query, target_class=target_class)
        scale = self._feature_scale

        best_x: Optional[np.ndarray] = None
        best_dist = np.inf
        best_success = False
        used_lambda = float(self.lambda_schedule[-1])

        for lam in self.lambda_schedule:
            # Each lambda controls the trade-off between staying close to the
            # query and reaching the decision boundary of the target class.
            starts = [x_query.copy()]
            for _ in range(self.n_restarts):
                noise = self.rng.normal(0.0, scale * self.restart_scale, size=x_query.shape)
                starts.append(x_query + noise)

            for x_start in starts:
                def _obj(x: np.ndarray) -> float:
                    dist_sq = float(np.sum(((x - x_query) / scale) ** 2))
                    probs = self.model.predict_proba(x[None, :])[0]
                    # Wachter's objective only needs to reach the classification
                    # boundary; it does not try to drive the target probability to 1.
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
                y_cand = int(self.model.predict(x_cand[None, :])[0])
                if y_cand == target_class:
                    dist = float(np.linalg.norm(x_cand - x_query, ord=2))
                    if dist < best_dist:
                        best_dist = dist
                        best_x = x_cand
                        best_success = True
                        used_lambda = lam

            if best_success:
                break  # found a valid CF — no need for larger lambda

        x_cf = (best_x if best_x is not None else x_query).astype(np.float32)
        return CounterfactualResult(
            x_cf=x_cf,
            success=best_success,
            distance=best_dist if best_success else 0.0,
            metadata={"target_class": target_class, "lambda": used_lambda},
        )

