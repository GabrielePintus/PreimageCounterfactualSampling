"""DiCE-inspired diverse counterfactual generation.

This implementation follows the paper/repo objective structure:
- validity term (y-loss toward desired class),
- proximity term (small perturbation),
- diversity term (spread among generated CFs).

Given the model-agnostic interface in this repository (predict/predict_proba only),
optimization uses a gradient-free evolution-strategy update rather than autograd.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from counterfactuals.core.base_classes import CounterfactualExample, CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface

from .base_method import ProbabilisticMethod


class DiceMethod(ProbabilisticMethod):
    """Generate diverse counterfactuals with a DiCE-style optimization objective."""

    def __init__(
        self,
        total_cfs: int = 4,
        num_candidates: Optional[int] = None,
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
        yloss_type: str = "hinge_loss",
        learning_rate: float = 0.05,
        min_iter: int = 100,
        max_iter: int = 400,
        loss_diff_thres: float = 1e-5,
        loss_converge_maxiter: int = 2,
        stopping_threshold: float = 0.5,
        init_near_query_instance: bool = True,
        init_noise_scale: float = 0.15,
        es_num_directions: int = 12,
        es_sigma: float = 0.05,
        random_seed: int = 42,
    ):
        super().__init__(random_seed=random_seed)
        # Backward compatibility: previous implementation exposed `num_candidates`.
        # Keep accepting it, but `total_cfs` controls how many diverse CFs are optimized.
        del num_candidates
        self.total_cfs = int(total_cfs)
        self.proximity_weight = float(proximity_weight)
        self.diversity_weight = diversity_weight
        self.yloss_type = yloss_type
        self.learning_rate = float(learning_rate)
        self.min_iter = int(min_iter)
        self.max_iter = int(max_iter)
        self.loss_diff_thres = float(loss_diff_thres)
        self.loss_converge_maxiter = int(loss_converge_maxiter)
        self.stopping_threshold = float(stopping_threshold)
        self.init_near_query_instance = bool(init_near_query_instance)
        self.init_noise_scale = float(init_noise_scale)
        self.es_num_directions = int(es_num_directions)
        self.es_sigma = float(es_sigma)
        self._feature_scale: Optional[np.ndarray] = None
        self._x_min: Optional[np.ndarray] = None
        self._x_max: Optional[np.ndarray] = None

        if self.total_cfs <= 0:
            raise ValueError("total_cfs must be > 0")
        if self.yloss_type not in {"hinge_loss", "log_loss", "l2_loss"}:
            raise ValueError("yloss_type must be one of: hinge_loss, log_loss, l2_loss")

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, model: ModelInterface) -> None:
        del y_train, model
        x_train = np.asarray(x_train, dtype=np.float32)
        std = np.std(x_train, axis=0)
        std[std == 0.0] = 1.0
        self._feature_scale = std
        self._x_min = np.min(x_train, axis=0)
        self._x_max = np.max(x_train, axis=0)
        self._is_fitted = True

    def generate(self, example: CounterfactualExample, model: ModelInterface) -> CounterfactualResult:
        if not self._is_fitted or self._feature_scale is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x0 = self._as_1d(example.x)
        target_class = self._resolve_target_class(example, model)
        candidates, final_loss, n_iter = self._optimize_candidates(x0=x0, model=model, target_class=target_class)

        valid_mask, probs_target = self._validity_mask(candidates, model=model, target_class=target_class)
        valid = candidates[valid_mask]

        if valid.shape[0] == 0:
            # Fall back to the best loss candidate if threshold-valid CFs are not found.
            best_idx = int(np.argmin(self._proximity_loss(candidates, x0)))
            fallback = candidates[best_idx]
            return CounterfactualResult(
                x_cf=fallback.astype(np.float32),
                success=False,
                distance=float(np.linalg.norm(fallback - x0, ord=2)),
                metadata={
                    "target_class": target_class,
                    "reason": "no_valid_candidate",
                    "optimization_loss": float(final_loss),
                    "iterations": int(n_iter),
                },
            )

        selected = self._select_best_valid(valid=valid, x0=x0)
        distance = float(np.linalg.norm(selected - x0, ord=2))
        return CounterfactualResult(
            x_cf=selected.astype(np.float32),
            success=True,
            distance=distance,
            metadata={
                "target_class": target_class,
                "n_valid": int(valid.shape[0]),
                "n_candidates": int(candidates.shape[0]),
                "optimization_loss": float(final_loss),
                "iterations": int(n_iter),
                "mean_target_proba": float(np.mean(probs_target)),
            },
        )

    def _optimize_candidates(
        self,
        x0: np.ndarray,
        model: ModelInterface,
        target_class: int,
    ) -> Tuple[np.ndarray, float, int]:
        candidates = self._initialize_candidates(x0)
        prev_loss = np.inf
        converge_count = 0

        for itr in range(self.max_iter):
            loss_now = self._objective(candidates=candidates, x0=x0, model=model, target_class=target_class)

            grad = self._es_gradient(candidates=candidates, x0=x0, model=model, target_class=target_class)
            candidates = candidates - self.learning_rate * grad
            candidates = np.clip(candidates, self._x_min[None, :], self._x_max[None, :])

            loss_diff = abs(prev_loss - loss_now)
            prev_loss = loss_now

            if itr + 1 < self.min_iter:
                continue

            if loss_diff <= self.loss_diff_thres:
                converge_count += 1
            else:
                converge_count = 0

            if converge_count >= self.loss_converge_maxiter:
                valid_mask, _ = self._validity_mask(candidates, model=model, target_class=target_class)
                if bool(np.any(valid_mask)):
                    return candidates, float(loss_now), itr + 1

        final_loss = self._objective(candidates=candidates, x0=x0, model=model, target_class=target_class)
        return candidates, float(final_loss), self.max_iter

    def _initialize_candidates(self, x0: np.ndarray) -> np.ndarray:
        if self.init_near_query_instance:
            noise = self.rng.normal(
                loc=0.0,
                scale=self.init_noise_scale,
                size=(self.total_cfs, x0.shape[0]),
            ).astype(np.float32)
            cfs = x0[None, :] + noise * self._feature_scale[None, :]
        else:
            low = self._x_min[None, :]
            high = self._x_max[None, :]
            cfs = self.rng.uniform(low=low, high=high, size=(self.total_cfs, x0.shape[0])).astype(np.float32)
        cfs = np.clip(cfs, self._x_min[None, :], self._x_max[None, :])
        return cfs.astype(np.float32)

    def _es_gradient(
        self,
        candidates: np.ndarray,
        x0: np.ndarray,
        model: ModelInterface,
        target_class: int,
    ) -> np.ndarray:
        grad = np.zeros_like(candidates, dtype=np.float32)
        sigma = self.es_sigma

        for _ in range(self.es_num_directions):
            direction = self.rng.normal(0.0, 1.0, size=candidates.shape).astype(np.float32)
            plus = np.clip(candidates + sigma * direction, self._x_min[None, :], self._x_max[None, :])
            minus = np.clip(candidates - sigma * direction, self._x_min[None, :], self._x_max[None, :])

            loss_plus = self._objective(candidates=plus, x0=x0, model=model, target_class=target_class)
            loss_minus = self._objective(candidates=minus, x0=x0, model=model, target_class=target_class)
            grad += ((loss_plus - loss_minus) / (2.0 * sigma)) * direction

        grad /= float(self.es_num_directions)
        return grad

    def _objective(
        self,
        candidates: np.ndarray,
        x0: np.ndarray,
        model: ModelInterface,
        target_class: int,
    ) -> float:
        yloss = self._y_loss(candidates=candidates, model=model, target_class=target_class)
        proximity = float(np.mean(self._proximity_loss(candidates, x0)))
        diversity = self._diversity(candidates)
        return float(yloss + self.proximity_weight * proximity - self.diversity_weight * diversity)

    def _y_loss(self, candidates: np.ndarray, model: ModelInterface, target_class: int) -> float:
        probs = model.predict_proba(candidates)
        target_prob = np.clip(probs[:, target_class], 1e-7, 1.0 - 1e-7)

        threshold = self._effective_stopping_threshold(target_class)
        if self.yloss_type == "hinge_loss":
            if target_class == 1:
                return float(np.mean(np.maximum(0.0, threshold - target_prob)))
            return float(np.mean(np.maximum(0.0, target_prob - threshold)))
        if self.yloss_type == "log_loss":
            return float(np.mean(-np.log(target_prob)))
        # l2_loss
        target_val = 1.0 if target_class == 1 else 0.0
        return float(np.mean((target_prob - target_val) ** 2))

    def _proximity_loss(self, candidates: np.ndarray, x0: np.ndarray) -> np.ndarray:
        return np.linalg.norm((candidates - x0[None, :]) / self._feature_scale[None, :], ord=1, axis=1)

    def _diversity(self, candidates: np.ndarray) -> float:
        if candidates.shape[0] <= 1:
            return 0.0
        scaled = candidates / self._feature_scale[None, :]
        diffs = scaled[:, None, :] - scaled[None, :, :]
        dists = np.linalg.norm(diffs, ord=2, axis=2)
        tri = np.triu_indices(candidates.shape[0], k=1)
        pairwise = dists[tri]
        return float(np.mean(pairwise)) if pairwise.size > 0 else 0.0

    def _validity_mask(
        self,
        candidates: np.ndarray,
        model: ModelInterface,
        target_class: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probs = model.predict_proba(candidates)
        target_prob = probs[:, target_class]
        thr = self._effective_stopping_threshold(target_class)
        if target_class == 1:
            return target_prob >= thr, target_prob
        return target_prob <= thr, target_prob

    def _effective_stopping_threshold(self, target_class: int) -> float:
        thr = self.stopping_threshold
        if target_class == 0 and thr > 0.5:
            return 0.25
        if target_class == 1 and thr < 0.5:
            return 0.75
        return thr

    def _select_best_valid(self, valid: np.ndarray, x0: np.ndarray) -> np.ndarray:
        prox = self._proximity_loss(valid, x0)
        idx = int(np.argmin(prox))
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
