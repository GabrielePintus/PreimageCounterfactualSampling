"""Gradient-based DiCE implementation for differentiable torch models.

This implementation follows the original DiCE paper and the official
``dice_ml`` PyTorch backend closely for binary tabular models:
- optimize continuous and one-hot features in a normalized [0, 1] space,
- use the original y-loss / proximity / diversity objective,
- add the categorical simplex regularizer used for one-hot blocks,
- round categorical blocks back to valid one-hot vectors,
- apply post-hoc sparsity only to continuous features.

The Adult benchmark in this repository uses a checkpointed torch model with
identity preprocessing, which matches the assumptions of the original method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from counterfactuals.core.base_classes import CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface
from counterfactuals.preprocessing.transforms import IdentityTransform, InverseTransformModel, OHEBlockSpec

from .base_method import ProbabilisticMethod

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - benchmark environment installs torch
    torch = None
    F = None


@dataclass(frozen=True)
class _DifferentiableModel:
    """Torch model metadata needed by the DiCE optimisation loop."""

    module: "torch.nn.Module"
    device: "torch.device"
    ohe_blocks: tuple[tuple[int, int], ...]


class DiceMethod(ProbabilisticMethod):
    """Generate counterfactuals with the original gradient-based DiCE objective."""

    def __init__(
        self,
        model: ModelInterface,
        total_cfs: int = 4,
        num_candidates: Optional[int] = None,
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
        categorical_penalty: float = 0.1,
        yloss_type: str = "hinge_loss",
        diversity_loss_type: str = "dpp_style:inverse_dist",
        feature_weights: str | Sequence[float] = "inverse_mad",
        optimizer: str = "pytorch:adam",
        learning_rate: float = 0.05,
        min_iter: int = 500,
        max_iter: int = 5000,
        project_iter: int = 0,
        loss_diff_thres: float = 1e-5,
        loss_converge_maxiter: int = 1,
        stopping_threshold: float = 0.5,
        init_near_query_instance: bool = True,
        init_noise_scale: float = 0.01,
        tie_random: bool = False,
        posthoc_sparsity_param: float = 0.1,
        posthoc_sparsity_algorithm: str = "linear",
        limit_steps_ls: int = 10000,
        es_num_directions: Optional[int] = None,
        es_sigma: Optional[float] = None,
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)
        if num_candidates is not None and total_cfs == 4:
            total_cfs = int(num_candidates)
        self.total_cfs = int(total_cfs)
        self.proximity_weight = float(proximity_weight)
        self.diversity_weight = float(diversity_weight)
        self.categorical_penalty = float(categorical_penalty)
        self.yloss_type = str(yloss_type)
        self.diversity_loss_type = str(diversity_loss_type)
        self.feature_weights = feature_weights
        self.optimizer = str(optimizer)
        self.learning_rate = float(learning_rate)
        self.min_iter = int(min_iter)
        self.max_iter = int(max_iter)
        self.project_iter = int(project_iter)
        self.loss_diff_thres = float(loss_diff_thres)
        self.loss_converge_maxiter = int(loss_converge_maxiter)
        self.stopping_threshold = float(stopping_threshold)
        self.init_near_query_instance = bool(init_near_query_instance)
        self.init_noise_scale = float(init_noise_scale)
        self.tie_random = bool(tie_random)
        self.posthoc_sparsity_param = float(posthoc_sparsity_param)
        self.posthoc_sparsity_algorithm = str(posthoc_sparsity_algorithm)
        self.limit_steps_ls = int(limit_steps_ls)

        # Backward-compatible no-op parameters from the previous ES approximation.
        self.es_num_directions = es_num_directions
        self.es_sigma = es_sigma

        if self.total_cfs <= 0:
            raise ValueError("total_cfs must be > 0")
        if self.yloss_type not in {"hinge_loss", "log_loss", "l2_loss"}:
            raise ValueError("yloss_type must be one of: hinge_loss, log_loss, l2_loss")
        if self.diversity_loss_type not in {"dpp_style:inverse_dist", "avg_dist"}:
            raise ValueError("diversity_loss_type must be 'dpp_style:inverse_dist' or 'avg_dist'")
        if self.optimizer not in {"pytorch:adam", "pytorch:rmsprop"}:
            raise ValueError("optimizer must be 'pytorch:adam' or 'pytorch:rmsprop'")
        if self.posthoc_sparsity_algorithm not in {"linear", "binary"}:
            raise ValueError("posthoc_sparsity_algorithm must be 'linear' or 'binary'")

        self._diff_model: Optional[_DifferentiableModel] = None
        self._x_train_eval: Optional[np.ndarray] = None
        self._minx: Optional[np.ndarray] = None
        self._maxx: Optional[np.ndarray] = None
        self._ranges: Optional[np.ndarray] = None
        self._feature_weights_arr: Optional[np.ndarray] = None
        self._continuous_indices: Optional[np.ndarray] = None
        self._continuous_steps: Optional[np.ndarray] = None
        self._sparsity_thresholds: Optional[np.ndarray] = None

    def _fit(self) -> None:
        if torch is None or F is None:
            raise ImportError("DiceMethod requires torch in this environment.")

        assert self._x_train is not None
        if self._x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array")

        self._diff_model = self._unwrap_model(model=self.model, n_features=self._x_train.shape[1])
        self._x_train_eval = self._x_train
        self._minx = np.min(self._x_train, axis=0).astype(np.float32)
        self._maxx = np.max(self._x_train, axis=0).astype(np.float32)

        # Continuous features are normalised from their observed train range.
        # Categorical OHE dimensions are forced to [0, 1] even if a category is
        # absent from the sampled training subset.
        cat_mask = np.zeros(self._x_train.shape[1], dtype=bool)
        for start, end in self._diff_model.ohe_blocks:
            cat_mask[start:end] = True
        if np.any(cat_mask):
            self._minx[cat_mask] = 0.0
            self._maxx[cat_mask] = 1.0
        self._ranges = np.maximum(self._maxx - self._minx, 1e-12).astype(np.float32)
        self._continuous_indices = np.flatnonzero(~cat_mask).astype(np.int64)

        self._feature_weights_arr = self._compute_feature_weights(x_train=self._x_train)
        self._continuous_steps = self._compute_continuous_steps(x_train=self._x_train)
        self._sparsity_thresholds = self._compute_sparsity_thresholds(x_train=self._x_train)

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        if not self._is_fitted or self._diff_model is None:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        # DiCE optimises in normalised feature space, but the benchmark and final
        # model validation both happen in evaluation space after rounding.
        x_query_eval = self._as_1d(x).astype(np.float32)
        target_class = self._resolve_target_class(x=x_query_eval, target_class=target_class)
        query_norm = self._normalize(x_query_eval)

        candidate_norm = torch.nn.Parameter(
            torch.tensor(
                self._initialize_candidates(query_norm),
                dtype=torch.float32,
                device=self._diff_model.device,
            )
        )
        optimizer = self._build_optimizer([candidate_norm])

        prev_loss = 0.0
        converge_count = 0
        best_backup_eval: Optional[np.ndarray] = None
        best_backup_gap = np.inf
        final_loss = np.inf
        n_iter = self.max_iter

        for itr in range(self.max_iter):
            # The optimisation variable stays continuous during gradient updates;
            # projection to valid OHE vertices only happens for evaluation/backup.
            optimizer.zero_grad()
            cfs_norm = torch.clamp(candidate_norm, 0.0, 1.0)
            probs = self._predict_proba_from_norm(cfs_norm)
            loss = self._objective(cfs_norm=cfs_norm, probs=probs, query_norm=query_norm, target_class=target_class)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                candidate_norm.data.clamp_(0.0, 1.0)

            if self.project_iter > 0 and (itr + 1) % self.project_iter == 0:
                rounded_eval = self._round_cfs_to_eval(candidate_norm.detach().cpu().numpy())
                rounded_norm = self._normalize(rounded_eval)
                candidate_norm.data.copy_(
                    torch.tensor(rounded_norm, dtype=torch.float32, device=self._diff_model.device)
                )

            final_loss = float(loss.detach().cpu().item())
            loss_diff = abs(prev_loss - final_loss)
            prev_loss = final_loss

            # Track the best fully valid rounded batch as a backup in case the final
            # iterate is not valid after categorical snapping.
            rounded_eval = self._round_cfs_to_eval(candidate_norm.detach().cpu().numpy())
            valid_mask, probs_target = self._validity_mask_eval(rounded_eval, target_class=target_class)
            if np.all(valid_mask):
                gap = float(np.mean(np.abs(probs_target - self._effective_stopping_threshold(target_class))))
                if gap < best_backup_gap:
                    best_backup_gap = gap
                    best_backup_eval = rounded_eval.copy()

            if itr + 1 < self.min_iter:
                continue

            if loss_diff <= self.loss_diff_thres:
                converge_count += 1
            else:
                converge_count = 0

            if converge_count >= self.loss_converge_maxiter and np.all(valid_mask):
                n_iter = itr + 1
                break

        candidates_eval = self._round_cfs_to_eval(candidate_norm.detach().cpu().numpy())
        valid_mask, probs_target = self._validity_mask_eval(candidates_eval, target_class=target_class)
        if not np.all(valid_mask) and best_backup_eval is not None:
            candidates_eval = best_backup_eval
            valid_mask, probs_target = self._validity_mask_eval(candidates_eval, target_class=target_class)

        if np.any(valid_mask):
            valid_eval = candidates_eval[valid_mask]
            # Post-hoc sparsity is only applied to already-valid counterfactuals,
            # matching the original DiCE workflow.
            valid_eval = self._apply_posthoc_sparsity(
                candidates_eval=valid_eval,
                x_query_eval=x_query_eval,
                target_class=target_class,
            )
            valid_mask_post, probs_target_post = self._validity_mask_eval(valid_eval, target_class=target_class)
            if np.any(valid_mask_post):
                valid_eval = valid_eval[valid_mask_post]
                probs_target = probs_target_post[valid_mask_post]
            else:
                probs_target = np.asarray([], dtype=np.float32)
        else:
            valid_eval = np.empty((0, x_query_eval.shape[0]), dtype=np.float32)

        if valid_eval.shape[0] == 0:
            fallback = candidates_eval[0].astype(np.float32)
            return CounterfactualResult(
                x_cf=fallback,
                success=False,
                distance=float(np.linalg.norm(fallback - x_query_eval, ord=2)),
                metadata={
                    "target_class": target_class,
                    "reason": "no_valid_candidate",
                    "optimization_loss": final_loss,
                    "iterations": int(n_iter),
                },
            )

        selected = self._select_best_valid(valid_eval=valid_eval, x_query_eval=x_query_eval)
        distance = float(np.linalg.norm(selected - x_query_eval, ord=2))
        return CounterfactualResult(
            x_cf=selected.astype(np.float32),
            success=True,
            distance=distance,
            metadata={
                "target_class": target_class,
                "n_valid": int(valid_eval.shape[0]),
                "n_candidates": int(candidates_eval.shape[0]),
                "optimization_loss": final_loss,
                "iterations": int(n_iter),
                "mean_target_proba": float(np.mean(probs_target)) if probs_target.size else float("nan"),
            },
        )

    def _predict_model(self) -> ModelInterface:
        """Minimal model interface needed for target-class resolution."""
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._predict_proba_eval(np.asarray(x, dtype=np.float32))

    def _unwrap_model(self, model: ModelInterface, n_features: int) -> _DifferentiableModel:
        base_model = model
        ohe_blocks: Sequence[OHEBlockSpec] | None = None
        if isinstance(model, InverseTransformModel):
            # This implementation is intentionally strict: it only matches the
            # original DiCE formulation when generation and evaluation spaces match.
            if not isinstance(model.transform, IdentityTransform):
                raise ValueError(
                    "DiceMethod matches the original DiCE formulation only with identity preprocessing."
                )
            base_model = model.base_model
            ohe_blocks = model.ohe_blocks

        module = getattr(base_model, "model", None)
        if torch is None or not isinstance(module, torch.nn.Module):
            raise TypeError(
                "DiceMethod requires a differentiable torch model (TorchModelWrapper or InverseTransformModel)."
            )

        device = getattr(base_model, "device", None)
        if device is None:
            param = next(module.parameters(), None)
            device = param.device if param is not None else torch.device("cpu")
        device = torch.device(device)

        blocks: list[tuple[int, int]] = []
        for block in ohe_blocks or ():
            if hasattr(block, "start") and hasattr(block, "end"):
                start = int(block.start)
                end = int(block.end)
            else:
                start = int(block[0])
                end = int(block[1])
            blocks.append((start, end))

        for start, end in blocks:
            if start < 0 or end > n_features or start >= end:
                raise ValueError("Invalid OHE block metadata for DiceMethod.")

        module = module.eval().to(device)
        return _DifferentiableModel(module=module, device=device, ohe_blocks=tuple(blocks))

    def _compute_feature_weights(self, x_train: np.ndarray) -> np.ndarray:
        assert self._continuous_indices is not None
        if isinstance(self.feature_weights, str):
            if self.feature_weights != "inverse_mad":
                raise ValueError("Only feature_weights='inverse_mad' is supported in this DiCE implementation.")
            weights = np.ones(x_train.shape[1], dtype=np.float32)
            if self._continuous_indices.size > 0:
                # Continuous features follow DiCE's inverse-MAD weighting in the
                # normalised space; categorical OHE dimensions keep unit weight.
                x_train_norm = self._normalize(x_train)
                cont = x_train_norm[:, self._continuous_indices]
                mad = np.median(np.abs(cont - np.median(cont, axis=0)), axis=0)
                mad[mad < 1e-12] = 1.0
                weights[self._continuous_indices] = np.round(1.0 / mad, 2).astype(np.float32)
            return weights

        weights_arr = np.asarray(self.feature_weights, dtype=np.float32)
        if weights_arr.shape != (x_train.shape[1],):
            raise ValueError("feature_weights must have shape (n_features,)")
        return weights_arr

    def _compute_continuous_steps(self, x_train: np.ndarray) -> np.ndarray:
        assert self._continuous_indices is not None
        steps = np.zeros(x_train.shape[1], dtype=np.float32)
        for idx in self._continuous_indices:
            unique = np.unique(np.asarray(x_train[:, idx], dtype=np.float64))
            diffs = np.diff(unique)
            diffs = diffs[diffs > 1e-12]
            if diffs.size > 0:
                steps[idx] = float(np.min(diffs))
            else:
                steps[idx] = 0.0
        return steps

    def _compute_sparsity_thresholds(self, x_train: np.ndarray) -> np.ndarray:
        assert self._continuous_indices is not None
        thresholds = np.zeros(x_train.shape[1], dtype=np.float32)
        q = float(np.clip(self.posthoc_sparsity_param, 0.0, 1.0))
        if q <= 0.0:
            return thresholds
        for idx in self._continuous_indices:
            col = np.asarray(x_train[:, idx], dtype=np.float64)
            median = float(np.median(col))
            mad = float(np.median(np.abs(col - median)))
            # DiCE's post-hoc sparsity threshold uses the smaller of MAD and a low
            # percentile of non-identical values around the median.
            non_identical = np.unique(col[col != median])
            if non_identical.size > 0:
                quantile = float(np.quantile(np.abs(non_identical - median), q))
                thresholds[idx] = float(min(mad, quantile))
            else:
                thresholds[idx] = mad
        return thresholds

    def _normalize(self, x_eval: np.ndarray) -> np.ndarray:
        assert self._minx is not None and self._ranges is not None
        arr = np.asarray(x_eval, dtype=np.float32)
        return ((arr - self._minx) / self._ranges).astype(np.float32)

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        assert self._minx is not None and self._ranges is not None
        arr = np.asarray(x_norm, dtype=np.float32)
        return (arr * self._ranges + self._minx).astype(np.float32)

    def _initialize_candidates(self, query_norm: np.ndarray) -> np.ndarray:
        if self.init_near_query_instance:
            # The official implementation starts near the query by adding a small
            # deterministic offset to each CF candidate.
            offsets = (np.arange(self.total_cfs, dtype=np.float32) * 0.01)[:, None]
            candidates = query_norm[None, :] + offsets
        else:
            candidates = self.rng.uniform(
                low=0.0,
                high=1.0,
                size=(self.total_cfs, query_norm.shape[0]),
            ).astype(np.float32)
        return np.clip(candidates, 0.0, 1.0).astype(np.float32)

    def _build_optimizer(self, params: Sequence["torch.nn.Parameter"]):
        assert torch is not None
        if self.optimizer == "pytorch:rmsprop":
            return torch.optim.RMSprop(params, lr=self.learning_rate)
        return torch.optim.Adam(params, lr=self.learning_rate)

    def _predict_proba_from_norm(self, cfs_norm: "torch.Tensor") -> "torch.Tensor":
        assert torch is not None and self._diff_model is not None
        # The network is trained in evaluation space, so every optimisation step
        # denormalises candidates before running the torch model.
        cfs_eval = cfs_norm * torch.tensor(self._ranges, device=self._diff_model.device) + torch.tensor(
            self._minx, device=self._diff_model.device
        )
        logits = self._diff_model.module(cfs_eval)
        if logits.ndim == 1 or logits.shape[1] == 1:
            p1 = torch.sigmoid(logits.reshape(-1))
            return torch.stack([1.0 - p1, p1], dim=1)
        return torch.softmax(logits, dim=1)

    def _predict_proba_eval(self, cfs_eval: np.ndarray) -> np.ndarray:
        assert torch is not None and self._diff_model is not None
        arr = np.asarray(cfs_eval, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        with torch.no_grad():
            tensor = torch.tensor(arr, dtype=torch.float32, device=self._diff_model.device)
            logits = self._diff_model.module(tensor)
            if logits.ndim == 1 or logits.shape[1] == 1:
                p1 = torch.sigmoid(logits.reshape(-1))
                probs = torch.stack([1.0 - p1, p1], dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float32)

    def _objective(
        self,
        cfs_norm: "torch.Tensor",
        probs: "torch.Tensor",
        query_norm: np.ndarray,
        target_class: int,
    ) -> "torch.Tensor":
        assert torch is not None and F is not None and self._feature_weights_arr is not None
        # Final objective = target validity + weighted proximity - diversity +
        # categorical simplex regularisation.
        target_loss = self._yloss(probs=probs, target_class=target_class)
        query = torch.tensor(query_norm, dtype=torch.float32, device=cfs_norm.device)
        weights = torch.tensor(self._feature_weights_arr, dtype=torch.float32, device=cfs_norm.device)
        proximity = torch.sum(torch.abs(cfs_norm - query.unsqueeze(0)) * weights.unsqueeze(0), dim=1)
        proximity = torch.mean(proximity) / float(query.shape[0])
        diversity = self._diversity(cfs_norm=cfs_norm, weights=weights)
        regularization = self._categorical_regularizer(cfs_norm=cfs_norm)
        return target_loss + self.proximity_weight * proximity - self.diversity_weight * diversity + self.categorical_penalty * regularization

    def _yloss(self, probs: "torch.Tensor", target_class: int) -> "torch.Tensor":
        assert torch is not None and F is not None
        if probs.shape[1] == 2:
            p_pos = torch.clamp(probs[:, 1], 1e-7, 1.0 - 1e-7)
            target_val = 1.0 if target_class == 1 else 0.0
            if self.yloss_type == "hinge_loss":
                logits = torch.log(p_pos / (1.0 - p_pos))
                y_signed = 1.0 if target_class == 1 else -1.0
                return torch.mean(F.relu(1.0 - y_signed * logits))
            if self.yloss_type == "log_loss":
                target = torch.full_like(p_pos, target_val)
                return torch.mean(
                    -(target * torch.log(p_pos) + (1.0 - target) * torch.log(1.0 - p_pos))
                )
            target = torch.full_like(p_pos, target_val)
            return torch.mean((p_pos - target) ** 2)

        target_prob = torch.clamp(probs[:, target_class], 1e-7, 1.0 - 1e-7)
        if self.yloss_type == "hinge_loss":
            logits = torch.log(target_prob / (1.0 - target_prob))
            return torch.mean(F.relu(1.0 - logits))
        if self.yloss_type == "log_loss":
            return torch.mean(-torch.log(target_prob))
        return torch.mean((target_prob - 1.0) ** 2)

    def _diversity(self, cfs_norm: "torch.Tensor", weights: "torch.Tensor") -> "torch.Tensor":
        assert torch is not None
        if cfs_norm.shape[0] <= 1:
            return torch.tensor(0.0, device=cfs_norm.device)
        if self.diversity_loss_type == "avg_dist":
            # This branch mirrors the official alternative "average distance"
            # diversity loss, expressed as 1 - mean similarity.
            pairwise = torch.sum(
                torch.abs(cfs_norm.unsqueeze(1) - cfs_norm.unsqueeze(0)) * weights.view(1, 1, -1),
                dim=2,
            )
            sims = 1.0 / (1.0 + pairwise)
            n = cfs_norm.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=cfs_norm.device)
            return 1.0 - torch.mean(sims[mask])

        pairwise = torch.sum(
            torch.abs(cfs_norm.unsqueeze(1) - cfs_norm.unsqueeze(0)) * weights.view(1, 1, -1),
            dim=2,
        )
        kernel = 1.0 / (1.0 + pairwise)
        kernel = kernel + 1e-4 * torch.eye(kernel.shape[0], device=kernel.device)
        return torch.det(kernel)

    def _categorical_regularizer(self, cfs_norm: "torch.Tensor") -> "torch.Tensor":
        assert torch is not None and self._diff_model is not None
        if not self._diff_model.ohe_blocks:
            return torch.tensor(0.0, device=cfs_norm.device)
        penalty = torch.tensor(0.0, device=cfs_norm.device)
        for start, end in self._diff_model.ohe_blocks:
            penalty = penalty + torch.sum((torch.sum(cfs_norm[:, start:end], dim=1) - 1.0) ** 2)
        return penalty

    def _round_cfs_to_eval(self, cfs_norm: np.ndarray) -> np.ndarray:
        assert self._diff_model is not None
        cfs_norm = np.clip(np.asarray(cfs_norm, dtype=np.float32), 0.0, 1.0)
        if cfs_norm.ndim == 1:
            cfs_norm = cfs_norm[None, :]
        cfs_eval = self._denormalize(cfs_norm)

        # Snap each one-hot block to a valid vertex, matching the official DiCE
        # output step after continuous optimisation.
        for start, end in self._diff_model.ohe_blocks:
            block = cfs_eval[:, start:end]
            if self.tie_random:
                ties = block == block.max(axis=1, keepdims=True)
                winners = np.array([self.rng.choice(np.flatnonzero(row)) for row in ties], dtype=np.int64)
            else:
                winners = np.argmax(block, axis=1)
            block.fill(0.0)
            block[np.arange(block.shape[0]), winners] = 1.0
            cfs_eval[:, start:end] = block

        return cfs_eval.astype(np.float32)

    def _validity_mask_eval(self, candidates_eval: np.ndarray, target_class: int) -> tuple[np.ndarray, np.ndarray]:
        probs = self._predict_proba_eval(candidates_eval)
        thr = self._effective_stopping_threshold(target_class)
        if probs.shape[1] == 2:
            p_pos = probs[:, 1]
            if target_class == 1:
                return p_pos >= thr, p_pos
            return p_pos <= thr, p_pos
        target_prob = probs[:, target_class]
        return target_prob >= thr, target_prob

    def _effective_stopping_threshold(self, target_class: int) -> float:
        thr = self.stopping_threshold
        if target_class == 0 and thr >= 0.5:
            return 0.25
        if target_class == 1 and thr < 0.5:
            return 0.75
        return thr

    def _apply_posthoc_sparsity(
        self,
        candidates_eval: np.ndarray,
        x_query_eval: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        assert self._continuous_indices is not None and self._sparsity_thresholds is not None
        if self.posthoc_sparsity_param <= 0.0 or self._continuous_indices.size == 0:
            return candidates_eval.astype(np.float32)

        # Continuous features are revisited from the loosest threshold to the
        # tightest, restoring query values whenever validity is preserved.
        feature_order = self._continuous_indices[np.argsort(self._sparsity_thresholds[self._continuous_indices])[::-1]]
        improved = np.asarray(candidates_eval, dtype=np.float32).copy()
        for row_idx in range(improved.shape[0]):
            current = improved[row_idx].copy()
            for feat_idx in feature_order:
                if np.abs(current[feat_idx] - x_query_eval[feat_idx]) > self._sparsity_thresholds[feat_idx]:
                    continue
                if self.posthoc_sparsity_algorithm == "binary":
                    trial = self._binary_search_feature(current, x_query_eval, feat_idx, target_class)
                else:
                    trial = self._linear_search_feature(current, x_query_eval, feat_idx, target_class)
                current = trial
            improved[row_idx] = current
        return improved.astype(np.float32)

    def _linear_search_feature(
        self,
        candidate_eval: np.ndarray,
        x_query_eval: np.ndarray,
        feat_idx: int,
        target_class: int,
    ) -> np.ndarray:
        current = candidate_eval.copy()
        step = float(self._continuous_linear_step(feat_idx))
        if step <= 0.0:
            return current

        # Move the feature back toward the query in small increments until the CF
        # would leave the target class.
        query_val = float(x_query_eval[feat_idx])
        direction = np.sign(query_val - float(current[feat_idx]))
        if direction == 0.0:
            return current

        best = current.copy()
        for _ in range(self.limit_steps_ls):
            proposal = best.copy()
            proposal[feat_idx] += direction * step
            if (direction > 0 and proposal[feat_idx] > query_val) or (direction < 0 and proposal[feat_idx] < query_val):
                proposal[feat_idx] = query_val
            valid, _ = self._validity_mask_eval(proposal[None, :], target_class=target_class)
            if not bool(valid[0]):
                break
            best = proposal
            if proposal[feat_idx] == query_val:
                break
        return best.astype(np.float32)

    def _binary_search_feature(
        self,
        candidate_eval: np.ndarray,
        x_query_eval: np.ndarray,
        feat_idx: int,
        target_class: int,
    ) -> np.ndarray:
        current = candidate_eval.copy()
        query_val = float(x_query_eval[feat_idx])

        # Fast path: if resetting the feature all the way back to the query keeps
        # the counterfactual valid, no search is needed.
        proposal = current.copy()
        proposal[feat_idx] = query_val
        valid, _ = self._validity_mask_eval(proposal[None, :], target_class=target_class)
        if bool(valid[0]):
            return proposal.astype(np.float32)

        # Otherwise search along the segment between the current CF value and the
        # original query value, using decimal precision derived from the feature range.
        decimal_prec = self._continuous_decimal_precision(feat_idx)
        change = 10.0 ** (-decimal_prec)
        diff = query_val - float(current[feat_idx])
        if diff > 0:
            left = float(current[feat_idx])
            right = query_val
            while left <= right:
                current_val = round(left + ((right - left) / 2.0), decimal_prec)
                proposal = current.copy()
                proposal[feat_idx] = current_val
                valid, _ = self._validity_mask_eval(proposal[None, :], target_class=target_class)
                if current_val == right or current_val == left:
                    break
                if bool(valid[0]):
                    current = proposal
                    left = current_val + change
                else:
                    right = current_val - change
        else:
            left = query_val
            right = float(current[feat_idx])
            while right >= left:
                current_val = round(right - ((right - left) / 2.0), decimal_prec)
                proposal = current.copy()
                proposal[feat_idx] = current_val
                valid, _ = self._validity_mask_eval(proposal[None, :], target_class=target_class)
                if current_val == right or current_val == left:
                    break
                if bool(valid[0]):
                    current = proposal
                    right = current_val - change
                else:
                    left = current_val + change
        return current.astype(np.float32)

    def _continuous_decimal_precision(self, feat_idx: int) -> int:
        assert self._ranges is not None
        feat_range = float(self._ranges[feat_idx])
        if feat_range <= 0.0:
            return 6
        return int(max(0, min(6, np.ceil(-np.log10(feat_range / 1000.0 + 1e-12)))))

    def _continuous_linear_step(self, feat_idx: int) -> float:
        decimal_prec = self._continuous_decimal_precision(feat_idx)
        return float(10.0 ** (-decimal_prec))

    def _select_best_valid(self, valid_eval: np.ndarray, x_query_eval: np.ndarray) -> np.ndarray:
        assert self._feature_weights_arr is not None
        valid_norm = self._normalize(valid_eval)
        query_norm = self._normalize(x_query_eval)
        weighted_l1 = np.sum(np.abs(valid_norm - query_norm[None, :]) * self._feature_weights_arr[None, :], axis=1)
        idx = int(np.argmin(weighted_l1))
        return valid_eval[idx]

    def _resolve_target_class(self, x: np.ndarray, target_class: Optional[int]) -> int:
        if target_class is not None:
            return int(target_class)
        pred = int(self._predict_model().predict(x)[0])
        if self._predict_model().predict_proba(x).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
