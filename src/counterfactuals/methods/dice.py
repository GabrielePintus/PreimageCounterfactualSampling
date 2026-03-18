"""Gradient-based DiCE implementation for differentiable torch models.

This implementation follows the original DiCE paper and the official
``dice_ml`` PyTorch backend closely for binary tabular models:
- use the original y-loss / proximity / diversity objective,
- add the categorical simplex regularizer used for one-hot blocks,
- round categorical blocks back to valid one-hot vectors,
- apply post-hoc sparsity only to continuous features.

The Adult benchmark in this repository uses a checkpointed torch model with
StandardScaler-preprocessed numerical features and binary OHE features,
so no additional [0, 1] normalization is needed.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from counterfactuals.core.base_classes import CounterfactualResult
from counterfactuals.core.interfaces import ModelInterface
from counterfactuals.preprocessing.transforms import IdentityTransform, InverseTransformModel, OHEBlockSpec

from counterfactuals.core.base_classes import BaseCounterfactualMethod

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - benchmark environment installs torch
    torch = None
    F = None



class DiceMethod(BaseCounterfactualMethod):
    """Generate counterfactuals with the original gradient-based DiCE objective."""

    def __init__(
        self,
        # The trained classifier
        model: ModelInterface,

        # Main DiCE objective weights and losses
        total_cfs: int = 4,
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
        categorical_penalty: float = 0.1,
        # Optimizer
        learning_rate: float = 0.05,

        # Convergence / stopping
        min_iter: int = 500,
        max_iter: int = 5000,
        loss_diff_thres: float = 1e-5,
        loss_converge_maxiter: int = 1,

        # Candidate initialization
        tie_random: bool = False,

        # Post-hoc sparsity
        posthoc_sparsity_param: float = 0.1,
        posthoc_sparsity_algorithm: str = "linear",
        limit_steps_ls: int = 10000,

        # Random seed for reproducibility
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed)

        # Main DiCE objective weights and losses
        self.total_cfs = total_cfs
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty
        # Optimizer
        self.learning_rate = learning_rate

        # Convergence / stopping
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.loss_diff_thres = loss_diff_thres
        self.loss_converge_maxiter = loss_converge_maxiter

        # Candidate initialization
        self.tie_random = tie_random

        # Post-hoc sparsity
        self.posthoc_sparsity_param = posthoc_sparsity_param
        self.posthoc_sparsity_algorithm = posthoc_sparsity_algorithm
        self.limit_steps_ls = limit_steps_ls

        if self.total_cfs <= 0:
            raise ValueError("total_cfs must be > 0")
        if self.posthoc_sparsity_algorithm not in {"linear", "binary"}:
            raise ValueError("posthoc_sparsity_algorithm must be 'linear' or 'binary'")

        self._module = None
        self._device = None
        self._ohe_blocks: tuple[tuple[int, int], ...] = ()
        self.feature_weights: Optional[np.ndarray] = None
        self._feature_weights_t = None  # on-device tensor version, built in _fit
        self.continuous_indices: Optional[np.ndarray] = None
        self._continuous_steps: Optional[np.ndarray] = None
        self.sparsity_thresholds: Optional[np.ndarray] = None

    def _fit(self) -> None:
        """Extract the differentiable model, compute inverse-MAD weights and sparsity thresholds from training data."""
        if self._x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array")

        self._module, self._device, self._ohe_blocks = self._unwrap_model(model=self.model, n_features=self._x_train.shape[1])

        # Identify which feature indices are continuous (complement of OHE blocks)
        cat_mask = np.zeros(self._x_train.shape[1], dtype=bool)
        for start, end in self._ohe_blocks:
            cat_mask[start:end] = True
        self.continuous_indices = np.flatnonzero(~cat_mask).astype(np.int64)

        # Precompute training-data statistics used during generation
        # inverse-MAD per feature, used in proximity and diversity terms
        self.feature_weights = self.compute_feature_weights(x_train=self._x_train)
        self._feature_weights_t = torch.tensor(self.feature_weights, dtype=torch.float32, device=self._device)
        # P10 step size per continuous feature, used in post-hoc sparsity search
        self._continuous_steps = self.compute_continuous_steps(x_train=self._x_train)
        # min(MAD, P10) per feature, threshold for snapping to query value
        self.sparsity_thresholds = self.compute_sparsity_thresholds(x_train=self._x_train)

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        """Run the DiCE gradient loop and return the best valid post-hoc sparse counterfactual."""
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")

        x_query = np.asarray(x, dtype=np.float32).reshape(-1)
        target_class = self.resolve_target_class(x=x_query, target_class=target_class)

        candidates = torch.nn.Parameter(
            torch.tensor(
                self.initialize_candidates(x_query.shape[0]),
                dtype=torch.float32,
                device=self._device,
            )
        )
        optimizer = self.build_optimizer([candidates])

        prev_loss = 0.0
        converge_count = 0
        best_backup_eval = None  # torch.Tensor on self._device, set when all candidates are valid
        final_loss = np.inf
        n_iter = self.max_iter

        for itr in range(self.max_iter):
            # The optimisation variable stays continuous during gradient updates;
            # projection to valid OHE vertices only happens for evaluation/backup.
            optimizer.zero_grad()
            probs = self._predict_proba(candidates)
            loss = self.objective(cfs=candidates, probs=probs, x_query=x_query, target_class=target_class)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.detach().cpu().item())
            loss_diff = abs(prev_loss - final_loss)
            prev_loss = final_loss

            # Track the most recent fully valid rounded batch as a backup in case the
            # final iterate is not valid after categorical snapping. All operations stay
            # on-device; no GPU→CPU transfer happens here.
            rounded_t = self.round_cfs_to_eval_torch(candidates)
            valid_mask_t, _ = self.validity_mask_eval_torch(rounded_t, target_class)
            if valid_mask_t.all():
                best_backup_eval = rounded_t.detach().clone()

            if itr + 1 < self.min_iter:
                continue

            if loss_diff <= self.loss_diff_thres:
                converge_count += 1
            else:
                converge_count = 0

            if converge_count >= self.loss_converge_maxiter and valid_mask_t.all():
                n_iter = itr + 1
                break

        # Resolve the final candidate tensor on-device, then move to CPU once.
        final_t = self.round_cfs_to_eval_torch(candidates)
        valid_mask_t, probs_target_t = self.validity_mask_eval_torch(final_t, target_class)
        if not valid_mask_t.all() and best_backup_eval is not None:
            final_t = best_backup_eval
            valid_mask_t, probs_target_t = self.validity_mask_eval_torch(final_t, target_class)
        candidates_eval = final_t.cpu().numpy()
        valid_mask = valid_mask_t.cpu().numpy()
        probs_target = probs_target_t.cpu().numpy()

        if np.any(valid_mask):
            valid_eval = candidates_eval[valid_mask]
            # Post-hoc sparsity is only applied to already-valid counterfactuals,
            # matching the original DiCE workflow.
            valid_eval = self.apply_posthoc_sparsity(
                candidates_eval=valid_eval,
                x_query=x_query,
                target_class=target_class,
            )
            valid_mask_post, probs_target_post = self.validity_mask_eval(valid_eval, target_class=target_class)
            if np.any(valid_mask_post):
                valid_eval = valid_eval[valid_mask_post]
                probs_target = probs_target_post[valid_mask_post]
            else:
                probs_target = np.asarray([], dtype=np.float32)
        else:
            valid_eval = np.empty((0, x_query.shape[0]), dtype=np.float32)

        if valid_eval.shape[0] == 0:
            fallback = candidates_eval[0].astype(np.float32)
            return CounterfactualResult(
                x_cf=fallback,
                success=False,
                distance=float(np.linalg.norm(fallback - x_query, ord=2)),
                metadata={
                    "target_class": target_class,
                    "reason": "no_valid_candidate",
                    "optimization_loss": final_loss,
                    "iterations": int(n_iter),
                },
            )

        selected = self.select_best_valid(valid_eval=valid_eval, x_query=x_query)
        distance = float(np.linalg.norm(selected - x_query, ord=2))
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

    def _unwrap_model(self, model: ModelInterface, n_features: int) -> tuple:
        """Extract the raw torch module, device, and OHE block specs from the model wrapper."""
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
        return module, device, tuple(blocks)

    def compute_feature_weights(self, x_train: np.ndarray) -> np.ndarray:
        """Compute inverse-MAD weights for continuous features (paper eq. 5); OHE dims get weight 1."""
        assert self.continuous_indices is not None
        weights = np.ones(x_train.shape[1], dtype=np.float32)
        if self.continuous_indices.size > 0:
            cont = x_train[:, self.continuous_indices]
            mad = np.median(np.abs(cont - np.median(cont, axis=0)), axis=0)
            mad[mad < 1e-12] = 1.0
            weights[self.continuous_indices] = np.round(1.0 / mad, 2).astype(np.float32)
        return weights

    def compute_continuous_steps(self, x_train: np.ndarray) -> np.ndarray:
        """Compute the minimum observed unique-value gap per continuous feature, used as the linear sparsity step size."""
        assert self.continuous_indices is not None
        steps = np.zeros(x_train.shape[1], dtype=np.float32)
        for idx in self.continuous_indices:
            unique = np.unique(np.asarray(x_train[:, idx], dtype=np.float64))
            diffs = np.diff(unique)
            diffs = diffs[diffs > 1e-12]
            if diffs.size > 0:
                steps[idx] = float(np.min(diffs))
            else:
                steps[idx] = 0.0
        return steps

    def compute_sparsity_thresholds(self, x_train: np.ndarray) -> np.ndarray:
        """Compute per-feature post-hoc sparsity thresholds as min(MAD, P_q of non-identical distances from median)."""
        assert self.continuous_indices is not None
        thresholds = np.zeros(x_train.shape[1], dtype=np.float32)
        q = float(np.clip(self.posthoc_sparsity_param, 0.0, 1.0))
        if q <= 0.0:
            return thresholds
        for idx in self.continuous_indices:
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

    def initialize_candidates(self, d: int) -> np.ndarray:
        """Draw initial candidate counterfactuals from a standard normal distribution."""
        return self.rng.standard_normal(size=(self.total_cfs, d)).astype(np.float32)

    def build_optimizer(self, params: Sequence):
        """Create the Adam optimizer used to update the candidate tensor (paper's optimizer choice)."""
        assert torch is not None
        return torch.optim.Adam(params, lr=self.learning_rate)

    def _predict_proba(self, cfs):
        """Run the torch module on candidates to get class probabilities (differentiable, used in the optimisation loop)."""
        assert torch is not None
        logits = self._module(cfs)
        if logits.ndim == 1 or logits.shape[1] == 1:
            p1 = torch.sigmoid(logits.reshape(-1))
            return torch.stack([1.0 - p1, p1], dim=1)
        return torch.softmax(logits, dim=1)

    def _predict_proba_torch_no_grad(self, cfs):
        """Run the torch module on an on-device tensor without gradient tracking (used for in-loop validity checks)."""
        assert torch is not None
        with torch.no_grad():
            logits = self._module(cfs)
            if logits.ndim == 1 or logits.shape[1] == 1:
                p1 = torch.sigmoid(logits.reshape(-1))
                return torch.stack([1.0 - p1, p1], dim=1)
            return torch.softmax(logits, dim=1)

    def round_cfs_to_eval_torch(self, cfs):
        """Snap each OHE block to a one-hot vertex via argmax, returning an on-device tensor.

        Operates on a detached clone so the autograd graph of the live candidates is not affected.
        Note: tie_random is not supported in this path; ties are broken by lowest index (argmax default).
        """
        assert torch is not None
        result = cfs.detach().clone()
        for start, end in self._ohe_blocks:
            block = result[:, start:end]
            winners = torch.argmax(block, dim=1, keepdim=True)
            result[:, start:end] = torch.zeros_like(block).scatter_(1, winners, 1.0)
        return result

    def validity_mask_eval_torch(self, rounded, target_class: int):
        """Return (valid_mask, probs_target) as on-device tensors; valid iff argmax(probs) == target_class."""
        probs = self._predict_proba_torch_no_grad(rounded)
        valid_mask = torch.argmax(probs, dim=1) == target_class
        probs_target = probs[:, target_class]
        return valid_mask, probs_target

    def predict_proba_eval(self, cfs_eval: np.ndarray) -> np.ndarray:
        """Run the torch module on evaluation-space candidates without gradient tracking (used for validity checks and backup selection)."""
        assert torch is not None
        arr = np.asarray(cfs_eval, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        with torch.no_grad():
            tensor = torch.tensor(arr, dtype=torch.float32, device=self._device)
            logits = self._module(tensor)
            if logits.ndim == 1 or logits.shape[1] == 1:
                p1 = torch.sigmoid(logits.reshape(-1))
                probs = torch.stack([1.0 - p1, p1], dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float32)

    def objective(self, cfs, probs, x_query: np.ndarray, target_class: int):
        """Full DiCE loss: validity (hinge) + proximity_weight * MAD-weighted L1 - diversity_weight * DPP-det + categorical_penalty * simplex regulariser."""
        assert torch is not None and F is not None and self._feature_weights_t is not None
        target_loss = self.yloss(probs=probs, target_class=target_class)
        query = torch.tensor(x_query, dtype=torch.float32, device=cfs.device)
        weights = self._feature_weights_t
        proximity = torch.sum(torch.abs(cfs - query.unsqueeze(0)) * weights.unsqueeze(0), dim=1)
        d_cont = float(len(self.continuous_indices)) if self.continuous_indices is not None and len(self.continuous_indices) > 0 else 1.0
        proximity = torch.mean(proximity) / d_cont
        diversity = self.diversity(cfs_norm=cfs, weights=weights)
        regularization = self.categorical_regularizer(cfs_norm=cfs)
        return target_loss + self.proximity_weight * proximity - self.diversity_weight * diversity + self.categorical_penalty * regularization

    def yloss(self, probs, target_class: int):
        """Hinge loss on the model log-odds: mean(max(0, 1 - z * logit(p))), z=±1 for binary, z=+1 for multiclass target."""
        assert torch is not None and F is not None
        if probs.shape[1] == 2:
            p_pos = torch.clamp(probs[:, 1], 1e-7, 1.0 - 1e-7)
            logits = torch.log(p_pos / (1.0 - p_pos))
            y_signed = 1.0 if target_class == 1 else -1.0
            return torch.mean(F.relu(1.0 - y_signed * logits))

        target_prob = torch.clamp(probs[:, target_class], 1e-7, 1.0 - 1e-7)
        logits = torch.log(target_prob / (1.0 - target_prob))
        return torch.mean(F.relu(1.0 - logits))

    def diversity(self, cfs_norm, weights):
        """DPP determinant diversity: det(K) where K_ij = 1 / (1 + weighted-L1(c_i, c_j)) (paper eq. 4)."""
        assert torch is not None
        if cfs_norm.shape[0] <= 1:
            return torch.tensor(0.0, device=cfs_norm.device)
        pairwise = torch.sum(
            torch.abs(cfs_norm.unsqueeze(1) - cfs_norm.unsqueeze(0)) * weights.view(1, 1, -1),
            dim=2,
        )
        kernel = 1.0 / (1.0 + pairwise)
        kernel = kernel + 1e-4 * torch.eye(kernel.shape[0], device=kernel.device)
        return torch.det(kernel)

    def categorical_regularizer(self, cfs_norm):
        """Penalise deviation from the OHE simplex: sum((sum(block) - 1)^2) over all OHE blocks."""
        assert torch is not None
        if not self._ohe_blocks:
            return torch.tensor(0.0, device=cfs_norm.device)
        penalty = torch.tensor(0.0, device=cfs_norm.device)
        for start, end in self._ohe_blocks:
            penalty = penalty + torch.sum((torch.sum(cfs_norm[:, start:end], dim=1) - 1.0) ** 2)
        return penalty

    def round_cfs_to_eval(self, cfs: np.ndarray) -> np.ndarray:
        """Snap each OHE block to a valid one-hot vertex via argmax (paper's post-optimisation step)."""
        cfs = np.asarray(cfs, dtype=np.float32)
        if cfs.ndim == 1:
            cfs = cfs[None, :]
        cfs_eval = cfs.copy()

        # Snap each one-hot block to a valid vertex, matching the official DiCE
        # output step after continuous optimisation.
        for start, end in self._ohe_blocks:
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

    def validity_mask_eval(self, candidates_eval: np.ndarray, target_class: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a boolean validity mask and target-class probabilities; valid iff argmax(probs) == target_class."""
        probs = self.predict_proba_eval(candidates_eval)
        target_prob = probs[:, target_class]
        valid = np.argmax(probs, axis=1) == target_class
        return valid, target_prob

    def apply_posthoc_sparsity(
        self,
        candidates_eval: np.ndarray,
        x_query: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """Greedily restore query values for features within the sparsity threshold, from loosest to tightest, while preserving validity."""
        assert self.continuous_indices is not None and self.sparsity_thresholds is not None
        if self.posthoc_sparsity_param <= 0.0 or self.continuous_indices.size == 0:
            return candidates_eval.astype(np.float32)

        # Continuous features are revisited from the loosest threshold to the
        # tightest, restoring query values whenever validity is preserved.
        feature_order = self.continuous_indices[np.argsort(self.sparsity_thresholds[self.continuous_indices])[::-1]]
        improved = np.asarray(candidates_eval, dtype=np.float32).copy()
        for row_idx in range(improved.shape[0]):
            current = improved[row_idx].copy()
            for feat_idx in feature_order:
                if np.abs(current[feat_idx] - x_query[feat_idx]) > self.sparsity_thresholds[feat_idx]:
                    continue
                if self.posthoc_sparsity_algorithm == "binary":
                    trial = self._binary_search_feature(current, x_query, feat_idx, target_class)
                else:
                    trial = self._linear_search_feature(current, x_query, feat_idx, target_class)
                current = trial
            improved[row_idx] = current
        return improved.astype(np.float32)

    def _linear_search_feature(
        self,
        candidate_eval: np.ndarray,
        x_query: np.ndarray,
        feat_idx: int,
        target_class: int,
    ) -> np.ndarray:
        """Move one feature toward the query in discrete steps until validity would be lost."""
        current = candidate_eval.copy()
        step = float(self._continuous_linear_step(feat_idx))
        if step <= 0.0:
            return current

        # Move the feature back toward the query in small increments until the CF
        # would leave the target class.
        query_val = float(x_query[feat_idx])
        direction = np.sign(query_val - float(current[feat_idx]))
        if direction == 0.0:
            return current

        best = current.copy()
        for _ in range(self.limit_steps_ls):
            proposal = best.copy()
            proposal[feat_idx] += direction * step
            if (direction > 0 and proposal[feat_idx] > query_val) or (direction < 0 and proposal[feat_idx] < query_val):
                proposal[feat_idx] = query_val
            valid, _ = self.validity_mask_eval(proposal[None, :], target_class=target_class)
            if not bool(valid[0]):
                break
            best = proposal
            if proposal[feat_idx] == query_val:
                break
        return best.astype(np.float32)

    def _binary_search_feature(
        self,
        candidate_eval: np.ndarray,
        x_query: np.ndarray,
        feat_idx: int,
        target_class: int,
    ) -> np.ndarray:
        """Binary search along the segment [CF value, query value] for the furthest snap point that keeps the CF valid."""
        current = candidate_eval.copy()
        query_val = float(x_query[feat_idx])

        # Fast path: if resetting the feature all the way back to the query keeps
        # the counterfactual valid, no search is needed.
        proposal = current.copy()
        proposal[feat_idx] = query_val
        valid, _ = self.validity_mask_eval(proposal[None, :], target_class=target_class)
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
                valid, _ = self.validity_mask_eval(proposal[None, :], target_class=target_class)
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
                valid, _ = self.validity_mask_eval(proposal[None, :], target_class=target_class)
                if current_val == right or current_val == left:
                    break
                if bool(valid[0]):
                    current = proposal
                    right = current_val - change
                else:
                    left = current_val + change
        return current.astype(np.float32)

    def _continuous_decimal_precision(self, feat_idx: int) -> int:
        """Infer the number of decimal places needed for a feature from its minimum observed step size."""
        assert self._continuous_steps is not None
        step = float(self._continuous_steps[feat_idx])
        if step <= 0.0:
            return 6
        return int(max(0, min(6, np.ceil(-np.log10(step + 1e-12)))))

    def _continuous_linear_step(self, feat_idx: int) -> float:
        """Return the linear search step size (10^-precision) for a feature."""
        decimal_prec = self._continuous_decimal_precision(feat_idx)
        return float(10.0 ** (-decimal_prec))

    def select_best_valid(self, valid_eval: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        """Return the valid candidate closest to the query under inverse-MAD weighted L1."""
        assert self.feature_weights is not None
        weighted_l1 = np.sum(np.abs(valid_eval - x_query[None, :]) * self.feature_weights[None, :], axis=1)
        idx = int(np.argmin(weighted_l1))
        return valid_eval[idx]

    def resolve_target_class(self, x: np.ndarray, target_class: Optional[int]) -> int:
        """Return the given target class, or the opposite predicted class for binary models when target_class is None."""
        if target_class is not None:
            return int(target_class)
        pred = int(self.model.predict(x[None, :])[0])
        if self.model.predict_proba(x[None, :]).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
