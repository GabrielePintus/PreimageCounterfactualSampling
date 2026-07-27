"""Adapter for the CertCF method."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from counterfactuals.core.base_classes import CounterfactualResult, BaseCounterfactualMethod
from counterfactuals.core.interfaces import ModelInterface
from counterfactuals.methods._constraints import (
    directional_metadata,
    normalize_directional_dims,
    normalize_fixed_dims,
    validate_disjoint_directional_dims,
)
from certcf.atlas import CertCFAtlas
from certcf.eps_strategies import EpsStrategy


def _strip_dropout_modules(module: nn.Module) -> nn.Module:
    """Deep-copy a module and replace all dropout layers with identities."""
    clean = copy.deepcopy(module)
    for name, child in list(clean.named_children()):
        if isinstance(child, nn.Dropout):
            setattr(clean, name, nn.Identity())
        else:
            setattr(clean, name, _strip_dropout_modules(child))
    return clean


class CertCF(BaseCounterfactualMethod):
    """Wrap ``certcf.CertCFAtlas`` into the common method interface.

    The wrapper treats the labels provided to ``fit(...)`` as the operative
    atlas support labels. In the benchmark pipeline those labels are now
    prediction-aligned and computed once up front; outside the benchmark the
    caller remains free to pass any support labels intentionally.
    """

    def __init__(
        self,
        model: ModelInterface,

        # Build configuration
        norm: int = 1,
        distance_norm: Optional[float] = None,
        lirpa_method: str = "backward",
        delta: float = 0.0,
        robust_norm: Optional[Union[int, float, str]] = None,
        eps_strategy: Optional[EpsStrategy] = None,
        batch_size: Optional[int] = None,
        ohe_slices: Optional[List[Tuple[int, int]]] = None,
        input_bounds: Optional[Sequence[float]] = None,

        # Model / atlas configuration
        cnn: bool = False,
        default_query_method: str = "sorted",
        query_k_candidates: int = 1,
        solver_maxiter: int = 500,
        query_parallelism: int = 1,
        cvxpy_solvers: Optional[List[str]] = None,
        cvxpy_solver_options: Optional[Dict[str, Dict[str, Any]]] = None,
        cvxpy_accept_statuses: Optional[Dict[str, List[str]]] = None,
        classification_margin: float = 0.0,
        adaptive_eps: bool = False,
        adaptive_eps_shrink_factor: float = 0.5,
        adaptive_eps_max_shrinks: int = 8,
        adaptive_eps_min: float = 1.0e-6,
        adaptive_eps_center_tol: float = 1.0e-6,
        adaptive_eps_binary_search_steps: int = 0,
        ohe_decode_mode: str = "exact",
        decode_beam_width: int = 8,
        decode_beam_branch_top_k: int = 3,
        decode_beam_max_solver_calls: int = 32,
        sparsity_penalty: str = "none",
        sparsity_lambda: float = 0.0,
        sparsity_reweight_iters: int = 0,
        sparsity_eps: float = 1.0e-3,
        sparsity_group_ohe: bool = True,
        fixed_dims: Optional[Sequence[int]] = None,
        immutable_features: Optional[Sequence[str]] = None,
        nondecreasing_dims: Optional[Sequence[int]] = None,
        nonincreasing_dims: Optional[Sequence[int]] = None,
        nondecreasing_features: Optional[Sequence[str]] = None,
        nonincreasing_features: Optional[Sequence[str]] = None,

        # Custom downsampling strategy
        k_per_class: Optional[int] = None,
        subsample_method: str = "kmedoids",
        subsample_space: str = "input",
        boundary_beta: float = 0.5,

        # Random seed for reproducibility (e.g., in subsampling)
        random_seed: int = 42,
    ):
        subsample_space = str(subsample_space).lower()
        if subsample_space not in {"input", "latent"}:
            raise ValueError("subsample_space must be one of {'input', 'latent'}")
        base_subsample_method = "random" if str(subsample_method).lower() == "boundary_random" else subsample_method
        super().__init__(model=model, random_seed=random_seed,
                         k_per_class=k_per_class, subsample_method=base_subsample_method)
        self.subsample_method = str(subsample_method).lower()
        # Build config — used in _fit()
        self.norm = norm
        self.distance_norm = distance_norm
        self.lirpa_method = str(lirpa_method)
        self.delta = float(delta)
        self.robust_norm = None if robust_norm is None else CertCFAtlas._normalize_lp_norm(robust_norm)
        self.eps_strategy = eps_strategy
        self.batch_size = batch_size
        self.ohe_slices = ohe_slices
        normalize_input_bounds = getattr(CertCFAtlas, "_normalize_input_bounds", None)
        if normalize_input_bounds is None:
            normalize_input_bounds = __import__(
                "certcf.atlas",
                fromlist=["CertCFAtlas"],
            ).CertCFAtlas._normalize_input_bounds
        self.input_bounds = normalize_input_bounds(input_bounds)
        self.fixed_dims = normalize_fixed_dims(fixed_dims)
        self.immutable_features = tuple(str(name) for name in (immutable_features or ()))
        self.nondecreasing_dims = normalize_directional_dims(
            nondecreasing_dims,
            name="nondecreasing_dims",
        )
        self.nonincreasing_dims = normalize_directional_dims(
            nonincreasing_dims,
            name="nonincreasing_dims",
        )
        validate_disjoint_directional_dims(self.nondecreasing_dims, self.nonincreasing_dims)
        self.nondecreasing_features = tuple(str(name) for name in (nondecreasing_features or ()))
        self.nonincreasing_features = tuple(str(name) for name in (nonincreasing_features or ()))
        # Atlas config
        self.cnn = cnn
        self.default_query_method = default_query_method
        self.query_k_candidates = int(query_k_candidates)
        if self.query_k_candidates <= 0:
            raise ValueError("query_k_candidates must be positive")
        self.solver_maxiter = solver_maxiter
        self.query_parallelism = int(query_parallelism)
        if self.query_parallelism <= 0:
            raise ValueError("query_parallelism must be positive")
        (
            self.cvxpy_solvers,
            self.cvxpy_solver_options,
            self.cvxpy_accept_statuses,
        ) = CertCFAtlas._normalize_cvxpy_solver_config(
            cvxpy_solvers=cvxpy_solvers,
            cvxpy_solver_options=cvxpy_solver_options,
            cvxpy_accept_statuses=cvxpy_accept_statuses,
        )
        self.classification_margin = float(classification_margin)
        if self.classification_margin < 0.0:
            raise ValueError("classification_margin must be non-negative")
        self.adaptive_eps = bool(adaptive_eps)
        self.adaptive_eps_shrink_factor = float(adaptive_eps_shrink_factor)
        if not (0.0 < self.adaptive_eps_shrink_factor < 1.0):
            raise ValueError("adaptive_eps_shrink_factor must be in (0, 1)")
        self.adaptive_eps_max_shrinks = int(adaptive_eps_max_shrinks)
        if self.adaptive_eps_max_shrinks < 0:
            raise ValueError("adaptive_eps_max_shrinks must be non-negative")
        self.adaptive_eps_min = float(adaptive_eps_min)
        if self.adaptive_eps_min < 0.0:
            raise ValueError("adaptive_eps_min must be non-negative")
        self.adaptive_eps_center_tol = float(adaptive_eps_center_tol)
        if self.adaptive_eps_center_tol < 0.0:
            raise ValueError("adaptive_eps_center_tol must be non-negative")
        self.adaptive_eps_binary_search_steps = int(adaptive_eps_binary_search_steps)
        if self.adaptive_eps_binary_search_steps < 0:
            raise ValueError("adaptive_eps_binary_search_steps must be non-negative")
        (
            self.ohe_decode_mode,
            self.decode_beam_width,
            self.decode_beam_branch_top_k,
            self.decode_beam_max_solver_calls,
        ) = CertCFAtlas._normalize_ohe_decode_config(
            ohe_decode_mode=ohe_decode_mode,
            decode_beam_width=decode_beam_width,
            decode_beam_branch_top_k=decode_beam_branch_top_k,
            decode_beam_max_solver_calls=decode_beam_max_solver_calls,
        )
        atlas_cls_for_sparsity = CertCFAtlas
        normalize_sparsity_config = getattr(atlas_cls_for_sparsity, "_normalize_sparsity_config", None)
        if normalize_sparsity_config is None:
            atlas_cls_for_sparsity = __import__(
                "certcf.atlas",
                fromlist=["CertCFAtlas"],
            ).CertCFAtlas
            normalize_sparsity_config = atlas_cls_for_sparsity._normalize_sparsity_config
        normalize_lp_norm = getattr(atlas_cls_for_sparsity, "_normalize_lp_norm")
        (
            self.sparsity_penalty,
            self.sparsity_lambda,
            self.sparsity_reweight_iters,
            self.sparsity_eps,
            self.sparsity_group_ohe,
        ) = normalize_sparsity_config(
            sparsity_penalty=sparsity_penalty,
            sparsity_lambda=sparsity_lambda,
            sparsity_reweight_iters=sparsity_reweight_iters,
            sparsity_eps=sparsity_eps,
            sparsity_group_ohe=sparsity_group_ohe,
            distance_norm=normalize_lp_norm(distance_norm if distance_norm is not None else norm),
        )
        self.subsample_space = subsample_space
        self.boundary_beta = float(boundary_beta)
        if not (0.0 <= self.boundary_beta <= 1.0):
            raise ValueError("boundary_beta must be in [0, 1].")
        self.atlas: Optional[CertCFAtlas] = None

    def _resolve_torch_module_and_device(self) -> tuple[nn.Module, torch.device]:
        """Extract the raw torch module and normalized device from the model wrapper."""
        module = getattr(self.model, "model", None)
        if not isinstance(module, nn.Module):
            raise TypeError(
                "CertCF requires a TorchModelWrapper (model.model must be an nn.Module)."
            )
        device = getattr(self.model, "device", torch.device("cpu"))
        return module, torch.device(device)

    def _extract_penultimate_embeddings(
        self,
        x: np.ndarray,
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Return penultimate activations for supported tabular torch classifiers."""
        if self.cnn:
            raise ValueError(
                "subsample_space='latent' is currently supported only for tabular torch classifiers, "
                "not CNN/MNIST models."
            )

        module, device = self._resolve_torch_module_and_device()
        clean_module = _strip_dropout_modules(module).eval().to(device)

        if not isinstance(clean_module, nn.Sequential):
            raise ValueError(
                "subsample_space='latent' currently requires the wrapped model to be an nn.Sequential "
                "tabular classifier."
            )

        layers = list(clean_module.children())
        if len(layers) < 2 or not isinstance(layers[-1], nn.Linear):
            raise ValueError(
                "subsample_space='latent' currently requires a tabular sequential model whose final "
                "layer is nn.Linear."
            )

        feature_extractor = nn.Sequential(*layers[:-1]).eval().to(device)
        effective_batch_size = int(batch_size or self.batch_size or 1024)
        x_np = np.asarray(x, dtype=np.float32)
        parts: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(x_np), effective_batch_size):
                batch = torch.from_numpy(x_np[start:start + effective_batch_size]).to(device)
                features = feature_extractor(batch)
                if features.ndim > 2:
                    features = features.view(features.shape[0], -1)
                parts.append(features.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(parts, axis=0) if parts else np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def _predicted_class_indices_from_outputs(outputs: torch.Tensor) -> np.ndarray:
        """Convert batched model outputs into integer class predictions."""
        if outputs.ndim == 0:
            raise ValueError("CertCF training-label prediction requires batched model outputs.")
        if outputs.ndim == 1:
            preds = (outputs > 0).to(dtype=torch.int64)
        else:
            flat_outputs = outputs.view(outputs.shape[0], -1)
            if flat_outputs.shape[1] == 1:
                column = flat_outputs[:, 0]
                if torch.all((column >= 0) & (column <= 1)):
                    preds = (column >= 0.5).to(dtype=torch.int64)
                else:
                    preds = (column > 0).to(dtype=torch.int64)
            else:
                preds = torch.argmax(flat_outputs, dim=1).to(dtype=torch.int64)
        return preds.detach().cpu().numpy().astype(np.int64, copy=False)

    @staticmethod
    def _boundary_scores_from_outputs(outputs: torch.Tensor) -> np.ndarray:
        """Convert batched model outputs into boundary proximity scores in [0, 1]."""
        if outputs.ndim == 0:
            raise ValueError("CertCF boundary-score prediction requires batched model outputs.")

        if outputs.ndim == 1:
            column = outputs
        else:
            flat_outputs = outputs.view(outputs.shape[0], -1)
            if flat_outputs.shape[1] == 1:
                column = flat_outputs[:, 0]
            else:
                top2 = torch.topk(flat_outputs, k=2, dim=1).values
                margins = torch.abs(top2[:, 0] - top2[:, 1])
                is_prob_like = bool(
                    torch.all((flat_outputs >= -1e-6) & (flat_outputs <= 1.0 + 1e-6)).item()
                    and torch.max(torch.abs(flat_outputs.sum(dim=1) - 1.0)).item() <= 1e-3
                )
                if is_prob_like:
                    scores = 1.0 - torch.clamp(margins, min=0.0, max=1.0)
                else:
                    scores = 1.0 / (1.0 + margins)
                scores = torch.clamp(scores, min=0.0, max=1.0)
                return scores.detach().cpu().numpy().astype(np.float64, copy=False)

        if torch.all((column >= -1e-6) & (column <= 1.0 + 1e-6)):
            scores = 1.0 - torch.abs(2.0 * column - 1.0)
            scores = torch.clamp(scores, min=0.0, max=1.0)
        else:
            scores = 1.0 / (1.0 + torch.abs(column))
        return scores.detach().cpu().numpy().astype(np.float64, copy=False)

    def _predict_training_labels(
        self,
        x: np.ndarray,
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Return model-predicted labels for the training support passed to ``fit``."""
        module, device = self._resolve_torch_module_and_device()
        clean_module = module.eval().to(device)
        effective_batch_size = int(batch_size or self.batch_size or 1024)
        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np[None, :]

        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_np), effective_batch_size):
                batch = torch.from_numpy(x_np[start:start + effective_batch_size]).to(device)
                outputs = clean_module(batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                if not isinstance(outputs, torch.Tensor):
                    raise TypeError("CertCF training-label prediction requires the wrapped torch model to return a tensor.")
                parts.append(self._predicted_class_indices_from_outputs(outputs))

        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.int64)

    def _predict_training_boundary_scores(
        self,
        x: np.ndarray,
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Return per-point boundary proximity scores (higher means closer to boundary)."""
        module, device = self._resolve_torch_module_and_device()
        clean_module = module.eval().to(device)
        effective_batch_size = int(batch_size or self.batch_size or 1024)
        x_np = np.asarray(x, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np[None, :]

        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_np), effective_batch_size):
                batch = torch.from_numpy(x_np[start:start + effective_batch_size]).to(device)
                outputs = clean_module(batch)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                if not isinstance(outputs, torch.Tensor):
                    raise TypeError("CertCF boundary-score prediction requires the wrapped torch model to return a tensor.")
                parts.append(self._boundary_scores_from_outputs(outputs))

        return np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float64)

    def _select_boundary_weighted_indices(
        self,
        scores: np.ndarray,
        k: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample k unique indices with a convex mix of uniform and boundary-biased probabilities."""
        n = int(len(scores))
        k = min(int(k), n)
        if k == n:
            return np.arange(n, dtype=np.int64)

        uniform = np.full(n, 1.0 / n, dtype=np.float64)
        clean_scores = np.asarray(scores, dtype=np.float64)
        clean_scores = np.nan_to_num(clean_scores, nan=0.0, posinf=0.0, neginf=0.0)
        clean_scores = np.clip(clean_scores, a_min=0.0, a_max=None)
        total = float(clean_scores.sum())

        if total <= 0.0:
            probs = uniform
        else:
            boundary = clean_scores / total
            probs = (1.0 - self.boundary_beta) * uniform + self.boundary_beta * boundary
            probs = probs / probs.sum()

        chosen = rng.choice(n, size=k, replace=False, p=probs)
        return np.asarray(chosen, dtype=np.int64)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Store the provided support labels, optionally subsample, then build the atlas."""
        self._x_train = np.asarray(x_train, dtype=np.float32)
        support_labels = np.asarray(y_train, dtype=np.int64)
        if self._x_train.shape[0] != support_labels.shape[0]:
            raise ValueError("x_train and y_train must have the same number of rows")
        self._y_train_support_full = support_labels.copy()
        self._y_train = self._y_train_support_full.copy()
        self._y_train_support = self._y_train.copy()

        support_classes = np.unique(self._y_train_support_full)
        if support_classes.size < 2:
            raise ValueError(
                "CertCF training support collapsed to fewer than 2 classes; "
                "the provided labels do not define a useful atlas."
            )

        if self.k_per_class is not None:
            from counterfactuals.utils.clustering import select_prototype_indices

            selection_space_data = self._x_train
            if self.subsample_space == "latent":
                selection_space_data = self._extract_penultimate_embeddings(self._x_train)
            boundary_scores_full = None
            if self.subsample_method == "boundary_random":
                boundary_scores_full = self._predict_training_boundary_scores(self._x_train)
            sampling_rng = np.random.default_rng(self.random_seed)

            parts_x, parts_y = [], []
            for cls in np.unique(self._y_train):
                idx = np.where(self._y_train == cls)[0]
                if self.subsample_method == "boundary_random":
                    if boundary_scores_full is None:
                        raise RuntimeError("Internal error: boundary scores were not computed for boundary_random sampling.")
                    proto = self._select_boundary_weighted_indices(
                        boundary_scores_full[idx],
                        self.k_per_class,
                        rng=sampling_rng,
                    )
                else:
                    proto = select_prototype_indices(
                        selection_space_data[idx],
                        self.k_per_class,
                        method=self.subsample_method,
                        random_state=self.random_seed,
                    )
                parts_x.append(self._x_train[idx[proto]])
                parts_y.append(self._y_train[idx[proto]])
            self._x_train = np.concatenate(parts_x)
            self._y_train = np.concatenate(parts_y)
            self._y_train_support = self._y_train.copy()
        else:
            self._y_train_support = self._y_train.copy()

        self._fit()
        self._is_fitted = True

    def _fit(self) -> None:
        """Build the CertCFAtlas from training data."""
        module, device = self._resolve_torch_module_and_device()

        # Strip Dropout for deterministic LiRPA certification while preserving
        # the module's original forward logic (e.g. CNN flatten/view steps).
        clean_module = _strip_dropout_modules(module)

        dataset = TensorDataset(
            torch.from_numpy(self._x_train).float(),
            torch.from_numpy(self._y_train).long(),
        )

        self.atlas = CertCFAtlas(
            clean_module, dataset, device,
            cnn=self.cnn,
            norm=self.norm,
            distance_norm=self.distance_norm,
            lirpa_method=self.lirpa_method,
            eps_strategy=self.eps_strategy,
            batch_size=self.batch_size,
            ohe_slices=self.ohe_slices,
            input_bounds=self.input_bounds,
            default_query_method=self.default_query_method,
            solver_maxiter=self.solver_maxiter,
            query_parallelism=self.query_parallelism,
            cvxpy_solvers=self.cvxpy_solvers,
            cvxpy_solver_options=self.cvxpy_solver_options,
            cvxpy_accept_statuses=self.cvxpy_accept_statuses,
            classification_margin=self.classification_margin,
            adaptive_eps=self.adaptive_eps,
            adaptive_eps_shrink_factor=self.adaptive_eps_shrink_factor,
            adaptive_eps_max_shrinks=self.adaptive_eps_max_shrinks,
            adaptive_eps_min=self.adaptive_eps_min,
            adaptive_eps_center_tol=self.adaptive_eps_center_tol,
            adaptive_eps_binary_search_steps=self.adaptive_eps_binary_search_steps,
            ohe_decode_mode=self.ohe_decode_mode,
            decode_beam_width=self.decode_beam_width,
            decode_beam_branch_top_k=self.decode_beam_branch_top_k,
            decode_beam_max_solver_calls=self.decode_beam_max_solver_calls,
            sparsity_penalty=self.sparsity_penalty,
            sparsity_lambda=self.sparsity_lambda,
            sparsity_reweight_iters=self.sparsity_reweight_iters,
            sparsity_eps=self.sparsity_eps,
            sparsity_group_ohe=self.sparsity_group_ohe,
        )
        self.atlas.build()

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        """Find the closest certified counterfactual for query x."""
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")
        if target_class is None:
            raise ValueError("CertCF requires target_class.")

        return self.generate_batch(
            x=np.asarray(x, dtype=np.float32).reshape(1, -1),
            target_class=int(target_class),
        )[0]

    def generate_batch(
        self,
        x: np.ndarray,
        target_class: Optional[Union[int, Sequence[int], np.ndarray]] = None,
        timeout_s_per_query: Optional[float] = None,
    ) -> list[CounterfactualResult]:
        """Find the closest certified counterfactual for each query in x."""
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")
        if target_class is None:
            raise ValueError("CertCF requires target_class.")

        x_batch = np.asarray(x, dtype=np.float32)
        if x_batch.ndim == 1:
            x_batch = x_batch.reshape(1, -1)

        fixed_dims = getattr(self, "fixed_dims", None)
        nondecreasing_dims = getattr(self, "nondecreasing_dims", None)
        nonincreasing_dims = getattr(self, "nonincreasing_dims", None)
        results = self.atlas.find_counterfactual_batch(
            X_query=x_batch,
            target_class=target_class,
            delta=self.delta,
            robust_norm=self.robust_norm,
            fixed_dims=fixed_dims,
            nondecreasing_dims=nondecreasing_dims,
            nonincreasing_dims=nonincreasing_dims,
            query_k_candidates=self.query_k_candidates,
            timeout_s_per_query=timeout_s_per_query,
        )
        return [self._wrap_atlas_result(result) for result in results]

    def _wrap_atlas_result(self, result) -> CounterfactualResult:
        """Convert atlas-level results into the shared benchmark result type."""
        x_cf = np.asarray(result.x_cf, dtype=np.float32) if result.x_cf is not None else None
        metadata = dict(result.profiling)
        fixed_dims = getattr(self, "fixed_dims", None)
        immutable_features = getattr(self, "immutable_features", ())
        nondecreasing_dims = getattr(self, "nondecreasing_dims", None)
        nonincreasing_dims = getattr(self, "nonincreasing_dims", None)
        nondecreasing_features = getattr(self, "nondecreasing_features", ())
        nonincreasing_features = getattr(self, "nonincreasing_features", ())
        metadata.setdefault("sparsity_penalty", getattr(self, "sparsity_penalty", "none"))
        metadata.setdefault("sparsity_lambda", float(getattr(self, "sparsity_lambda", 0.0)))
        metadata.setdefault("sparsity_reweight_iters", int(getattr(self, "sparsity_reweight_iters", 0)))
        metadata.setdefault("sparsity_eps", float(getattr(self, "sparsity_eps", 1.0e-3)))
        metadata.setdefault("sparsity_group_count", 0)
        metadata.setdefault("sparsity_active_groups", 0)
        metadata.setdefault("sparsity_selection_score", np.inf)
        metadata.setdefault("sparsity_solver_calls", 0)
        metadata["fixed_dims_count"] = int(0 if fixed_dims is None else len(fixed_dims))
        metadata["immutable_features"] = ",".join(immutable_features)
        metadata.update(
            directional_metadata(
                nondecreasing_dims,
                nonincreasing_dims,
                nondecreasing_features,
                nonincreasing_features,
            )
        )
        return CounterfactualResult(
            x_cf=x_cf,
            success=bool(result.success),
            distance=float(result.distance),
            metadata=metadata,
        )
