"""Adapter for the CertCF method."""

from __future__ import annotations

import copy
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from counterfactuals.core.base_classes import CounterfactualResult, BaseCounterfactualMethod
from counterfactuals.core.interfaces import ModelInterface
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

        # Model / atlas configuration
        cnn: bool = False,
        default_query_method: str = "sorted",
        query_k_candidates: int = 1,
        solver_maxiter: int = 500,

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
        # Atlas config
        self.cnn = cnn
        self.default_query_method = default_query_method
        self.query_k_candidates = int(query_k_candidates)
        if self.query_k_candidates <= 0:
            raise ValueError("query_k_candidates must be positive")
        self.solver_maxiter = solver_maxiter
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
            default_query_method=self.default_query_method,
            solver_maxiter=self.solver_maxiter,
        )
        self.atlas.build()

    def generate(self, x: np.ndarray, target_class: Optional[int] = None) -> CounterfactualResult:
        """Find the closest certified counterfactual for query x."""
        if not self._is_fitted:
            raise RuntimeError("Method is not fitted. Call fit() before generate().")
        if target_class is None:
            raise ValueError("CertCF requires target_class.")

        result = self.atlas.find_counterfactual(
            x_query=np.asarray(x, dtype=np.float32),
            target_class=int(target_class),
            delta=self.delta,
            robust_norm=self.robust_norm,
            query_k_candidates=self.query_k_candidates,
        )

        x_cf = np.asarray(result.x_cf, dtype=np.float32) if result.x_cf is not None else None
        return CounterfactualResult(
            x_cf=x_cf,
            success=bool(result.success),
            distance=float(result.distance),
            metadata=result.profiling,
        )
