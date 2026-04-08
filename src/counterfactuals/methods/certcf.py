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
    """Wrap ``certcf.CertCFAtlas`` into the common method interface."""

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
        solver_maxiter: int = 500,

        # Custom downsampling strategy
        k_per_class: Optional[int] = None,
        subsample_method: str = "kmedoids",
        subsample_space: str = "input",

        # Random seed for reproducibility (e.g., in subsampling)
        random_seed: int = 42,
    ):
        subsample_space = str(subsample_space).lower()
        if subsample_space not in {"input", "latent"}:
            raise ValueError("subsample_space must be one of {'input', 'latent'}")
        super().__init__(model=model, random_seed=random_seed,
                         k_per_class=k_per_class, subsample_method=subsample_method)
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
        self.solver_maxiter = solver_maxiter
        self.subsample_space = subsample_space
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

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Store training data, optionally subsample in input or latent space, then build the atlas."""
        self._x_train = np.asarray(x_train, dtype=np.float32)
        self._y_train = np.asarray(y_train, dtype=np.int64)
        if self._x_train.shape[0] != self._y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of rows")

        if self.k_per_class is not None:
            from counterfactuals.utils.clustering import select_prototype_indices

            selection_space_data = self._x_train
            if self.subsample_space == "latent":
                selection_space_data = self._extract_penultimate_embeddings(self._x_train)

            parts_x, parts_y = [], []
            for cls in np.unique(self._y_train):
                idx = np.where(self._y_train == cls)[0]
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
        )

        x_cf = np.asarray(result.x_cf, dtype=np.float32) if result.x_cf is not None else None
        return CounterfactualResult(
            x_cf=x_cf,
            success=bool(result.success),
            distance=float(result.distance),
            metadata=result.profiling,
        )
