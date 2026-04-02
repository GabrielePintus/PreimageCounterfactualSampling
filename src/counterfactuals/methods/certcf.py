"""Adapter for the CertCF method.

"""

from __future__ import annotations

import copy
from typing import Optional, List, Tuple

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

        # Random seed for reproducibility (e.g., in subsampling)
        random_seed: int = 42,
    ):
        super().__init__(model=model, random_seed=random_seed,
                         k_per_class=k_per_class, subsample_method=subsample_method)
        # Build config — used in _fit()
        self.norm = norm
        self.eps_strategy = eps_strategy
        self.batch_size = batch_size
        self.ohe_slices = ohe_slices
        # Atlas config
        self.cnn = cnn
        self.default_query_method = default_query_method
        self.solver_maxiter = solver_maxiter
        self.atlas: Optional[CertCFAtlas] = None

    def _fit(self) -> None:
        """Build the CertCFAtlas from training data."""
        # Extract the raw torch module and device from the ModelInterface wrapper
        module = getattr(self.model, "model", None)
        if not isinstance(module, nn.Module):
            raise TypeError(
                "CertCF requires a TorchModelWrapper (model.model must be an nn.Module)."
            )
        device = getattr(self.model, "device", torch.device("cpu"))
        device = torch.device(device)

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
        )

        x_cf = np.asarray(result.x_cf, dtype=np.float32) if result.x_cf is not None else None
        return CounterfactualResult(
            x_cf=x_cf,
            success=bool(result.success),
            distance=float(result.distance),
            metadata=result.profiling,
        )
