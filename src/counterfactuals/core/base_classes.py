"""Base abstractions shared by all counterfactual algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field  # field used by CounterfactualResult
from typing import Any, Dict, Optional

import numpy as np

from .interfaces import ModelInterface


@dataclass
class CounterfactualResult:
    """Standardized output produced by all methods."""

    x_cf: np.ndarray
    success: bool
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCounterfactualMethod(ABC):
    """Abstract base class for all counterfactual methods."""

    def __init__(
        self,
        model: ModelInterface,
        random_seed: int = 42,
        k_per_class: Optional[int] = None,
        subsample_method: str = "kmedoids",
    ):
        allowed_subsample_methods = {
            "random",
            "kmedoids",
            "bandit_kmedoids",
            "kmeans",
            "fps",
            "density_flat_kmedoids",
        }
        if subsample_method not in allowed_subsample_methods:
            raise ValueError(
                "subsample_method must be one of "
                f"{sorted(allowed_subsample_methods)}"
            )
        self.model = model
        self.random_seed = random_seed
        self.k_per_class = k_per_class
        self.subsample_method = subsample_method
        self.rng = np.random.default_rng(random_seed)
        self._x_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Store training data, apply optional subsampling, then call ``_fit()``.

        The labels used for class-wise subsampling are exactly the labels passed
        into ``fit(...)``. In benchmark runs those labels may be model
        predictions; outside the benchmark the caller remains free to choose
        whichever support labels are semantically appropriate.
        """
        self._x_train = np.asarray(x_train, dtype=np.float32)
        self._y_train = np.asarray(y_train, dtype=np.int64)
        if self._x_train.shape[0] != self._y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of rows")
        if self.k_per_class is not None:
            from counterfactuals.utils.clustering import select_prototype_indices

            parts_x, parts_y = [], []
            for cls in np.unique(self._y_train):
                idx = np.where(self._y_train == cls)[0]
                proto = select_prototype_indices(
                    self._x_train[idx], self.k_per_class,
                    method=self.subsample_method, random_state=self.random_seed,
                )
                parts_x.append(self._x_train[idx[proto]])
                parts_y.append(self._y_train[idx[proto]])
            self._x_train = np.concatenate(parts_x)
            self._y_train = np.concatenate(parts_y)
        self._fit()
        self._is_fitted = True

    @abstractmethod
    def _fit(self) -> None:
        """Method-specific setup. self._x_train, self._y_train, and self.model are already set."""

    @abstractmethod
    def generate(
        self,
        x: np.ndarray,
        target_class: Optional[int] = None,
    ) -> CounterfactualResult:
        """Generate one counterfactual for a single point."""

    def generate_batch(
        self,
        x: np.ndarray,
        target_class: Optional[int] = None,
    ) -> list[CounterfactualResult]:
        """Generate one counterfactual per row in ``x`` with shared settings."""
        return [self.generate(x=row, target_class=target_class) for row in np.asarray(x)]

    def resolve_target_class(self, x: np.ndarray, target_class: Optional[int]) -> int:
        """Return target_class if given; otherwise flip the predicted class (binary only)."""
        if target_class is not None:
            return int(target_class)
        pred = int(self.model.predict(x.reshape(1, -1))[0])
        if self.model.predict_proba(x.reshape(1, -1)).shape[1] != 2:
            raise ValueError("target_class is required for non-binary tasks")
        return 1 - pred
