"""Base abstractions shared by all counterfactual algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .interfaces import ModelInterface


@dataclass(frozen=True)
class CounterfactualExample:
    """Input sample and optional metadata for generation."""

    x: np.ndarray
    target_class: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualResult:
    """Standardized output produced by all methods."""

    x_cf: np.ndarray
    success: bool
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCounterfactualMethod(ABC):
    """Abstract base class for all counterfactual methods."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self._is_fitted = False

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, model: ModelInterface) -> None:
        """Optional offline setup stage using train data and model."""

    @abstractmethod
    def generate(
        self,
        example: CounterfactualExample,
        model: ModelInterface,
    ) -> CounterfactualResult:
        """Generate one counterfactual for a single example."""

    def generate_batch(
        self,
        x: np.ndarray,
        model: ModelInterface,
        target_class: Optional[int] = None,
    ) -> list[CounterfactualResult]:
        """Generate one counterfactual per row in ``x`` with shared settings."""
        examples = [
            CounterfactualExample(x=row, target_class=target_class)
            for row in np.asarray(x)
        ]
        return [self.generate(example=example, model=model) for example in examples]
