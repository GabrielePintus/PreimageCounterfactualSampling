"""Core interfaces for models, datasets, and metric evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class ModelInterface(ABC):
    """Common model API consumed by all counterfactual methods."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices for a batch of inputs."""

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch of inputs."""


class DatasetInterface(ABC):
    """Common dataset API used by experiment runners."""

    @abstractmethod
    def load(self) -> None:
        """Load data into memory or trigger preprocessing."""

    @abstractmethod
    def get_train(self) -> tuple[np.ndarray, np.ndarray]:
        """Return train features and labels."""

    @abstractmethod
    def get_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Return test features and labels."""


class MetricInterface(ABC):
    """Evaluation metric interface."""

    name: str

    @abstractmethod
    def evaluate(
        self,
        x_orig: np.ndarray,
        x_cf: np.ndarray,
        context: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Compute a scalar metric value for one factual/counterfactual pair."""
