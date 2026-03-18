from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDensityEstimator(ABC):
    """Fits a density model on training data and evaluates it at arbitrary points."""

    @abstractmethod
    def fit(self, z_train: np.ndarray) -> None:
        """Learn the density from training embeddings."""

    @abstractmethod
    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Return density estimate at each row of z. Shape: (n,)"""
