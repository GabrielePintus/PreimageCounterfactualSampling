"""Dataset implementations for experiment orchestration."""

from __future__ import annotations

import numpy as np

from counterfactuals.core.interfaces import DatasetInterface


class BaseDataset(DatasetInterface):
    """Simple in-memory dataset base class."""

    spec = None

    def __init__(self):
        self._loaded = False

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Dataset is not loaded. Call load() first.")


class NumpyDataset(BaseDataset):
    """Dataset backed directly by provided numpy arrays."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__()
        self._x_train = np.asarray(x_train, dtype=np.float32)
        self._y_train = np.asarray(y_train, dtype=np.int64)
        self._x_test = np.asarray(x_test, dtype=np.float32)
        self._y_test = np.asarray(y_test, dtype=np.int64)

    def load(self) -> None:
        self._loaded = True

    def get_train(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_loaded()
        return self._x_train, self._y_train

    def get_test(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_loaded()
        return self._x_test, self._y_test
