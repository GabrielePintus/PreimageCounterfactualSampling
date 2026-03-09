"""Dataset loader adapters for experiments."""

from __future__ import annotations

import numpy as np

from .base_dataset import BaseDataset


class AdultDataset(BaseDataset):
    """Adapter that reuses the existing Lightning Adult datamodule."""

    def __init__(self, data_dir: str = "data/", seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self._x_train: np.ndarray
        self._y_train: np.ndarray
        self._x_test: np.ndarray
        self._y_test: np.ndarray

    def load(self) -> None:
        from training.datamodules.adult import AdultDataModule

        dm = AdultDataModule(
            data_dir=self.data_dir,
            batch_size=256,
            num_workers=0,
            seed=self.seed,
        )
        dm.setup()
        x_train, y_train = dm.train_ds.tensors
        x_test, y_test = dm.test_ds.tensors
        self._x_train = x_train.cpu().numpy().astype(np.float32)
        self._y_train = y_train.cpu().numpy().astype(np.int64)
        self._x_test = x_test.cpu().numpy().astype(np.float32)
        self._y_test = y_test.cpu().numpy().astype(np.int64)
        self._loaded = True

    def get_train(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_loaded()
        return self._x_train, self._y_train

    def get_test(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_loaded()
        return self._x_test, self._y_test
