"""Dataset loader adapters for experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dataset_specs import get_tabular_dataset_spec

from .base_dataset import BaseDataset


class CompasDataset(BaseDataset):
    """Adapter that wraps the ProPublica COMPAS LightningDataModule."""

    spec = get_tabular_dataset_spec("compas")

    def __init__(self, data_dir: str = "data/", seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self._x_train: np.ndarray
        self._y_train: np.ndarray
        self._x_test: np.ndarray
        self._y_test: np.ndarray

    def load(self) -> None:
        from training.datamodules.compas import CompasDataModule

        dm = CompasDataModule(
            filepath=str(Path(self.data_dir) / "Compas" / "raw.parquet"),
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


class GiveMeSomeCreditDataset(BaseDataset):
    """Adapter that wraps the Give Me Some Credit LightningDataModule."""

    spec = get_tabular_dataset_spec("give_me_some_credit")

    def __init__(self, data_dir: str = "data/", seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self._x_train: np.ndarray
        self._y_train: np.ndarray
        self._x_test: np.ndarray
        self._y_test: np.ndarray

    def load(self) -> None:
        from training.datamodules.give_me_some_credit import GiveMeSomeCreditDataModule

        dm = GiveMeSomeCreditDataModule(
            filepath=str(Path(self.data_dir) / "Give Me Some Credit" / "raw.parquet"),
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


class HELOCDataset(BaseDataset):
    """Adapter that wraps the FICO HELOC LightningDataModule."""

    spec = get_tabular_dataset_spec("heloc")

    def __init__(self, data_dir: str = "data/", seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self._x_train: np.ndarray
        self._y_train: np.ndarray
        self._x_test: np.ndarray
        self._y_test: np.ndarray

    def load(self) -> None:
        from training.datamodules.heloc import HELOCDataModule

        dm = HELOCDataModule(
            filepath=str(Path(self.data_dir) / "Heloc" / "raw.parquet"),
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


class GermanCreditDataset(BaseDataset):
    """Adapter that wraps the UCI German Credit LightningDataModule."""

    spec = get_tabular_dataset_spec("german_credit")

    def __init__(self, data_dir: str = "data/", seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self._x_train: np.ndarray
        self._y_train: np.ndarray
        self._x_test: np.ndarray
        self._y_test: np.ndarray

    def load(self) -> None:
        from training.datamodules.german_credit import GermanCreditDataModule

        dm = GermanCreditDataModule(
            filepath=str(Path(self.data_dir) / "GermanCredit" / "raw.parquet"),
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


class LendingClubDataset(BaseDataset):
    """Adapter that wraps the LendingClub LightningDataModule."""

    spec = get_tabular_dataset_spec("lending_club")

    def __init__(self, data_dir: str = "data/", seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self._x_train: np.ndarray
        self._y_train: np.ndarray
        self._x_test: np.ndarray
        self._y_test: np.ndarray

    def load(self) -> None:
        from training.datamodules.lending_club import LendingClubDataModule

        dm = LendingClubDataModule(
            filepath=str(Path(self.data_dir) / "LendingClub" / "raw.parquet"),
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


class AdultDataset(BaseDataset):
    """Adapter that reuses the existing Lightning Adult datamodule."""

    spec = get_tabular_dataset_spec("adult")

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
            filepath=str(Path(self.data_dir) / "Adult" / "raw.parquet"),
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


class MNISTDataset(BaseDataset):
    """Adapter that wraps the MNIST LightningDataModule and flattens images."""

    def __init__(self, data_dir: str = "data/", seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.seed = seed
        self._x_train: np.ndarray
        self._y_train: np.ndarray
        self._x_test: np.ndarray
        self._y_test: np.ndarray

    def load(self) -> None:
        from training.datamodules.mnist import MNISTDataModule

        dm = MNISTDataModule(
            data_dir=self.data_dir,
            batch_size=256,
            val_size=5000,
            num_workers=0,
            augment=False,
        )
        dm.setup()
        train_subset = dm.train_ds
        if hasattr(train_subset, "tensors"):
            x_train, y_train = train_subset.tensors
            if x_train.ndim == 4:
                x_train = x_train[:, 0]
        else:
            train_base = train_subset.dataset
            train_idx = torch.as_tensor(train_subset.indices, dtype=torch.long)
            x_train = train_base.data[train_idx].float().div(255.0)
            y_train = train_base.targets[train_idx]
        x_test, y_test = dm.test_ds.data, dm.test_ds.targets

        self._x_train = x_train.view(x_train.shape[0], -1).cpu().numpy().astype(np.float32)
        self._y_train = y_train.cpu().numpy().astype(np.int64)
        self._x_test = (x_test.float().div(255.0).view(x_test.shape[0], -1).cpu().numpy().astype(np.float32))
        self._y_test = y_test.cpu().numpy().astype(np.int64)
        self._loaded = True

    def get_train(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_loaded()
        return self._x_train, self._y_train

    def get_test(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_loaded()
        return self._x_test, self._y_test
