"""LightningDataModule for the Wisconsin Breast Cancer dataset."""

from pathlib import Path

import lightning as L
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataset_specs import get_tabular_dataset_spec

_SPEC = get_tabular_dataset_spec("wisconsin_breast_cancer")
_FEATURE_COLS = list(_SPEC.feature_names)

CARDINALITIES: list = list(_SPEC.cardinalities)
INPUT_TYPES = list(_SPEC.input_types)
N_FEATURES = int(_SPEC.n_features)
OHE_FEATURE_TYPES: list = list(_SPEC.ohe_feature_types)


class WisconsinBreastCancerDataModule(L.LightningDataModule):
    """
    LightningDataModule for the Wisconsin Diagnostic Breast Cancer dataset.

    Binary classification:
    - 0 = benign
    - 1 = malignant

    All 30 features are numerical. Preprocessing mirrors the other tabular
    numerical-only datasets:
    - parquet input with a fixed schema
    - random train/val/test split from a seeded permutation
    - StandardScaler fit on train only
    """

    INPUT_TYPES = INPUT_TYPES
    CARDINALITIES = CARDINALITIES
    N_FEATURES = N_FEATURES
    OHE_FEATURE_TYPES = OHE_FEATURE_TYPES

    def __init__(
        self,
        filepath: str = "data/WisconsinBreastCancer/raw.parquet",
        batch_size: int = 256,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if not Path(self.hparams.filepath).exists():
            raise FileNotFoundError(f"Parquet file not found: {self.hparams.filepath}")

    def setup(self, stage=None) -> None:
        import pandas as pd

        df = pd.read_parquet(self.hparams.filepath)

        y = df[_SPEC.target_name].values.astype(np.int64)
        X = df[_FEATURE_COLS].values.astype(np.float32)

        rng = np.random.default_rng(self.hparams.seed)
        idx = rng.permutation(len(df))
        n = len(df)
        n_test = int(n * self.hparams.test_fraction)
        n_val = int(n * self.hparams.val_fraction)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        self.scaler = StandardScaler()
        self.scaler.fit(X[train_idx])
        X = self.scaler.transform(X).astype(np.float32)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        self.n_features_out = int(X_train.shape[1])
        self.train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        self.test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
