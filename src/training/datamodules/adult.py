"""LightningDataModule for the UCI Adult (Census Income) dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Column ordering (matches UCI Adult schema)
_ALL_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]
_NUMERICAL = {"age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"}
_CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]

# Module-level constants: reference values for TabularClassifier init.
# Cardinalities are computed after NaN → "missing" fill + LabelEncoder on the full dataset.
INPUT_TYPES = ["numerical" if c in _NUMERICAL else "categorical" for c in _ALL_COLS]
CARDINALITIES = [9, 16, 7, 15, 6, 5, 2, 42]  # workclass, education, marital-status,
                                               # occupation, relationship, race, sex, native-country


class AdultDataModule(L.LightningDataModule):
    """
    LightningDataModule for the UCI Adult (Census Income) dataset.

    Binary classification: income <=50K (class 0) vs >50K (class 1).

    Preprocessing:
    - Categorical NaN → "missing"; integer-encoded via LabelEncoder (fit on full dataset).
    - Numerical: StandardScaler (fit on train split only).

    Parameters
    ----------
    data_dir : str
        Cache directory for sklearn OpenML download.
    batch_size : int
        Batch size for all dataloaders.
    val_fraction : float
        Fraction of data held out for validation.
    test_fraction : float
        Fraction of data held out for testing.
    seed : int
        Random seed for the train/val/test split permutation.
    num_workers : int
        Workers for DataLoader.
    """

    INPUT_TYPES = INPUT_TYPES
    CARDINALITIES = CARDINALITIES

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 256,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        fetch_openml("adult", version=2, data_home=self.hparams.data_dir, as_frame=True)

    def setup(self, stage=None):
        bunch = fetch_openml(
            "adult", version=2, data_home=self.hparams.data_dir, as_frame=True
        )
        df = bunch.frame.copy()

        # Target: <=50K → 0, >50K → 1
        y = (df["class"] == ">50K").astype(int).values.astype(np.int64)

        # Features
        X = df[_ALL_COLS].copy()

        # Encode categoricals (fit on full dataset — all values are seen)
        # astype(object) first to drop the Categorical dtype, then fillna works with new strings
        self.label_encoders = {}
        for col in _CATEGORICAL:
            X[col] = X[col].astype(object).fillna("missing").astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le

        X = X.values.astype(np.float32)

        # Shuffle and split
        rng = np.random.default_rng(self.hparams.seed)
        idx = rng.permutation(len(X))
        n = len(X)
        n_test = int(n * self.hparams.test_fraction)
        n_val = int(n * self.hparams.val_fraction)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        # Normalize numerical columns (fit on train only)
        num_idx = [i for i, t in enumerate(INPUT_TYPES) if t == "numerical"]
        self.scaler = StandardScaler()
        X_train[:, num_idx] = self.scaler.fit_transform(X_train[:, num_idx])
        X_val[:, num_idx] = self.scaler.transform(X_val[:, num_idx])
        X_test[:, num_idx] = self.scaler.transform(X_test[:, num_idx])

        self.train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self.val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        self.test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size,
            shuffle=True, num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers,
        )
