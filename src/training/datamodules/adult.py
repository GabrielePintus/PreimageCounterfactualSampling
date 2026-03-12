"""LightningDataModule for the UCI Adult (Census Income) dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Column ordering (matches UCI Adult schema)
_ALL_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]
_NUMERICAL = {"age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"}
_CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]

# Module-level constants for TabularClassifier init.
# Cardinalities are computed after NaN-row removal (empirically verified, see data/Adult/README.md).
INPUT_TYPES = ["numerical" if c in _NUMERICAL else "categorical" for c in _ALL_COLS]
CARDINALITIES = [7, 16, 7, 14, 6, 5, 2, 41]  # workclass, education, marital-status,
                                              # occupation, relationship, race, sex, native-country
# Computed from fetch_openml("adult", version=2) after dropna() — NaN rows are dropped,
# so "missing" is never a category. The CSV-based count differs because it included NaN rows.
N_FEATURES = len(_NUMERICAL) + sum(CARDINALITIES)  # 6 + 98 = 104

# Per-OHE-dimension type annotation (length N_FEATURES = 108).
# Used downstream for MAD computation: numerical dims get MAD-scaled, categorical dims get 1.0.
_cat_iter = iter(CARDINALITIES)
OHE_FEATURE_TYPES: list = []
for _t in INPUT_TYPES:
    if _t == "numerical":
        OHE_FEATURE_TYPES.append("numerical")
    else:
        OHE_FEATURE_TYPES.extend(["categorical"] * next(_cat_iter))


class AdultDataModule(L.LightningDataModule):
    """
    LightningDataModule for the UCI Adult (Census Income) dataset.

    Binary classification: income <=50K (class 0) vs >50K (class 1).

    Preprocessing:
    - Rows with any NaN in the 14 feature columns are dropped.
    - Categorical features: one-hot encoded via OneHotEncoder fit on the full
      cleaned dataset (stable column assignments across splits).
    - Numerical features: StandardScaler fit on the train split only.
    - Output feature dimension: N_FEATURES = 108 (6 numerical + 102 OHE).

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
    N_FEATURES = N_FEATURES
    OHE_FEATURE_TYPES = OHE_FEATURE_TYPES

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
        df = bunch.frame[_ALL_COLS + ["class"]].dropna().reset_index(drop=True)

        # Target: <=50K → 0, >50K → 1
        y = (df["class"] == ">50K").astype(int).values.astype(np.int64)

        # Shuffle and split
        rng = np.random.default_rng(self.hparams.seed)
        idx = rng.permutation(len(df))
        n = len(df)
        n_test = int(n * self.hparams.test_fraction)
        n_val = int(n * self.hparams.val_fraction)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        # Fit OHE on full cleaned dataset for stable column assignments.
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        self.ohe.fit(df[_CATEGORICAL].astype(str).values)

        # OHE transform: shape (N, 102), columns ordered as in _CATEGORICAL.
        X_ohe = self.ohe.transform(df[_CATEGORICAL].astype(str).values)

        # Map each categorical column to its slice in X_ohe.
        _cat_offsets: dict = {}
        offset = 0
        for j, col in enumerate(_CATEGORICAL):
            card = len(self.ohe.categories_[j])
            _cat_offsets[col] = (offset, offset + card)
            offset += card

        # Build full feature matrix in _ALL_COLS order (interleaved numerical + OHE blocks).
        parts = []
        for col in _ALL_COLS:
            if col in _NUMERICAL:
                parts.append(df[[col]].values.astype(np.float32))
            else:
                s, e = _cat_offsets[col]
                parts.append(X_ohe[:, s:e])
        X = np.concatenate(parts, axis=1)  # (N, 108)

        # StandardScaler on numerical positions, fit on train only.
        num_pos = [i for i, t in enumerate(OHE_FEATURE_TYPES) if t == "numerical"]
        self.scaler = StandardScaler()
        self.scaler.fit(X[train_idx][:, num_pos].astype(np.float64))
        X[:, num_pos] = self.scaler.transform(
            X[:, num_pos].astype(np.float64)
        ).astype(np.float32)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

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
