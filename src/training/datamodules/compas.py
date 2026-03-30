"""LightningDataModule for the ProPublica COMPAS (recidivism) dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Features used (standard ProPublica subset)
_NUMERICAL_COLS = ["age", "priors_count", "c_days_from_compas", "days_b_screening_arrest"]
_CATEGORICAL_COLS = ["sex", "race", "c_charge_degree"]

# Empirically verified from the filtered ProPublica dataset.
# sex: Male/Female (2), race: African-American/Asian/Caucasian/Hispanic/Native American/Other (6),
# c_charge_degree: F/M (2).
CARDINALITIES = [2, 6, 2]

INPUT_TYPES = (
    ["numerical"] * len(_NUMERICAL_COLS)
    + ["categorical"] * len(_CATEGORICAL_COLS)
)

N_FEATURES = len(_NUMERICAL_COLS) + sum(CARDINALITIES)  # 4 + 10 = 14

# Per-OHE-dimension type annotation (length N_FEATURES = 14).
OHE_FEATURE_TYPES: list = (
    ["numerical"] * len(_NUMERICAL_COLS)
    + ["categorical"] * sum(CARDINALITIES)
)


class CompasDataModule(L.LightningDataModule):
    """
    LightningDataModule for the ProPublica COMPAS recidivism dataset.

    Binary classification: two_year_recid (0 = no recidivism, 1 = recidivated within 2 years).

    Standard ProPublica preprocessing filters are applied:
    - |days_b_screening_arrest| <= 30
    - is_recid != -1
    - c_charge_degree != "O" (ordinary traffic)
    - score_text != "N/A"

    Features:
    - Numerical (4): age, priors_count, c_days_from_compas, days_b_screening_arrest
    - Categorical (3 → 10 OHE dims): sex (2), race (6), c_charge_degree (2)

    Total: N_FEATURES = 14.

    Preprocessing:
    - OHE fitted on full cleaned dataset (stable column assignments across splits).
    - StandardScaler fitted on train numerical features only.
    """

    INPUT_TYPES = INPUT_TYPES
    CARDINALITIES = CARDINALITIES
    N_FEATURES = N_FEATURES
    OHE_FEATURE_TYPES = OHE_FEATURE_TYPES

    def __init__(
        self,
        filepath: str = "data/Compas/raw.parquet",
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

        # Target
        y = df["two_year_recid"].values.astype(np.int64)

        # Fit OHE on full cleaned dataset for stable column assignments.
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        self.ohe.fit(df[_CATEGORICAL_COLS].astype(str).values)
        X_cat = self.ohe.transform(df[_CATEGORICAL_COLS].astype(str).values)

        # Numerical features (raw, scaling applied after split)
        X_num = df[_NUMERICAL_COLS].values.astype(np.float32)

        # Full feature matrix: numericals first, then OHE categoricals
        X = np.concatenate([X_num, X_cat], axis=1)  # (N, 14)

        # Shuffle and split
        rng = np.random.default_rng(self.hparams.seed)
        idx = rng.permutation(len(df))
        n = len(df)
        n_test = int(n * self.hparams.test_fraction)
        n_val = int(n * self.hparams.val_fraction)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        # StandardScaler on numerical positions, fit on train only.
        num_pos = list(range(len(_NUMERICAL_COLS)))  # [0, 1, 2, 3]
        self.scaler = StandardScaler()
        self.scaler.fit(X[train_idx][:, num_pos])
        X[:, num_pos] = self.scaler.transform(X[:, num_pos]).astype(np.float32)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        self.n_features_out = int(X_train.shape[1])
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
