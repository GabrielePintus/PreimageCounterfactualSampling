"""LightningDataModule for the Kaggle LendingClub (2007–2011) dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_specs import get_tabular_dataset_spec
from training.datamodules._tabular_utils import compute_inverse_frequency_class_weights

_SPEC = get_tabular_dataset_spec("lending_club")
_NUMERICAL_COLS = list(_SPEC.numerical_features)
_CATEGORICAL_COLS = list(_SPEC.categorical_features)

# Empirically verified from standard Kaggle 2007-2011 filtered to Fully Paid/Charged Off.
# term: 36/60 months (2), grade: A-G (7), home_ownership: MORTGAGE/OWN/RENT/OTHER (4),
# verification_status: Not Verified/Source Verified/Verified (3).
CARDINALITIES = list(_SPEC.cardinalities)

INPUT_TYPES = list(_SPEC.input_types)

N_FEATURES = int(_SPEC.n_features)

# Per-OHE-dimension type annotation (length N_FEATURES = 25).
OHE_FEATURE_TYPES: list = list(_SPEC.ohe_feature_types)


class LendingClubDataModule(L.LightningDataModule):
    """
    LightningDataModule for the Kaggle LendingClub (2007–2011) dataset.

    Binary classification: 0 = Fully Paid, 1 = Charged Off.
    Rows with other loan_status values are dropped.

    Features:
    - Numerical (8): loan_amnt, int_rate, annual_inc, dti,
                     delinq_2yrs, open_acc, pub_rec, revol_util
    - Categorical (4 → 17 OHE dims): term (2), grade (7),
                                      home_ownership (5), verification_status (3)

    Total: N_FEATURES = 25.

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
        filepath: str = "data/LendingClub/raw.parquet",
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

        # Keep only Fully Paid / Charged Off
        df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].reset_index(drop=True)

        # Target: Fully Paid → 0, Charged Off → 1
        y = (df["loan_status"] == "Charged Off").values.astype(np.int64)

        # Clean percentage columns. Depending on the parquet writer, these can
        # arrive as object/string dtype or already numeric.
        for col in ["int_rate", "revol_util"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.rstrip("%"),
                errors="coerce",
            ).astype(np.float32)
        for col in _CATEGORICAL_COLS:
            df[col] = df[col].astype(str).str.strip()

        # Drop rows with NaN in any feature column
        feature_cols = _NUMERICAL_COLS + _CATEGORICAL_COLS
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        y = (df["loan_status"] == "Charged Off").values.astype(np.int64)

        # Fit OHE on full cleaned dataset for stable column assignments.
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        self.ohe.fit(df[_CATEGORICAL_COLS].values)
        X_cat = self.ohe.transform(df[_CATEGORICAL_COLS].values)

        # Numerical features (scaling applied after split)
        X_num = df[_NUMERICAL_COLS].values.astype(np.float32)

        # Full feature matrix: numericals first, then OHE categoricals
        X = np.concatenate([X_num, X_cat], axis=1)  # (N, 25)

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
        num_pos = list(range(len(_NUMERICAL_COLS)))
        self.scaler = StandardScaler()
        self.scaler.fit(X[train_idx][:, num_pos])
        X[:, num_pos] = self.scaler.transform(X[:, num_pos]).astype(np.float32)

        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        self.n_features_out = int(X_train.shape[1])
        self.class_weights = compute_inverse_frequency_class_weights(y_train)
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
