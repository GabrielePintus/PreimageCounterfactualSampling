"""LightningDataModule for the UCI Statlog German Credit dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_specs import get_tabular_dataset_spec
from training.datamodules._tabular_utils import compute_inverse_frequency_class_weights

_SPEC = get_tabular_dataset_spec("german_credit")
_NUMERICAL_COLS = list(_SPEC.numerical_features)
_CATEGORICAL_COLS = list(_SPEC.categorical_features)

# Empirically verified from the UCI German Credit dataset (1000 samples).
# status(4), credit_history(5), purpose(10), savings(5), employment(5),
# personal_status(4, A95 absent), other_debtors(3), property(4),
# other_installment_plans(3), housing(3), job(4), telephone(2), foreign_worker(2).
CARDINALITIES = list(_SPEC.cardinalities)

INPUT_TYPES = list(_SPEC.input_types)

N_FEATURES = int(_SPEC.n_features)

# Per-OHE-dimension type annotation (length N_FEATURES = 61).
OHE_FEATURE_TYPES: list = list(_SPEC.ohe_feature_types)


class GermanCreditDataModule(L.LightningDataModule):
    """
    LightningDataModule for the UCI Statlog German Credit dataset.

    Binary classification: 0 = good credit risk, 1 = bad credit risk.
    (Original encoding: 1=good, 2=bad; remapped as y = target - 1.)

    No row filters are applied; all 1000 samples are used.

    Features:
    - Numerical (7): duration, credit_amount, installment_rate, present_residence,
                     age, num_credits, num_liable
    - Categorical (13 → 54 OHE dims): status(4), credit_history(5), purpose(10),
      savings(5), employment(5), personal_status(4), other_debtors(3), property(4),
      other_installment_plans(3), housing(3), job(4), telephone(2), foreign_worker(2)

    Total: N_FEATURES = 61.

    Preprocessing:
    - OHE fitted on full dataset (stable column assignments across splits).
    - StandardScaler fitted on train numerical features only.
    """

    INPUT_TYPES = INPUT_TYPES
    CARDINALITIES = CARDINALITIES
    N_FEATURES = N_FEATURES
    OHE_FEATURE_TYPES = OHE_FEATURE_TYPES

    def __init__(
        self,
        filepath: str = "data/GermanCredit/raw.parquet",
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

        # Target: 1=good → 0, 2=bad → 1
        y = (df["target"] == 2).values.astype(np.int64)

        # Fit OHE on full dataset for stable column assignments across splits.
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        self.ohe.fit(df[_CATEGORICAL_COLS].astype(str).values)
        X_cat = self.ohe.transform(df[_CATEGORICAL_COLS].astype(str).values)

        # Numerical features (scaling applied after split)
        X_num = df[_NUMERICAL_COLS].values.astype(np.float32)

        # Full feature matrix: numericals first, then OHE categoricals
        X = np.concatenate([X_num, X_cat], axis=1)

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
