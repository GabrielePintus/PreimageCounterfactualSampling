"""LightningDataModule for the FICO HELOC (Home Equity Line of Credit) dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.preprocessing import StandardScaler

_FEATURE_COLS = [
    "estimate_of_risk",
    "months_since_first_trade",
    "months_since_last_trade",
    "average_duration_of_resolution",
    "number_of_satisfactory_trades",
    "nr_trades_insolvent_for_over_60_days",
    "nr_trades_insolvent_for_over_90_days",
    "percentage_of_legal_trades",
    "months_since_last_illegal_trade",
    "maximum_illegal_trades_over_last_year",
    "maximum_illegal_trades",
    "nr_total_trades",
    "nr_trades_initiated_in_last_year",
    "percentage_of_installment_trades",
    "months_since_last_inquiry_not_recent",
    "nr_inquiries_in_last_6_months",
    "nr_inquiries_in_last_6_months_not_recent",
    "net_fraction_of_revolving_burden",
    "net_fraction_of_installment_burden",
    "nr_revolving_trades_with_balance",
    "nr_installment_trades_with_balance",
    "nr_banks_with_high_ratio",
    "percentage_trades_with_balance",
]

# All features are numerical — no OHE.
CARDINALITIES: list = []
INPUT_TYPES = ["numerical"] * len(_FEATURE_COLS)
N_FEATURES = len(_FEATURE_COLS)  # 23
OHE_FEATURE_TYPES: list = ["numerical"] * N_FEATURES


class HELOCDataModule(L.LightningDataModule):
    """
    LightningDataModule for the FICO HELOC dataset (via HuggingFace: mstz/heloc).

    Binary classification: 0 = Good (repaid), 1 = Bad (at risk / defaulted).
    Target column: is_at_risk.

    All 23 features are numerical. Missing-value sentinel codes (-7, -8, -9)
    are kept as-is — they encode meaningful absence of credit history.

    Preprocessing:
    - StandardScaler fitted on train features only.
    """

    INPUT_TYPES = INPUT_TYPES
    CARDINALITIES = CARDINALITIES
    N_FEATURES = N_FEATURES
    OHE_FEATURE_TYPES = OHE_FEATURE_TYPES

    def __init__(
        self,
        filepath: str = "data/Heloc/raw.parquet",
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

        y = df["is_at_risk"].values.astype(np.int64)
        X = df[_FEATURE_COLS].values.astype(np.float32)

        # Shuffle and split
        rng = np.random.default_rng(self.hparams.seed)
        idx = rng.permutation(len(df))
        n = len(df)
        n_test = int(n * self.hparams.test_fraction)
        n_val = int(n * self.hparams.val_fraction)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        # StandardScaler fitted on train only
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
