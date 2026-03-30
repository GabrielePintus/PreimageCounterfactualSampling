"""LightningDataModule for the Give Me Some Credit (Kaggle) dataset."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from sklearn.preprocessing import StandardScaler

_FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Columns with missing values — imputed with median (fit on train only).
_IMPUTE_COLS = ["MonthlyIncome", "NumberOfDependents"]

# All features are numerical — no OHE.
CARDINALITIES: list = []
INPUT_TYPES = ["numerical"] * len(_FEATURE_COLS)
N_FEATURES = len(_FEATURE_COLS)  # 10
OHE_FEATURE_TYPES: list = ["numerical"] * N_FEATURES


class GiveMeSomeCreditDataModule(L.LightningDataModule):
    """
    LightningDataModule for the Give Me Some Credit dataset (Kaggle competition).

    Binary classification: 0 = no delinquency, 1 = serious delinquency within 2 years.
    Target column: SeriousDlqin2yrs.

    All 10 features are numerical. Missing values in MonthlyIncome and NumberOfDependents
    are imputed with the train-split median.

    Preprocessing:
    - Median imputation fitted on train split only.
    - StandardScaler fitted on train split only (after imputation).
    """

    INPUT_TYPES = INPUT_TYPES
    CARDINALITIES = CARDINALITIES
    N_FEATURES = N_FEATURES
    OHE_FEATURE_TYPES = OHE_FEATURE_TYPES

    def __init__(
        self,
        filepath: str = "data/Give Me Some Credit/raw.parquet",
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

        y = df["SeriousDlqin2yrs"].values.astype(np.int64)
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

        # Median imputation — fit on train only
        impute_positions = [_FEATURE_COLS.index(c) for c in _IMPUTE_COLS]
        self.impute_medians = np.nanmedian(X[train_idx][:, impute_positions], axis=0)
        for pos, median in zip(impute_positions, self.impute_medians):
            nan_mask = np.isnan(X[:, pos])
            X[nan_mask, pos] = median

        # StandardScaler — fit on train only
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
