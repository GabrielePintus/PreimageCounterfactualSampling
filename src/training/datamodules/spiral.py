"""LightningDataModule for the synthetic 2D spiral dataset."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import lightning as L

from preimage_sampling.utils.data import make_spiral


class SpiralDataModule(L.LightningDataModule):
    """
    LightningDataModule for the synthetic 2D multi-class spiral dataset.

    Parameters
    ----------
    n_samples_per_class : int
        Samples per class arm.
    n_classes : int
        Number of spiral arms / classes.
    noise : float
        Angular noise for spiral generation.
    seed : int
        Random seed for reproducibility.
    val_fraction : float
        Fraction of data held out for validation.
    test_fraction : float
        Fraction of data held out for testing.
    batch_size : int
        Batch size for all dataloaders.
    num_workers : int
        Workers for DataLoader.
    """

    def __init__(
        self,
        n_samples_per_class: int = 300,
        n_classes: int = 10,
        noise: float = 0.1,
        seed: int = 42,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        X, y = make_spiral(
            n_samples_per_class=self.hparams.n_samples_per_class,
            n_classes=self.hparams.n_classes,
            noise=self.hparams.noise,
            seed=self.hparams.seed,
        )
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        full_ds = TensorDataset(X_t, y_t)

        n = len(full_ds)
        n_test = int(n * self.hparams.test_fraction)
        n_val = int(n * self.hparams.val_fraction)
        n_train = n - n_val - n_test

        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

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
