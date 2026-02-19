"""LightningDataModule for MNIST."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    """
    LightningDataModule for MNIST.

    Provides standard MNIST with optional augmentation.
    Splits the 60k training set into train/val (55k/5k by default).

    Parameters
    ----------
    data_dir : str
        Root directory for MNIST download/cache.
    batch_size : int
        Batch size for all dataloaders.
    val_size : int
        Number of samples reserved for validation.
    num_workers : int
        Workers for DataLoader.
    augment : bool
        If True, apply RandomAffine + GaussianNoise augmentation to training set.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        val_size: int = 5000,
        num_workers: int = 4,
        augment: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)

    def _base_transform(self):
        return transforms.ToTensor()

    def _train_transform(self):
        if not self.hparams.augment:
            return self._base_transform()
        return transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        full_train = datasets.MNIST(
            self.data_dir, train=True, transform=self._train_transform()
        )
        val_size = self.hparams.val_size
        train_size = len(full_train) - val_size
        self.train_ds, self.val_ds = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        # Validation uses base (no augmentation) transform
        self.val_ds.dataset = datasets.MNIST(
            self.data_dir, train=True, transform=self._base_transform()
        )
        self.test_ds = datasets.MNIST(
            self.data_dir, train=False, transform=self._base_transform()
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
