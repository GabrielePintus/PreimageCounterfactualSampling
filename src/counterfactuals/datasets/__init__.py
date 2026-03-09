"""Dataset adapters."""

from .base_dataset import BaseDataset, NumpyDataset
from .loaders import AdultDataset

__all__ = ["BaseDataset", "NumpyDataset", "AdultDataset"]
