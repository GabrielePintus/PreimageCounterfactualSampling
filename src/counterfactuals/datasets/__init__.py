"""Dataset adapters."""

from .base_dataset import BaseDataset, NumpyDataset
from .loaders import AdultDataset, CompasDataset, GermanCreditDataset, GiveMeSomeCreditDataset, HELOCDataset

__all__ = ["BaseDataset", "NumpyDataset", "AdultDataset", "CompasDataset", "GermanCreditDataset", "GiveMeSomeCreditDataset", "HELOCDataset"]
