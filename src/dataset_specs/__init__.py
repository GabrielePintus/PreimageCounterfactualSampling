"""Shared dataset schema metadata used by training and benchmarking."""

from .tabular import OHEBlockSpec, TABULAR_DATASET_SPECS, TabularDatasetSpec, get_tabular_dataset_spec

__all__ = [
    "OHEBlockSpec",
    "TabularDatasetSpec",
    "TABULAR_DATASET_SPECS",
    "get_tabular_dataset_spec",
]
