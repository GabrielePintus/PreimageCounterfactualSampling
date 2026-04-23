"""Shared helpers for tabular LightningDataModules."""

from __future__ import annotations

import numpy as np


def compute_inverse_frequency_class_weights(y_train: np.ndarray) -> list[float]:
    """Return inverse-frequency class weights normalized to mean 1."""
    y_arr = np.asarray(y_train, dtype=np.int64).reshape(-1)
    if y_arr.size == 0:
        raise ValueError("Cannot compute class weights from an empty training target array.")

    counts = np.bincount(y_arr)
    if np.any(counts == 0):
        raise ValueError("Cannot compute class weights when some classes are absent from the train split.")

    weights = y_arr.size / (len(counts) * counts.astype(np.float64))
    return weights.astype(np.float32).tolist()
