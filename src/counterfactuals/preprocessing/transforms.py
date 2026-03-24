"""Shared representation transforms for benchmark-time preprocessing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA

from counterfactuals.core.interfaces import ModelInterface


class RepresentationTransform(ABC):
    """Bidirectional feature-space transform used by methods in benchmarks."""

    name: str

    @abstractmethod
    def fit(self, x_train: np.ndarray) -> None:
        """Fit transform parameters on training features."""

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Map evaluation-space input to generation space."""

    @abstractmethod
    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Map generation-space vectors back to evaluation space."""


class IdentityTransform(RepresentationTransform):
    """No-op transform."""

    name = "identity"

    def fit(self, x_train: np.ndarray) -> None:
        del x_train

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float32)


class PCATransform(RepresentationTransform):
    """Principal component transform with reversible projection."""

    name = "pca"

    def __init__(
        self,
        n_components: Optional[float | int] = 0.99,
        svd_solver: str = "full",
        whiten: bool = False,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        self._pca = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            whiten=whiten,
            random_state=random_state,
        )

    def fit(self, x_train: np.ndarray) -> None:
        self._pca.fit(np.asarray(x_train, dtype=np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        was_1d = arr.ndim == 1
        arr_2d = arr.reshape(1, -1) if was_1d else arr
        z = self._pca.transform(arr_2d).astype(np.float32)
        return z[0] if was_1d else z

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=np.float32)
        was_1d = arr.ndim == 1
        arr_2d = arr.reshape(1, -1) if was_1d else arr
        x = self._pca.inverse_transform(arr_2d).astype(np.float32)
        return x[0] if was_1d else x

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(np.sum(self._pca.explained_variance_ratio_))


@dataclass(frozen=True)
class OHEBlockSpec:
    """One-hot block boundaries in flattened feature vectors."""

    start: int
    end: int


def snap_ohe_blocks(x: np.ndarray, blocks: Sequence[OHEBlockSpec]) -> np.ndarray:
    """Project categorical OHE blocks onto valid one-hot vertices via argmax."""
    arr = np.asarray(x, dtype=np.float32)
    arr_2d = arr.reshape(1, -1) if arr.ndim == 1 else arr.copy()

    for block in blocks:
        block_vals = arr_2d[:, block.start:block.end]
        idx = np.argmax(block_vals, axis=1)
        block_vals.fill(0.0)
        block_vals[np.arange(arr_2d.shape[0]), idx] = 1.0
        arr_2d[:, block.start:block.end] = block_vals

    return arr_2d[0] if arr.ndim == 1 else arr_2d


def compas_ohe_blocks() -> list[OHEBlockSpec]:
    """Infer COMPAS one-hot block boundaries from datamodule constants."""
    from training.datamodules.compas import CARDINALITIES, INPUT_TYPES

    blocks: list[OHEBlockSpec] = []
    position = 0
    cardinality_idx = 0
    for feature_type in INPUT_TYPES:
        if feature_type == "numerical":
            position += 1
            continue
        cardinality = CARDINALITIES[cardinality_idx]
        blocks.append(OHEBlockSpec(start=position, end=position + cardinality))
        position += cardinality
        cardinality_idx += 1
    return blocks


def german_credit_ohe_blocks() -> list[OHEBlockSpec]:
    """Infer German Credit one-hot block boundaries from datamodule constants."""
    from training.datamodules.german_credit import CARDINALITIES, INPUT_TYPES

    blocks: list[OHEBlockSpec] = []
    position = 0
    cardinality_idx = 0
    for feature_type in INPUT_TYPES:
        if feature_type == "numerical":
            position += 1
            continue
        cardinality = CARDINALITIES[cardinality_idx]
        blocks.append(OHEBlockSpec(start=position, end=position + cardinality))
        position += cardinality
        cardinality_idx += 1
    return blocks


def adult_ohe_blocks() -> list[OHEBlockSpec]:
    """Infer Adult one-hot block boundaries from datamodule constants."""
    from training.datamodules.adult import CARDINALITIES, INPUT_TYPES

    blocks: list[OHEBlockSpec] = []
    position = 0
    cardinality_idx = 0
    for feature_type in INPUT_TYPES:
        if feature_type == "numerical":
            position += 1
            continue
        cardinality = CARDINALITIES[cardinality_idx]
        blocks.append(OHEBlockSpec(start=position, end=position + cardinality))
        position += cardinality
        cardinality_idx += 1
    return blocks


class InverseTransformModel(ModelInterface):
    """Model adapter that accepts generation-space inputs and predicts in eval space."""

    def __init__(
        self,
        base_model: ModelInterface,
        transform: RepresentationTransform,
        ohe_blocks: Optional[Sequence[OHEBlockSpec]] = None,
    ):
        self.base_model = base_model
        self.transform = transform
        self.ohe_blocks = list(ohe_blocks) if ohe_blocks is not None else None

    def _to_eval_space(self, x: np.ndarray) -> np.ndarray:
        x_eval = self.transform.inverse_transform(x)
        if self.ohe_blocks:
            x_eval = snap_ohe_blocks(x_eval, self.ohe_blocks)
        return np.asarray(x_eval, dtype=np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_eval = self._to_eval_space(x)
        return self.base_model.predict(x_eval)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_eval = self._to_eval_space(x)
        return self.base_model.predict_proba(x_eval)
