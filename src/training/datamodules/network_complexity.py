"""Deterministic synthetic data for the CertCF network-complexity experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataset_specs import get_tabular_dataset_spec
from training.datamodules._tabular_utils import compute_inverse_frequency_class_weights

_SPEC = get_tabular_dataset_spec("network_complexity")


def dataset_recipe(
    *,
    n_train: int = 10_000,
    n_validation: int = 1_000,
    n_test: int = 1_000,
    n_features: int = 32,
    seed: int = 42,
    class_sep: float = 1.5,
    flip_y: float = 0.01,
    n_queries: int = 1_000,
    queries_per_class: int = 500,
) -> dict[str, Any]:
    """Return the canonical, JSON-serializable dataset recipe."""
    return {
        "n_samples": int(n_train + n_validation + n_test),
        "n_train": int(n_train),
        "n_validation": int(n_validation),
        "n_test": int(n_test),
        "n_features": int(n_features),
        "n_informative": int(n_features),
        "n_redundant": 0,
        "n_repeated": 0,
        "n_classes": 2,
        "n_clusters_per_class": 2,
        "weights": [0.5, 0.5],
        "class_sep": float(class_sep),
        "flip_y": float(flip_y),
        "seed": int(seed),
        "n_queries": int(n_queries),
        "queries_per_class": int(queries_per_class),
    }


def recipe_fingerprint(recipe: dict[str, Any]) -> str:
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_dataset_metadata(cache_path: str | Path) -> dict[str, Any]:
    """Read cache metadata without enabling pickle."""
    with np.load(Path(cache_path), allow_pickle=False) as cached:
        return json.loads(str(cached["metadata_json"].item()))


def prepare_network_complexity_dataset(
    cache_path: str | Path,
    *,
    n_train: int = 10_000,
    n_validation: int = 1_000,
    n_test: int = 1_000,
    n_features: int = 32,
    seed: int = 42,
    class_sep: float = 1.5,
    flip_y: float = 0.01,
    n_queries: int = 1_000,
    queries_per_class: int = 500,
    force: bool = False,
) -> dict[str, Any]:
    """Generate, split, scale, and atomically cache the shared synthetic dataset."""
    cache_path = Path(cache_path)
    recipe = dataset_recipe(
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        n_features=n_features,
        seed=seed,
        class_sep=class_sep,
        flip_y=flip_y,
        n_queries=n_queries,
        queries_per_class=queries_per_class,
    )
    fingerprint = recipe_fingerprint(recipe)
    if cache_path.exists() and not force:
        metadata = read_dataset_metadata(cache_path)
        if metadata.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Dataset cache {cache_path} was made with a different recipe. "
                "Use force=True or a different output directory."
            )
        return metadata

    split_sizes = {
        "train": int(n_train),
        "validation": int(n_validation),
        "test": int(n_test),
    }
    if n_features <= 0 or any(size <= 0 for size in split_sizes.values()):
        raise ValueError("n_features and every split size must be positive")
    if any(size % 2 for size in split_sizes.values()):
        raise ValueError("Every split size must be even for exact binary class balance")
    if not 0.0 <= float(flip_y) <= 1.0:
        raise ValueError("flip_y must lie in [0, 1]")
    if n_queries != 2 * queries_per_class:
        raise ValueError("n_queries must equal 2 * queries_per_class for balanced binary queries")
    if queries_per_class > n_test // 2:
        raise ValueError("queries_per_class cannot exceed the per-class test size")

    # Generate exactly the retained dataset size. Built-in flip_y can perturb
    # class counts, so apply the same noise rate symmetrically: flip an equal
    # number of labels in both directions and preserve exact 50/50 balance.
    total_required = int(n_train + n_validation + n_test)
    required_per_class = total_required // 2
    x_candidates, y_candidates = make_classification(
        n_samples=total_required,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        class_sep=class_sep,
        flip_y=0.0,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    flips_per_class = int(round(float(flip_y) * total_required / 2.0))
    if flips_per_class > required_per_class:
        raise ValueError("flip_y is too large for symmetric balanced label noise")
    y_candidates = y_candidates.astype(np.int64, copy=True)
    for label in (0, 1):
        label_indices = np.flatnonzero(y_candidates == label)
        flip_indices = rng.choice(label_indices, size=flips_per_class, replace=False)
        y_candidates[flip_indices] = 1 - label

    retained_by_class = []
    for label in (0, 1):
        candidates = np.flatnonzero(y_candidates == label)
        if len(candidates) != required_per_class:
            raise RuntimeError("Symmetric label-noise generation did not preserve class balance")
        retained_by_class.append(rng.permutation(candidates))

    train_per_class = n_train // 2
    validation_per_class = n_validation // 2
    test_per_class = n_test // 2

    def balanced_split(start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.concatenate(
            [class_indices[start : start + count] for class_indices in retained_by_class]
        )
        rng.shuffle(indices)
        return (
            x_candidates[indices].astype(np.float32),
            y_candidates[indices].astype(np.int64),
        )

    x_train_raw, y_train = balanced_split(0, train_per_class)
    x_val_raw, y_val = balanced_split(train_per_class, validation_per_class)
    x_test_raw, y_test = balanced_split(
        train_per_class + validation_per_class,
        test_per_class,
    )

    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    query_parts: list[np.ndarray] = []
    for label in (0, 1):
        candidates = np.flatnonzero(y_test == label)
        if len(candidates) < queries_per_class:
            raise ValueError(
                f"Test split has only {len(candidates)} rows for class {label}; "
                f"cannot choose {queries_per_class} fixed queries."
            )
        query_parts.append(rng.choice(candidates, size=queries_per_class, replace=False))
    query_indices = np.concatenate(query_parts).astype(np.int64)
    rng.shuffle(query_indices)

    metadata = {
        "fingerprint": fingerprint,
        "recipe": recipe,
        "split_sizes": {
            "train": int(n_train),
            "validation": int(n_validation),
            "test": int(n_test),
        },
        "split_class_counts": {
            "train": np.bincount(y_train, minlength=2).astype(int).tolist(),
            "validation": np.bincount(y_val, minlength=2).astype(int).tolist(),
            "test": np.bincount(y_test, minlength=2).astype(int).tolist(),
            "queries": np.bincount(y_test[query_indices], minlength=2).astype(int).tolist(),
        },
        "scaler_fit_split": "train",
        "rows_generated": int(total_required),
        "retained_rows": int(total_required),
        "symmetric_label_flips": int(2 * flips_per_class),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary_path,
        x_train=x_train,
        y_train=y_train.astype(np.int64),
        x_validation=x_val,
        y_validation=y_val.astype(np.int64),
        x_test=x_test,
        y_test=y_test.astype(np.int64),
        x_train_raw=x_train_raw.astype(np.float32),
        x_validation_raw=x_val_raw.astype(np.float32),
        x_test_raw=x_test_raw.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        query_indices=query_indices,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary_path, cache_path)
    return metadata


class NetworkComplexityDataModule(L.LightningDataModule):
    """Lightning view of the cached network-complexity dataset."""

    INPUT_TYPES = list(_SPEC.input_types)
    CARDINALITIES = list(_SPEC.cardinalities)
    N_FEATURES = int(_SPEC.n_features)
    OHE_FEATURE_TYPES = list(_SPEC.ohe_feature_types)

    def __init__(
        self,
        cache_path: str = "results/network_complexity/data/dataset.npz",
        batch_size: int = 128,
        num_workers: int = 0,
        seed: int = 42,
        n_train: int = 10_000,
        n_validation: int = 1_000,
        n_test: int = 1_000,
        n_features: int = 32,
        class_sep: float = 1.5,
        flip_y: float = 0.01,
        n_queries: int = 1_000,
        queries_per_class: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        prepare_network_complexity_dataset(
            self.hparams.cache_path,
            n_train=self.hparams.n_train,
            n_validation=self.hparams.n_validation,
            n_test=self.hparams.n_test,
            n_features=self.hparams.n_features,
            seed=self.hparams.seed,
            class_sep=self.hparams.class_sep,
            flip_y=self.hparams.flip_y,
            n_queries=self.hparams.n_queries,
            queries_per_class=self.hparams.queries_per_class,
        )

    def setup(self, stage: str | None = None) -> None:
        del stage
        with np.load(self.hparams.cache_path, allow_pickle=False) as cached:
            x_train = cached["x_train"].astype(np.float32)
            y_train = cached["y_train"].astype(np.int64)
            x_val = cached["x_validation"].astype(np.float32)
            y_val = cached["y_validation"].astype(np.int64)
            x_test = cached["x_test"].astype(np.float32)
            y_test = cached["y_test"].astype(np.int64)
            self.query_indices = cached["query_indices"].astype(np.int64)
            self.scaler_mean = cached["scaler_mean"].astype(np.float64)
            self.scaler_scale = cached["scaler_scale"].astype(np.float64)

        self.n_features_out = int(x_train.shape[1])
        self.class_weights = compute_inverse_frequency_class_weights(y_train)
        self.train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        self.val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        self.test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(int(self.hparams.seed))
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            generator=generator,
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
