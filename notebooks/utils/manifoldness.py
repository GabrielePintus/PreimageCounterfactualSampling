"""Manifoldness helpers for benchmark notebooks."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

_HELPERS_DIR = Path(__file__).resolve().parent
_NOTEBOOKS_DIR = _HELPERS_DIR.parent
_REPO_ROOT = _NOTEBOOKS_DIR.parent
_SRC_ROOT = _REPO_ROOT / "src"

for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from counterfactuals.benchmarks import create_default_registries
from counterfactuals.utils.config import read_yaml
from scripts.benchmark import _build_torch_model_from_checkpoint

from .io import _resolve_existing_path
from .summaries import manifoldness_summary as summarize_manifoldness


def load_tabular_manifold_resources(
    config_path: str | Path,
    *,
    datasets: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Load tabular train data, predictions, and local density resources by dataset."""
    resolved = _resolve_existing_path(config_path)
    if resolved is None:
        raise FileNotFoundError(f"Missing benchmark config: {config_path}")

    cfg = read_yaml(resolved)
    config_dir = resolved.parent
    dataset_filter = set(datasets) if datasets is not None else None
    registries = create_default_registries()
    resources: dict[str, dict[str, Any]] = {}

    for ds_cfg in cfg.get("datasets", []):
        dataset_name = ds_cfg["name"]
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue

        model_cfg = ds_cfg.get("model", {})
        if model_cfg.get("name") != "tabular_classifier_ckpt":
            continue

        dataset_params = dict(ds_cfg.get("dataset_params", {}))
        dataset_params.setdefault("data_dir", str(_REPO_ROOT / "data"))
        dataset = registries["dataset"].create(dataset_name, **dataset_params)
        dataset.load()
        x_train, _ = dataset.get_train()

        params = dict(model_cfg.get("params", {}))
        checkpoint_path = Path(params["checkpoint"])
        if not checkpoint_path.is_absolute():
            candidates = [
                config_dir / checkpoint_path,
                _REPO_ROOT / checkpoint_path,
                Path.cwd() / checkpoint_path,
            ]
            for candidate in candidates:
                if candidate.exists():
                    checkpoint_path = candidate.resolve()
                    break

        model = _build_torch_model_from_checkpoint(
            checkpoint=str(checkpoint_path),
            device=device,
            dataset_module=params.get("dataset_module", dataset_name),
            hidden_dims=list(params.get("hidden_dims", [32, 8])),
            dropout=float(params.get("dropout", 0.2)),
        )
        y_train_pred = np.asarray(model.predict(x_train), dtype=np.int64)

        nn_all = NearestNeighbors(n_neighbors=min(5, len(x_train)), metric="euclidean")
        nn_all.fit(x_train)

        lof_neighbors = max(2, min(20, len(x_train) - 1))
        lof_all = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True)
        lof_all.fit(x_train)

        nn_target: dict[int, NearestNeighbors] = {}
        for cls in np.unique(y_train_pred):
            x_cls = x_train[y_train_pred == cls]
            if len(x_cls) == 0:
                continue
            nn_cls = NearestNeighbors(n_neighbors=min(5, len(x_cls)), metric="euclidean")
            nn_cls.fit(x_cls)
            nn_target[int(cls)] = nn_cls

        resources[dataset_name] = {
            "x_train": x_train,
            "y_train_pred": y_train_pred,
            "nn_all": nn_all,
            "nn_target": nn_target,
            "lof_all": lof_all,
        }

    if dataset_filter is not None:
        missing = sorted(dataset_filter.difference(resources))
        if missing:
            raise KeyError(f"No tabular manifold resources found for datasets: {missing}")
    return resources


def attach_tabular_manifoldness_metrics(
    df: pd.DataFrame,
    *,
    resources_by_dataset: dict[str, dict[str, Any]],
    dataset_col: str = "dataset",
    method_col: str = "method_label",
    success_only_rows: bool = True,
    cf_prefix: str = "x_cf_",
    target_pred_col: str = "y_cf",
) -> pd.DataFrame:
    """Attach kNN and LOF manifoldness metrics to successful tabular CF rows."""
    data = df[df["success"]].copy() if success_only_rows else df.copy()
    if data.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for dataset_name, dataset_df in data.groupby(dataset_col, sort=False):
        if dataset_name not in resources_by_dataset:
            continue
        resources = resources_by_dataset[dataset_name]
        n_features = resources["x_train"].shape[1]
        cf_cols = [f"{cf_prefix}{i}" for i in range(n_features)]
        missing = [col for col in cf_cols if col not in dataset_df.columns]
        if missing:
            raise ValueError(
                f"Dataset {dataset_name!r} is missing expected counterfactual feature columns, starting with {missing[0]!r}."
            )

        x_cf = dataset_df[cf_cols].to_numpy(dtype=np.float32)
        knn_all_dist, _ = resources["nn_all"].kneighbors(x_cf)
        lof_all_score = resources["lof_all"].score_samples(x_cf)
        neg_lof_all_score = -lof_all_score

        knn_target_mean = np.full(len(dataset_df), np.nan, dtype=np.float32)
        target_classes = dataset_df[target_pred_col].to_numpy(dtype=np.int64)
        for cls in np.unique(target_classes):
            class_rows = np.where(target_classes == cls)[0]
            nn_cls = resources["nn_target"].get(int(cls))
            if nn_cls is None:
                continue
            dist_cls, _ = nn_cls.kneighbors(x_cf[class_rows])
            knn_target_mean[class_rows] = dist_cls.mean(axis=1)

        out = dataset_df.copy()
        out["manifold_knn5_all"] = knn_all_dist.mean(axis=1)
        out["manifold_knn5_target_pred"] = knn_target_mean
        out["manifold_lof_score_all"] = lof_all_score
        out["manifold_neg_lof_all"] = neg_lof_all_score
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


__all__ = [
    "attach_tabular_manifoldness_metrics",
    "load_tabular_manifold_resources",
    "summarize_manifoldness",
]
