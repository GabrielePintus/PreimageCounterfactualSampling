"""Empirical counterfactual robustness helpers for benchmark notebooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HELPERS_DIR = Path(__file__).resolve().parent
_NOTEBOOKS_DIR = _HELPERS_DIR.parent
_REPO_ROOT = _NOTEBOOKS_DIR.parent
_SRC_ROOT = _REPO_ROOT / "src"

for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from counterfactuals.utils.config import read_yaml
from dataset_specs import get_tabular_dataset_spec

from .io import _resolve_existing_path, feature_columns


def numerical_feature_indices(dataset_name: str) -> np.ndarray:
    """Return encoded-space indices of numerical features for a tabular dataset."""
    spec = get_tabular_dataset_spec(dataset_name)
    return np.asarray(
        [idx for idx, feature_type in enumerate(spec.ohe_feature_types) if feature_type == "numerical"],
        dtype=np.int64,
    )


def _canonical_norm(norm: str | int | float) -> tuple[str, float]:
    """Normalize user-facing norm names to a stable label and numeric code."""
    if isinstance(norm, str):
        key = norm.strip().lower()
        if key in {"l1", "1"}:
            return "l1", 1.0
        if key in {"l2", "2"}:
            return "l2", 2.0
        if key in {"linf", "inf", "infinity", "l∞"}:
            return "linf", float("inf")
    else:
        if norm == 1:
            return "l1", 1.0
        if norm == 2:
            return "l2", 2.0
        if norm == np.inf:
            return "linf", float("inf")
    raise ValueError(f"Unsupported norm: {norm!r}")


def sample_lp_ball(
    rng: np.random.Generator,
    *,
    n_samples: int,
    dim: int,
    radius: float,
    norm: str | int | float,
) -> np.ndarray:
    """Sample points approximately uniformly from an L1/L2/Linf ball."""
    if dim < 0:
        raise ValueError("dim must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if dim == 0 or radius == 0:
        return np.zeros((n_samples, dim), dtype=np.float32)

    _, p = _canonical_norm(norm)

    if p == float("inf"):
        noise = rng.uniform(-radius, radius, size=(n_samples, dim))
        return noise.astype(np.float32)

    if p == 2.0:
        directions = rng.normal(size=(n_samples, dim))
        norms = np.linalg.norm(directions, ord=2, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        scales = rng.random(n_samples) ** (1.0 / dim)
        noise = directions / norms * (radius * scales[:, None])
        return noise.astype(np.float32)

    if p == 1.0:
        weights = rng.exponential(scale=1.0, size=(n_samples, dim))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_samples, dim))
        scales = rng.random(n_samples) ** (1.0 / dim)
        noise = signs * weights * (radius * scales[:, None])
        return noise.astype(np.float32)

    raise ValueError(f"Unsupported norm code: {p!r}")


def load_tabular_benchmark_models(
    config_path: str | Path,
    *,
    datasets: list[str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load the tabular benchmark classifiers declared in a benchmark YAML config."""
    from scripts.benchmark import _build_torch_model_from_checkpoint

    resolved = _resolve_existing_path(config_path)
    if resolved is None:
        raise FileNotFoundError(f"Missing benchmark config: {config_path}")

    cfg = read_yaml(resolved)
    config_dir = resolved.parent
    dataset_filter = set(datasets) if datasets is not None else None
    models: dict[str, Any] = {}

    for ds_cfg in cfg.get("datasets", []):
        dataset_name = ds_cfg["name"]
        if dataset_filter is not None and dataset_name not in dataset_filter:
            continue

        model_cfg = ds_cfg.get("model", {})
        if model_cfg.get("name") != "tabular_classifier_ckpt":
            continue

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
        models[dataset_name] = _build_torch_model_from_checkpoint(
            checkpoint=str(checkpoint_path),
            device=device,
            dataset_module=params.get("dataset_module", dataset_name),
            hidden_dims=list(params.get("hidden_dims", [32, 8])),
            dropout=float(params.get("dropout", 0.2)),
        )

    if dataset_filter is not None:
        missing = sorted(dataset_filter.difference(models))
        if missing:
            raise KeyError(f"No tabular benchmark model config found for datasets: {missing}")

    return models


def evaluate_empirical_robustness_curves(
    df: pd.DataFrame,
    *,
    models_by_dataset: dict[str, Any],
    eps_by_norm: dict[str, list[float]],
    n_samples: int = 64,
    seed: int = 42,
    dataset_col: str = "dataset",
    method_col: str = "method_label",
    target_col: str = "target_class",
) -> pd.DataFrame:
    """Evaluate empirical robustness of counterfactuals under random Lp perturbations.

    Robustness is tested only on successful counterfactual rows. Perturbations are
    applied to numerical dimensions only; categorical one-hot blocks remain fixed.
    A counterfactual counts as robust for a given radius if all sampled
    perturbations preserve the target prediction.

    The input dataframe must contain the full dataset-shaped counterfactual
    feature schema (`x_cf_0` ... `x_cf_{n_features-1}`) for each dataset under
    analysis. This helper is intended for real benchmark outputs rather than
    partial toy frames.
    """
    data = df[df["success"]].copy()
    if data.empty:
        return pd.DataFrame()

    _, x_cf_cols_all = feature_columns(data)
    if not x_cf_cols_all:
        raise ValueError("Benchmark dataframe does not contain x_cf_* feature columns.")

    if target_col not in data.columns:
        source_col = "source_class" if "source_class" in data.columns else "y_orig" if "y_orig" in data.columns else None
        if source_col is None:
            raise ValueError(
                f"Benchmark dataframe is missing required column: {target_col!r}, "
                "and no fallback source_class/y_orig column is available."
            )
        source_values = pd.to_numeric(data[source_col], errors="coerce")
        unique_values = set(source_values.dropna().astype(int).unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Benchmark dataframe is missing required column: {target_col!r}, "
                f"and fallback from {source_col!r} is only supported for binary tasks."
            )
        data[target_col] = 1 - source_values.astype(np.int64)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for dataset_name, dataset_df in data.groupby(dataset_col, sort=False):
        if dataset_name not in models_by_dataset:
            continue

        model = models_by_dataset[dataset_name]
        spec = get_tabular_dataset_spec(dataset_name)
        x_cf_cols = [f"x_cf_{idx}" for idx in range(spec.n_features) if f"x_cf_{idx}" in dataset_df.columns]
        if len(x_cf_cols) != spec.n_features:
            raise ValueError(
                f"Dataset {dataset_name!r} is missing expected x_cf feature columns "
                f"(found {len(x_cf_cols)}, expected {spec.n_features})."
            )
        numeric_idx = numerical_feature_indices(dataset_name)
        if numeric_idx.size == 0:
            continue

        for method_label, method_df in dataset_df.groupby(method_col, sort=False):
            x_cf = method_df[x_cf_cols].to_numpy(dtype=np.float32)
            targets = method_df[target_col].to_numpy(dtype=np.int64)
            n_queries = len(method_df)

            for norm_name, eps_values in eps_by_norm.items():
                norm_label, _ = _canonical_norm(norm_name)
                group_rows: list[dict[str, object]] = []
                for epsilon in eps_values:
                    base = np.repeat(x_cf, n_samples, axis=0)
                    noise = sample_lp_ball(
                        rng,
                        n_samples=len(base),
                        dim=int(numeric_idx.size),
                        radius=float(epsilon),
                        norm=norm_label,
                    )
                    base[:, numeric_idx] += noise

                    preds = model.predict(base).reshape(n_queries, n_samples)
                    preserved = preds == targets[:, None]
                    robust_flags = preserved.all(axis=1)

                    group_rows.append(
                        {
                            dataset_col: dataset_name,
                            method_col: method_label,
                            "norm": norm_label,
                            "epsilon": float(epsilon),
                            "success_pct": float(100.0 * robust_flags.mean()),
                            "mean_preservation_pct": float(100.0 * preserved.mean()),
                            "n_queries": int(n_queries),
                            "n_samples": int(n_samples),
                        }
                    )
                if not group_rows:
                    continue

                eps_arr = np.asarray([row["epsilon"] for row in group_rows], dtype=float)
                success_arr = np.asarray([row["success_pct"] for row in group_rows], dtype=float) / 100.0
                if eps_arr.size > 1 and eps_arr[-1] > 0:
                    area = np.trapezoid(success_arr, eps_arr) if hasattr(np, "trapezoid") else np.trapz(success_arr, eps_arr)
                    auc_norm = float(area / eps_arr[-1])
                else:
                    auc_norm = float(success_arr[-1]) if success_arr.size else float("nan")

                for row in group_rows:
                    row["auc_norm"] = auc_norm
                    rows.append(row)

    return pd.DataFrame(rows)
