"""MNIST-specific benchmark analysis helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


MNIST_IMAGE_SHAPE = (1, 28, 28)
MNIST_N_PIXELS = 28 * 28
MNIST_TASK_COLUMNS = ("query_idx", "source_class", "target_class")


def mnist_task_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return source-vs-target task coverage counts."""
    return (
        df[list(MNIST_TASK_COLUMNS)]
        .drop_duplicates()
        .groupby(["source_class", "target_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(10), columns=range(10), fill_value=0)
    )


def mnist_validity_by_target_table(
    df: pd.DataFrame,
    *,
    method_col: str = "method_label",
) -> pd.DataFrame:
    """Return validity percentage by method and target digit."""
    return (
        df.groupby([method_col, "target_class"])["success"]
        .mean()
        .mul(100.0)
        .unstack("target_class")
    )


def validate_mnist_benchmark_df(
    df: pd.DataFrame,
    *,
    expected_tasks: int | None = None,
) -> dict[str, object]:
    """Validate the image schema and return a compact run-audit dictionary."""
    required = {
        *MNIST_TASK_COLUMNS,
        "success",
        "source_class",
        "target_class",
        "y_cf",
        "x_orig_0",
        f"x_orig_{MNIST_N_PIXELS - 1}",
        "x_cf_0",
        f"x_cf_{MNIST_N_PIXELS - 1}",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"MNIST benchmark is missing required columns: {missing}")

    task_rows = df[list(MNIST_TASK_COLUMNS)].drop_duplicates()
    if len(task_rows) != len(df):
        raise ValueError(
            "MNIST benchmark contains duplicate source-target tasks; expected "
            "one row per (query_idx, source_class, target_class)."
        )
    if expected_tasks is not None and len(df) != int(expected_tasks):
        raise ValueError(
            f"MNIST benchmark contains {len(df)} tasks; expected {expected_tasks}."
        )
    if bool((df["source_class"].astype(int) == df["target_class"].astype(int)).any()):
        raise ValueError("MNIST benchmark contains tasks whose source equals target.")

    y_true_mismatch = (
        int((df["y_true"].astype(int) != df["source_class"].astype(int)).sum())
        if "y_true" in df.columns
        else None
    )
    target_mismatch = int(
        (df["y_cf"].astype(int) != df["target_class"].astype(int)).sum()
    )
    return {
        "n_tasks": int(len(df)),
        "n_source_images": int(df["query_idx"].nunique()),
        "n_source_classes": int(df["source_class"].nunique()),
        "n_target_classes": int(df["target_class"].nunique()),
        "n_success": int(df["success"].fillna(False).astype(bool).sum()),
        "validity_pct": float(100.0 * df["success"].fillna(False).mean()),
        "y_true_source_mismatches": y_true_mismatch,
        "target_prediction_mismatches": target_mismatch,
    }


def mnist_pair_metric_table(
    df: pd.DataFrame,
    value_col: str,
    *,
    agg: str | Callable = "median",
    scale: float = 1.0,
) -> pd.DataFrame:
    """Aggregate a metric into a complete 10x10 source-target table."""
    table = (
        df.groupby(["source_class", "target_class"])[value_col]
        .agg(agg)
        .mul(scale)
        .unstack("target_class")
    )
    return table.reindex(index=range(10), columns=range(10))


def add_mnist_pixel_metrics(
    df: pd.DataFrame,
    *,
    thresholds: Iterable[float] = (1.0 / 255.0, 0.01, 0.05),
    clip: bool = True,
) -> pd.DataFrame:
    """Attach image-domain distances and multi-threshold L0 metrics."""
    x_orig_cols = [f"x_orig_{idx}" for idx in range(MNIST_N_PIXELS)]
    x_cf_cols = [f"x_cf_{idx}" for idx in range(MNIST_N_PIXELS)]
    missing = [col for col in (*x_orig_cols, *x_cf_cols) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing MNIST pixel column: {missing[0]}")

    out = df.copy()
    x_orig = out[x_orig_cols].to_numpy(dtype=np.float32)
    x_cf_raw = out[x_cf_cols].to_numpy(dtype=np.float32)
    x_cf = np.clip(x_cf_raw, 0.0, 1.0) if clip else x_cf_raw
    diff = np.abs(x_cf - x_orig)

    out["image_l1"] = diff.sum(axis=1)
    out["image_l1_per_pixel"] = diff.mean(axis=1)
    out["image_l2"] = np.linalg.norm(diff, ord=2, axis=1)
    out["image_linf"] = diff.max(axis=1)
    for threshold in thresholds:
        suffix = _threshold_suffix(float(threshold))
        changed = diff > float(threshold)
        out[f"image_l0_count_{suffix}"] = changed.sum(axis=1)
        out[f"image_l0_fraction_{suffix}"] = changed.mean(axis=1)
    out["cf_raw_min"] = x_cf_raw.min(axis=1)
    out["cf_raw_max"] = x_cf_raw.max(axis=1)
    out["cf_out_of_bounds_pixels"] = (
        (x_cf_raw < -1.0e-6) | (x_cf_raw > 1.0 + 1.0e-6)
    ).sum(axis=1)
    return out


def _threshold_suffix(threshold: float) -> str:
    if np.isclose(threshold, 1.0 / 255.0):
        return "1_over_255"
    return f"{threshold:g}".replace(".", "p")


def clustered_bootstrap_ci(
    values: np.ndarray | pd.Series,
    clusters: np.ndarray | pd.Series,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 123,
) -> dict[str, float | int]:
    """Return a percentile CI while resampling complete source-image clusters."""
    values_arr = np.asarray(values, dtype=float).reshape(-1)
    clusters_arr = np.asarray(clusters).reshape(-1)
    if len(values_arr) != len(clusters_arr):
        raise ValueError("values and clusters must have the same length")
    finite = np.isfinite(values_arr) & pd.notna(clusters_arr)
    values_arr = values_arr[finite]
    clusters_arr = clusters_arr[finite]
    if not len(values_arr):
        return {
            "n": 0,
            "n_clusters": 0,
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    unique_clusters = pd.unique(clusters_arr)
    grouped_values = [
        values_arr[clusters_arr == cluster] for cluster in unique_clusters
    ]
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(n_resamples), dtype=float)
    for idx in range(int(n_resamples)):
        sampled = rng.integers(0, len(grouped_values), size=len(grouped_values))
        bootstrap_values = np.concatenate([grouped_values[pos] for pos in sampled])
        estimates[idx] = float(statistic(bootstrap_values))

    alpha = (1.0 - float(confidence_level)) / 2.0
    ci_low, ci_high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {
        "n": int(len(values_arr)),
        "n_clusters": int(len(unique_clusters)),
        "estimate": float(statistic(values_arr)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def clustered_metric_summary(
    df: pd.DataFrame,
    metric_columns: Iterable[str],
    *,
    cluster_col: str = "query_idx",
    n_resamples: int = 10_000,
    seed: int = 123,
) -> pd.DataFrame:
    """Summarize metrics with mean CI, median, and interquartile range."""
    rows: list[dict[str, object]] = []
    for offset, column in enumerate(metric_columns):
        stats = clustered_bootstrap_ci(
            df[column],
            df[cluster_col],
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        finite = (
            pd.to_numeric(df[column], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        rows.append(
            {
                "metric": column,
                **stats,
                "median": float(np.median(finite)) if len(finite) else np.nan,
                "q25": float(np.quantile(finite, 0.25)) if len(finite) else np.nan,
                "q75": float(np.quantile(finite, 0.75)) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _unwrap_lenet5(model: object) -> torch.nn.Module:
    module = getattr(model, "model", model)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("Expected a torch module or TorchModelWrapper.")
    if not hasattr(module, "features") and hasattr(module, "model"):
        module = module.model
    if not hasattr(module, "features") or not hasattr(module, "classifier"):
        raise TypeError("Expected a LeNet5Classifier-compatible module.")
    return module


@torch.no_grad()
def extract_lenet5_embeddings(
    model: object,
    x: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    """Extract the 84-D activation immediately before LeNet-5's last layer."""
    module = _unwrap_lenet5(model).eval()
    device = next(module.parameters()).device
    x_np = np.asarray(x, dtype=np.float32).reshape(-1, *MNIST_IMAGE_SHAPE)
    parts: list[np.ndarray] = []
    for start in range(0, len(x_np), int(batch_size)):
        batch = torch.from_numpy(x_np[start : start + int(batch_size)]).to(device)
        features = module.features(batch).reshape(len(batch), -1)
        embedding = module.classifier[:-1](features)
        parts.append(embedding.detach().cpu().numpy().astype(np.float32, copy=False))
    return (
        np.concatenate(parts, axis=0)
        if parts
        else np.empty((0, 84), dtype=np.float32)
    )


@torch.no_grad()
def predict_lenet5_logits(
    model: object,
    x: np.ndarray,
    *,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return LeNet-5 logits for flattened or image-shaped MNIST arrays."""
    module = _unwrap_lenet5(model).eval()
    device = next(module.parameters()).device
    x_np = np.asarray(x, dtype=np.float32).reshape(-1, *MNIST_IMAGE_SHAPE)
    parts: list[np.ndarray] = []
    for start in range(0, len(x_np), int(batch_size)):
        batch = torch.from_numpy(x_np[start : start + int(batch_size)]).to(device)
        parts.append(module(batch).detach().cpu().numpy())
    return (
        np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if parts
        else np.empty((0, 10), dtype=np.float32)
    )


def attach_target_confidence(
    df: pd.DataFrame,
    logits: np.ndarray,
    *,
    target_col: str = "target_class",
) -> pd.DataFrame:
    """Attach target softmax confidence and target-vs-best-other logit margin."""
    out = df.copy()
    logits_np = np.asarray(logits, dtype=np.float64)
    if logits_np.shape[0] != len(out):
        raise ValueError("logits row count must match dataframe row count")
    targets = out[target_col].to_numpy(dtype=np.int64)
    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    row_idx = np.arange(len(out))
    target_logits = logits_np[row_idx, targets]
    other_logits = logits_np.copy()
    other_logits[row_idx, targets] = -np.inf
    out["target_confidence"] = probabilities[row_idx, targets]
    out["target_logit_margin"] = target_logits - other_logits.max(axis=1)
    out["analysis_prediction"] = logits_np.argmax(axis=1)
    return out


def evaluate_mnist_robustness(
    model: object,
    x_cf: np.ndarray,
    targets: np.ndarray,
    query_idx: np.ndarray,
    *,
    perturbation: str,
    levels: Iterable[float],
    n_samples: int = 64,
    seed: int = 42,
    query_batch_size: int = 32,
    prediction_batch_size: int = 1024,
) -> pd.DataFrame:
    """Evaluate per-query Gaussian or uniform-Linf target preservation."""
    perturbation = str(perturbation).strip().lower()
    if perturbation not in {"gaussian", "linf"}:
        raise ValueError("perturbation must be 'gaussian' or 'linf'")
    x_np = np.asarray(x_cf, dtype=np.float32).reshape(-1, MNIST_N_PIXELS)
    targets_np = np.asarray(targets, dtype=np.int64).reshape(-1)
    query_np = np.asarray(query_idx).reshape(-1)
    if not (len(x_np) == len(targets_np) == len(query_np)):
        raise ValueError("x_cf, targets, and query_idx must have matching lengths")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rows: list[dict[str, object]] = []
    for level_idx, level in enumerate(levels):
        radius = float(level)
        rng = np.random.default_rng(seed + 10_000 * level_idx)
        for start in range(0, len(x_np), int(query_batch_size)):
            stop = min(start + int(query_batch_size), len(x_np))
            base = np.repeat(x_np[start:stop, None, :], int(n_samples), axis=1)
            if radius > 0:
                if perturbation == "gaussian":
                    noise = rng.normal(0.0, radius, size=base.shape)
                else:
                    noise = rng.uniform(-radius, radius, size=base.shape)
                base = base + noise.astype(np.float32)
            base = np.clip(base, 0.0, 1.0).astype(np.float32, copy=False)
            logits = predict_lenet5_logits(
                model,
                base.reshape(-1, MNIST_N_PIXELS),
                batch_size=prediction_batch_size,
            )
            pred = logits.argmax(axis=1).reshape(stop - start, int(n_samples))
            preserved = pred == targets_np[start:stop, None]
            for local_idx in range(stop - start):
                rows.append(
                    {
                        "query_idx": query_np[start + local_idx],
                        "target_class": int(targets_np[start + local_idx]),
                        "perturbation": perturbation,
                        "level": radius,
                        "n_samples": int(n_samples),
                        "preservation_rate": float(preserved[local_idx].mean()),
                        "all_preserved": bool(preserved[local_idx].all()),
                    }
                )
    return pd.DataFrame(rows)


def robustness_curve_summary(per_query_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-query robustness and attach a normalized curve AUC."""
    summary = (
        per_query_df.groupby(["perturbation", "level"], as_index=False)
        .agg(
            mean_preservation_pct=("preservation_rate", lambda s: 100.0 * s.mean()),
            all_preserved_pct=("all_preserved", lambda s: 100.0 * s.mean()),
            n_queries=("query_idx", "size"),
            n_source_images=("query_idx", "nunique"),
            n_samples=("n_samples", "first"),
        )
        .sort_values(["perturbation", "level"], ignore_index=True)
    )
    auc_by_kind: dict[str, float] = {}
    for perturbation, group in summary.groupby("perturbation"):
        x = group["level"].to_numpy(dtype=float)
        y = group["mean_preservation_pct"].to_numpy(dtype=float) / 100.0
        if len(x) > 1 and x.max() > x.min():
            area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
            auc_by_kind[str(perturbation)] = float(area / (x.max() - x.min()))
        else:
            auc_by_kind[str(perturbation)] = float(y[-1]) if len(y) else np.nan
    summary["auc_norm"] = summary["perturbation"].map(auc_by_kind)
    return summary


def same_class_neighbor_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cf: np.ndarray,
    target_labels: np.ndarray,
    *,
    metric: str = "manhattan",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute train references and per-CF same-class 1NN/2NN diagnostics."""
    train_np = np.asarray(x_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.int64).reshape(-1)
    cf_np = np.asarray(x_cf, dtype=np.float32)
    target_np = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if len(train_np) != len(y_train_np) or len(cf_np) != len(target_np):
        raise ValueError("Feature arrays and labels must have matching lengths")

    reference_frames: list[pd.DataFrame] = []
    cf_rows = pd.DataFrame(
        {
            "target_class": target_np,
            "same_class_d1": np.nan,
            "same_class_d2": np.nan,
            "same_class_percentile": np.nan,
            "neighbor_contrast": np.nan,
        }
    )
    for label in np.unique(y_train_np):
        train_mask = y_train_np == int(label)
        class_train = train_np[train_mask]
        if len(class_train) <= 1:
            continue
        train_nn = NearestNeighbors(
            n_neighbors=2, metric=metric, n_jobs=-1
        ).fit(class_train)
        train_distances = train_nn.kneighbors(class_train, return_distance=True)[0][
            :, 1
        ]
        reference = np.sort(train_distances.astype(float))
        reference_frames.append(
            pd.DataFrame(
                {
                    "class_label": int(label),
                    "same_class_train_d1": train_distances,
                }
            )
        )

        cf_indices = np.flatnonzero(target_np == int(label))
        if not len(cf_indices):
            continue
        n_neighbors = min(2, len(class_train))
        cf_nn = NearestNeighbors(
            n_neighbors=n_neighbors, metric=metric, n_jobs=-1
        ).fit(class_train)
        distances = cf_nn.kneighbors(cf_np[cf_indices], return_distance=True)[0]
        d1 = distances[:, 0].astype(float)
        d2 = distances[:, 1].astype(float) if n_neighbors > 1 else np.full(len(d1), np.nan)
        cf_rows.loc[cf_indices, "same_class_d1"] = d1
        cf_rows.loc[cf_indices, "same_class_d2"] = d2
        cf_rows.loc[cf_indices, "same_class_percentile"] = (
            np.searchsorted(reference, d1, side="right") / len(reference)
        )
        cf_rows.loc[cf_indices, "neighbor_contrast"] = np.divide(
            d1,
            d2,
            out=np.full_like(d1, np.nan),
            where=d2 > 0,
        )

    reference_df = (
        pd.concat(reference_frames, ignore_index=True)
        if reference_frames
        else pd.DataFrame(columns=["class_label", "same_class_train_d1"])
    )
    return reference_df, cf_rows


def relative_proximity_summary(
    reference_df: pd.DataFrame,
    cf_metrics_df: pd.DataFrame,
) -> dict[str, float]:
    """Return pooled paper-compatible and class-balanced relative proximity."""
    train_mean = float(reference_df["same_class_train_d1"].mean())
    cf_mean = float(cf_metrics_df["same_class_d1"].mean())
    ratios: list[float] = []
    for label, cf_group in cf_metrics_df.groupby("target_class"):
        train_values = reference_df.loc[
            reference_df["class_label"].astype(int) == int(label),
            "same_class_train_d1",
        ]
        if len(train_values) and float(train_values.mean()) > 0:
            ratios.append(float(cf_group["same_class_d1"].mean() / train_values.mean()))
    return {
        "train_same_class_d1_mean": train_mean,
        "cf_same_class_d1_mean": cf_mean,
        "rpr_pooled": float(cf_mean / train_mean) if train_mean > 0 else np.nan,
        "rpr_class_macro": float(np.mean(ratios)) if ratios else np.nan,
    }


def same_class_distribution_summary(
    reference_df: pd.DataFrame,
    cf_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return classwise Wasserstein distances on log1p same-class distances."""
    rows: list[dict[str, float | int]] = []
    for label, cf_group in cf_metrics_df.groupby("target_class"):
        train_values = reference_df.loc[
            reference_df["class_label"].astype(int) == int(label),
            "same_class_train_d1",
        ].to_numpy(dtype=float)
        cf_values = cf_group["same_class_d1"].dropna().to_numpy(dtype=float)
        if not len(train_values) or not len(cf_values):
            continue
        rows.append(
            {
                "target_class": int(label),
                "wasserstein_log1p": float(
                    wasserstein_distance(np.log1p(train_values), np.log1p(cf_values))
                ),
                "n_train": int(len(train_values)),
                "n_cf": int(len(cf_values)),
            }
        )
    return pd.DataFrame(rows)


def neighborhood_label_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_cf: np.ndarray,
    target_labels: np.ndarray,
    *,
    k: int = 20,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Return normalized kNN label entropy and target-class purity per CF."""
    train_np = np.asarray(x_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.int64).reshape(-1)
    cf_np = np.asarray(x_cf, dtype=np.float32)
    target_np = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    k_eff = min(int(k), len(train_np))
    neighbors = NearestNeighbors(
        n_neighbors=k_eff, metric=metric, n_jobs=-1
    ).fit(train_np)
    indices = neighbors.kneighbors(cf_np, return_distance=False)
    labels = y_train_np[indices]
    n_classes = max(1, len(np.unique(y_train_np)))
    entropies = []
    for row in labels:
        counts = np.bincount(row, minlength=n_classes).astype(float)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        entropies.append(entropy / np.log(n_classes) if n_classes > 1 else 0.0)
    purity = (labels == target_np[:, None]).mean(axis=1)
    return pd.DataFrame(
        {
            "knn_k": k_eff,
            "knn_normalized_entropy": entropies,
            "knn_target_purity": purity,
        }
    )


def embedding_lof_scores(
    train_embeddings: np.ndarray,
    cf_embeddings: np.ndarray,
    *,
    n_neighbors: int = 20,
) -> np.ndarray:
    """Return positive LOF novelty scores for counterfactual embeddings."""
    train_np = np.asarray(train_embeddings, dtype=np.float32)
    cf_np = np.asarray(cf_embeddings, dtype=np.float32)
    k_eff = min(int(n_neighbors), max(1, len(train_np) - 1))
    lof = LocalOutlierFactor(n_neighbors=k_eff, novelty=True, n_jobs=-1)
    lof.fit(train_np)
    return np.maximum(-lof.score_samples(cf_np), 1.0e-12)


def cache_fingerprint(
    paths: Iterable[str | Path],
    *,
    settings: dict[str, object] | None = None,
) -> str:
    """Hash source files and settings used by a cached notebook computation."""
    digest = hashlib.sha256()
    for path_like in paths:
        path = Path(path_like).resolve()
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(
        json.dumps(settings or {}, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


def load_or_compute_dataframe(
    cache_path: str | Path,
    fingerprint: str,
    compute: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, bool]:
    """Load a valid parquet cache or compute and persist a replacement."""
    path = Path(cache_path)
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return pd.read_parquet(path), True

    frame = compute()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Cached computation must return a pandas DataFrame")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    metadata_path.write_text(
        json.dumps({"fingerprint": fingerprint}, indent=2) + "\n",
        encoding="utf-8",
    )
    return frame, False


__all__ = [
    "MNIST_IMAGE_SHAPE",
    "MNIST_N_PIXELS",
    "MNIST_TASK_COLUMNS",
    "add_mnist_pixel_metrics",
    "attach_target_confidence",
    "cache_fingerprint",
    "clustered_bootstrap_ci",
    "clustered_metric_summary",
    "embedding_lof_scores",
    "evaluate_mnist_robustness",
    "extract_lenet5_embeddings",
    "load_or_compute_dataframe",
    "mnist_pair_metric_table",
    "mnist_task_coverage_table",
    "mnist_validity_by_target_table",
    "neighborhood_label_metrics",
    "predict_lenet5_logits",
    "relative_proximity_summary",
    "robustness_curve_summary",
    "same_class_distribution_summary",
    "same_class_neighbor_metrics",
    "validate_mnist_benchmark_df",
]
