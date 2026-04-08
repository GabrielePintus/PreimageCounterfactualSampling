"""Reusable summaries and curve helpers for benchmark notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import success_only
from .io import feature_columns


def benchmark_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact one-row overview table for a benchmark dataframe."""
    return pd.DataFrame(
        {
            "n_rows": [len(df)],
            "n_methods": [df["method_label"].nunique()],
            "n_queries": [df["query_idx"].nunique() if "query_idx" in df.columns else np.nan],
            "n_datasets": [df["dataset"].nunique() if "dataset" in df.columns else 1],
            "n_success": [int(df["success"].sum())],
        }
    )


def method_order(
    df: pd.DataFrame,
    *,
    metric: str = "success",
    method_col: str = "method_label",
) -> list[str]:
    """Order methods by the mean value of a metric."""
    ascending = metric not in {"success", "validity", "validity_pct"}
    return (
        df.groupby(method_col)[metric]
        .mean()
        .sort_values(ascending=ascending)
        .index
        .tolist()
    )


def validity_summary(
    df: pd.DataFrame,
    *,
    method_col: str = "method_label",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Per-method validity summary."""
    summary = (
        df.groupby(method_col)
        .agg(
            n_total=("success", "count"),
            n_valid=("success", "sum"),
            validity_pct=("success", lambda s: 100.0 * s.mean()),
        )
    )
    if order is not None:
        summary = summary.reindex(order)
    return summary


def proximity_summary(
    df: pd.DataFrame,
    *,
    method_col: str = "method_label",
    success_only_rows: bool = True,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Per-method proximity/sparsity summary."""
    data = success_only(df) if success_only_rows else df.copy()
    summary = (
        data.groupby(method_col)
        .agg(
            mean_l2=("l2_distance", "mean"),
            median_l2=("l2_distance", "median"),
            mean_l1=("l1_distance", "mean"),
            median_l1=("l1_distance", "median"),
            mean_mad_l1=("mad_l1_distance", "mean"),
            mean_sparsity=("l0_sparsity", "mean"),
            mean_redundancy=("redundancy", "mean"),
            n_rows=("success", "size"),
        )
    )
    if order is not None:
        summary = summary.reindex(order)
    return summary


def runtime_summary(
    df: pd.DataFrame,
    *,
    method_col: str = "method_label",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Per-method runtime summary."""
    summary = (
        df.groupby(method_col)
        .agg(
            build_time_s=("build_time_s", "first"),
            mean_query_time_s=("runtime_s", "mean"),
            total_query_time_s=("runtime_s", "sum"),
            n_tasks=("runtime_s", "size"),
        )
    )
    if order is not None:
        summary = summary.reindex(order)
    return summary


def metric_table(
    df: pd.DataFrame,
    value_col: str,
    *,
    index: str = "method_label",
    columns: str = "dataset",
    agg: str = "mean",
    scale: float = 1.0,
    success_only_rows: bool = False,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Build a pivot table for a metric across methods and datasets."""
    data = success_only(df) if success_only_rows else df.copy()
    table = (
        data.groupby([index, columns])[value_col]
        .agg(agg)
        .mul(scale)
        .unstack(columns)
    )
    if order is not None:
        table = table.reindex(order)
    return table


def failure_category(message: object) -> str:
    """Map raw error messages to stable high-level failure categories."""
    if pd.isna(message):
        return "unknown"
    msg = str(message).lower()
    if "timeout" in msg:
        return "timeout"
    if "fit_error" in msg or "build_error" in msg:
        return "fit/build error"
    if "no_valid" in msg:
        return "no valid CF found"
    return "other"


def failure_summary(
    df: pd.DataFrame,
    *,
    method_col: str = "method_label",
    order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-method failure counts and stacked category breakdown."""
    failures = df[~df["success"]].copy()
    if failures.empty:
        empty_counts = pd.DataFrame(columns=["n_failures"])
        empty_breakdown = pd.DataFrame()
        return empty_counts, empty_breakdown

    failures["error_category"] = failures["error"].apply(failure_category)
    counts = failures.groupby(method_col).agg(n_failures=("error", "count"))
    breakdown = failures.pivot_table(
        index=method_col,
        columns="error_category",
        aggfunc="size",
        fill_value=0,
    )
    if order is not None:
        counts = counts.reindex([m for m in order if m in counts.index])
        breakdown = breakdown.reindex([m for m in order if m in breakdown.index])
    return counts, breakdown


def feature_change_summary(
    df: pd.DataFrame,
    *,
    method_col: str = "method_label",
    order: list[str] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Mean absolute feature change per method in raw feature space."""
    x_orig_cols, x_cf_cols = feature_columns(df)
    if not x_orig_cols or not x_cf_cols:
        return pd.DataFrame()

    data = success_only(df)
    raw_methods = [
        method
        for method in data[method_col].unique()
        if data[data[method_col] == method]["space"].eq("raw").any()
    ]
    if not raw_methods:
        return pd.DataFrame()

    feature_changes: dict[str, np.ndarray] = {}
    for method in raw_methods:
        sub = data[data[method_col] == method]
        if sub.empty:
            continue
        diff = np.abs(
            sub[x_cf_cols].to_numpy(dtype=float) - sub[x_orig_cols].to_numpy(dtype=float)
        )
        feature_changes[method] = np.nanmean(diff, axis=0)

    if not feature_changes:
        return pd.DataFrame()

    table = pd.DataFrame(
        feature_changes,
        index=[f"f{i}" for i in range(len(x_orig_cols))],
    ).T
    if order is not None:
        table = table.reindex([m for m in order if m in table.index])
    if normalize:
        table = table.div(table.max(axis=0).replace(0, 1), axis=1)
    return table


def validity_proximity_curve(
    distances: np.ndarray,
    eps_grid: np.ndarray | None = None,
    n_thresholds: int = 300,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return epsilon grid, success curve in percent, and normalized AUC."""
    values = np.asarray(distances, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("distances must be a non-empty 1D array")

    finite = np.sort(values[np.isfinite(values)])
    max_finite = float(finite.max()) if finite.size else 0.0
    if eps_grid is None:
        eps_grid = (
            np.linspace(0.0, max_finite, n_thresholds)
            if max_finite > 0
            else np.array([0.0])
        )
    else:
        eps_grid = np.asarray(eps_grid, dtype=float)

    success_curve = np.searchsorted(finite, eps_grid, side="right") / values.size
    if eps_grid.size > 1 and eps_grid[-1] > 0:
        trapezoid = getattr(np, "trapezoid", None)
        area = trapezoid(success_curve, eps_grid) if trapezoid is not None else np.trapz(success_curve, eps_grid)
        auc = float(area / eps_grid[-1])
    else:
        auc = float(success_curve[-1]) if success_curve.size else 0.0
    return eps_grid, 100.0 * success_curve, auc


def build_validity_curve_df(
    df: pd.DataFrame,
    *,
    metric: str,
    group_cols: tuple[str, ...] = ("method_label",),
    eps_group_cols: tuple[str, ...] = (),
    n_thresholds: int = 300,
) -> pd.DataFrame:
    """Build a long dataframe for validity-distance curve plotting."""
    rows: list[dict[str, object]] = []
    group_names = list(group_cols)
    eps_group_names = list(eps_group_cols)

    for group_key, group_df in df.groupby(group_names, sort=False):
        group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_names, group_key_tuple))

        if eps_group_names:
            ref_mask = np.ones(len(df), dtype=bool)
            for col in eps_group_names:
                ref_mask &= df[col].eq(group_values[col]).to_numpy()
            ref_df = df.loc[ref_mask]
        else:
            ref_df = df

        finite_ref = ref_df.loc[ref_df["success"], metric].to_numpy(dtype=float)
        finite_ref = finite_ref[np.isfinite(finite_ref)]
        eps_grid = (
            np.linspace(0.0, float(finite_ref.max()), n_thresholds)
            if finite_ref.size
            else np.array([0.0])
        )

        distances = group_df[metric].to_numpy(dtype=float)
        distances = np.where(group_df["success"].to_numpy(dtype=bool), distances, np.inf)
        eps, success_pct, auc = validity_proximity_curve(distances, eps_grid=eps_grid, n_thresholds=n_thresholds)

        for epsilon, success_value in zip(eps, success_pct):
            row = dict(group_values)
            row.update(
                {
                    "metric": metric,
                    "epsilon": float(epsilon),
                    "success_pct": float(success_value),
                    "auc_norm": float(auc),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def curve_auc_table(
    curve_df: pd.DataFrame,
    *,
    index: str = "method_label",
    columns: str = "dataset",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize normalized AUC from a long curve dataframe."""
    if curve_df.empty:
        return pd.DataFrame()
    table = (
        curve_df.groupby([index, columns])["auc_norm"]
        .first()
        .unstack(columns)
    )
    if order is not None:
        table = table.reindex(order)
    return table


def curve_endpoint_table(
    curve_df: pd.DataFrame,
    *,
    index: str = "method_label",
    columns: str = "dataset",
    value_col: str = "success_pct",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize the final curve value at the largest tested epsilon."""
    if curve_df.empty:
        return pd.DataFrame()
    endpoint = (
        curve_df.sort_values("epsilon")
        .groupby([index, columns], sort=False)
        .tail(1)
    )
    table = endpoint.pivot(index=index, columns=columns, values=value_col)
    if order is not None:
        table = table.reindex(order)
    return table


def validity_closeness_score(distances: np.ndarray, budget: float) -> float:
    """Return the average clipped closeness score under a distance budget."""
    values = np.asarray(distances, dtype=float)
    values = np.where(np.isfinite(values), values, np.inf)
    return float(np.mean(np.maximum(1.0 - values / budget, 0.0)))
