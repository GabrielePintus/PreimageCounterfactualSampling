"""Constraint-quality diagnostics for benchmark analysis notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from counterfactuals.preprocessing import snap_ohe_blocks
from dataset_specs import get_tabular_dataset_spec

DEFAULT_OHE_VALID_TOL = 1e-4


def _x_orig_columns_for_dataset(dataset_df: pd.DataFrame, dataset_name: str) -> list[str]:
    """Return the encoded factual feature columns expected for one dataset."""
    spec = get_tabular_dataset_spec(dataset_name)
    cols = [f"x_orig_{idx}" for idx in range(spec.n_features)]
    missing = [col for col in cols if col not in dataset_df.columns]
    if missing:
        raise ValueError(
            f"Dataset {dataset_name!r} is missing expected x_orig feature columns, "
            f"starting with {missing[0]!r}."
        )
    return cols


def _x_cf_columns_for_dataset(dataset_df: pd.DataFrame, dataset_name: str) -> list[str]:
    """Return the encoded counterfactual feature columns expected for one dataset."""
    spec = get_tabular_dataset_spec(dataset_name)
    cols = [f"x_cf_{idx}" for idx in range(spec.n_features)]
    missing = [col for col in cols if col not in dataset_df.columns]
    if missing:
        raise ValueError(
            f"Dataset {dataset_name!r} is missing expected x_cf feature columns, "
            f"starting with {missing[0]!r}."
        )
    return cols


def _ohe_block_valid_mask(
    x_cf: np.ndarray,
    *,
    dataset_name: str,
    tol: float = DEFAULT_OHE_VALID_TOL,
) -> np.ndarray:
    """Return per-row, per-block validity for strict one-hot categorical blocks."""
    spec = get_tabular_dataset_spec(dataset_name)
    if not spec.ohe_blocks:
        return np.ones((len(x_cf), 0), dtype=bool)

    valid_blocks: list[np.ndarray] = []
    for block in spec.ohe_blocks:
        block_vals = x_cf[:, block.start:block.end]
        in_range = np.all((block_vals >= -tol) & (block_vals <= 1.0 + tol), axis=1)
        sum_ok = np.abs(block_vals.sum(axis=1) - 1.0) <= tol
        high_mask = block_vals >= (1.0 - tol)
        one_high = high_mask.sum(axis=1) == 1
        others_low = np.all(np.where(high_mask, True, block_vals <= tol), axis=1)
        valid_blocks.append(in_range & sum_ok & one_high & others_low)

    return np.column_stack(valid_blocks)


def _immutable_feature_change_mask(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    *,
    dataset_name: str,
    tol: float = DEFAULT_OHE_VALID_TOL,
) -> np.ndarray:
    """Return per-row, per-feature change flags for immutable features."""
    spec = get_tabular_dataset_spec(dataset_name)
    if not spec.immutable_features:
        return np.ones((len(x_cf), 0), dtype=bool)

    feature_to_index = {feature: idx for idx, feature in enumerate(spec.feature_names)}
    changed_features: list[np.ndarray] = []
    for feature in spec.immutable_features:
        idx = feature_to_index[feature]
        start, end = spec.feature_slices[idx]
        delta = np.abs(x_cf[:, start:end] - x_orig[:, start:end])
        changed_features.append(np.any(delta > tol, axis=1))

    return np.column_stack(changed_features)


def constraint_quality_rows(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    method_col: str = "method_label",
    tol: float = DEFAULT_OHE_VALID_TOL,
) -> pd.DataFrame:
    """Return per-counterfactual OHE constraint diagnostics for categorical datasets."""
    rows: list[pd.DataFrame] = []

    for dataset_name, dataset_df in df.groupby(dataset_col, sort=False):
        spec = get_tabular_dataset_spec(dataset_name)
        if not spec.ohe_blocks:
            continue

        x_orig_cols = _x_orig_columns_for_dataset(dataset_df, dataset_name)
        x_cf_cols = _x_cf_columns_for_dataset(dataset_df, dataset_name)
        x_orig = dataset_df[x_orig_cols].to_numpy(dtype=np.float32)
        x_cf = dataset_df[x_cf_cols].to_numpy(dtype=np.float32)
        block_valid = _ohe_block_valid_mask(x_cf, dataset_name=dataset_name, tol=tol)
        immutable_changed = _immutable_feature_change_mask(
            x_orig,
            x_cf,
            dataset_name=dataset_name,
            tol=tol,
        )
        snapped = snap_ohe_blocks(x_cf, spec.ohe_blocks)

        categorical_idx: list[int] = []
        for block in spec.ohe_blocks:
            categorical_idx.extend(range(block.start, block.end))

        diag_df = dataset_df[[dataset_col, method_col]].copy()
        diag_df["ohe_valid"] = block_valid.all(axis=1)
        diag_df["invalid_block_fraction"] = 1.0 - block_valid.mean(axis=1)
        diag_df["ohe_snap_l1"] = np.abs(x_cf[:, categorical_idx] - snapped[:, categorical_idx]).sum(axis=1)
        if immutable_changed.shape[1] > 0:
            immutable_idx: list[int] = []
            feature_to_index = {feature: idx for idx, feature in enumerate(spec.feature_names)}
            for feature in spec.immutable_features:
                start, end = spec.feature_slices[feature_to_index[feature]]
                immutable_idx.extend(range(start, end))
            diag_df["immutable_valid"] = ~immutable_changed.any(axis=1)
            diag_df["immutable_change_fraction"] = immutable_changed.mean(axis=1)
            diag_df["immutable_l1"] = np.abs(x_cf[:, immutable_idx] - x_orig[:, immutable_idx]).sum(axis=1)
        else:
            diag_df["immutable_valid"] = np.nan
            diag_df["immutable_change_fraction"] = np.nan
            diag_df["immutable_l1"] = np.nan
        rows.append(diag_df)

    if not rows:
        return pd.DataFrame(
            columns=[
                dataset_col,
                method_col,
                "ohe_valid",
                "invalid_block_fraction",
                "ohe_snap_l1",
                "immutable_valid",
                "immutable_change_fraction",
                "immutable_l1",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def constraint_quality_summary(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    method_col: str = "method_label",
    order: list[str] | None = None,
    tol: float = DEFAULT_OHE_VALID_TOL,
) -> pd.DataFrame:
    """Aggregate OHE constraint diagnostics across categorical datasets by method."""
    rows_df = constraint_quality_rows(df, dataset_col=dataset_col, method_col=method_col, tol=tol)
    if rows_df.empty:
        return pd.DataFrame(
            columns=["n_rows", "ohe_valid_pct", "mean_invalid_block_pct", "mean_ohe_snap_l1"]
        )

    summary = rows_df.groupby(method_col).agg(
        n_rows=("ohe_valid", "size"),
        ohe_valid_pct=("ohe_valid", lambda s: 100.0 * float(np.mean(s))),
        mean_invalid_block_pct=("invalid_block_fraction", lambda s: 100.0 * float(np.mean(s))),
        mean_ohe_snap_l1=("ohe_snap_l1", "mean"),
        immutable_valid_pct=("immutable_valid", lambda s: 100.0 * float(np.mean(s.dropna())) if s.notna().any() else np.nan),
        mean_immutable_change_pct=("immutable_change_fraction", lambda s: 100.0 * float(np.mean(s.dropna())) if s.notna().any() else np.nan),
        mean_immutable_l1=("immutable_l1", lambda s: float(np.mean(s.dropna())) if s.notna().any() else np.nan),
    )
    if order is not None:
        summary = summary.reindex([method for method in order if method in summary.index])
    return summary


def constraint_quality_tables(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    method_col: str = "method_label",
    order: list[str] | None = None,
    tol: float = DEFAULT_OHE_VALID_TOL,
) -> dict[str, pd.DataFrame]:
    """Return per-dataset OHE diagnostic tables for notebook display."""
    rows_df = constraint_quality_rows(df, dataset_col=dataset_col, method_col=method_col, tol=tol)
    if rows_df.empty:
        return {
            "ohe_valid_pct": pd.DataFrame(),
            "mean_invalid_block_pct": pd.DataFrame(),
            "mean_ohe_snap_l1": pd.DataFrame(),
            "immutable_valid_pct": pd.DataFrame(),
            "mean_immutable_change_pct": pd.DataFrame(),
            "mean_immutable_l1": pd.DataFrame(),
        }

    grouped = rows_df.groupby([method_col, dataset_col]).agg(
        ohe_valid_pct=("ohe_valid", lambda s: 100.0 * float(np.mean(s))),
        mean_invalid_block_pct=("invalid_block_fraction", lambda s: 100.0 * float(np.mean(s))),
        mean_ohe_snap_l1=("ohe_snap_l1", "mean"),
        immutable_valid_pct=("immutable_valid", lambda s: 100.0 * float(np.mean(s.dropna())) if s.notna().any() else np.nan),
        mean_immutable_change_pct=("immutable_change_fraction", lambda s: 100.0 * float(np.mean(s.dropna())) if s.notna().any() else np.nan),
        mean_immutable_l1=("immutable_l1", lambda s: float(np.mean(s.dropna())) if s.notna().any() else np.nan),
    )

    tables: dict[str, pd.DataFrame] = {}
    for metric in (
        "ohe_valid_pct",
        "mean_invalid_block_pct",
        "mean_ohe_snap_l1",
        "immutable_valid_pct",
        "mean_immutable_change_pct",
        "mean_immutable_l1",
    ):
        table = grouped[metric].unstack(dataset_col)
        table = table.dropna(axis=1, how="all")
        if order is not None:
            table = table.reindex([method for method in order if method in table.index])
        tables[metric] = table
    return tables
