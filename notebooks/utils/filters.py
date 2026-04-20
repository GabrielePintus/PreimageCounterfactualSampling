"""Common dataframe filters for benchmark notebooks."""

from __future__ import annotations

import pandas as pd


def select_methods(
    df: pd.DataFrame,
    methods: list[str] | None,
    *,
    method_col: str = "method_label",
) -> pd.DataFrame:
    """Keep only the selected methods, preserving dataframe shape otherwise."""
    if not methods:
        return df.copy()
    return df[df[method_col].isin(methods)].copy()


def success_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only successful counterfactual rows."""
    return df[df["success"]].copy()


def _shared_success_status(
    df: pd.DataFrame,
    *,
    by: tuple[str, ...],
    method_col: str,
) -> pd.DataFrame:
    """Return one row per task with a shared-success boolean."""
    status = (
        df.groupby([*by, method_col])["success"]
        .any()
        .unstack(method_col)
        .fillna(False)
        .all(axis=1)
        .rename("shared_success")
    )
    if isinstance(status.index, pd.MultiIndex):
        return status.reset_index()
    return status.rename_axis(by[0]).reset_index()


def shared_success_subset(
    df: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("query_idx",),
    method_col: str = "method_label",
) -> pd.DataFrame:
    """Keep rows for tasks where all methods succeeded, then require success=True."""
    status_df = _shared_success_status(df, by=by, method_col=method_col)
    shared_df = status_df[status_df["shared_success"]].drop(columns="shared_success")
    if shared_df.empty:
        return df.iloc[0:0].copy()

    merged = df.merge(shared_df.assign(_shared=True), on=list(by), how="inner")
    return merged[merged["success"]].drop(columns="_shared").copy()


def shared_success_retention_summary(
    df: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("query_idx",),
    group_cols: tuple[str, ...] = (),
    method_col: str = "method_label",
) -> pd.DataFrame:
    """Summarize how many tasks remain after shared-success filtering."""
    status_df = _shared_success_status(df, by=by, method_col=method_col)
    if group_cols:
        summary = (
            status_df.groupby(list(group_cols), dropna=False)
            .agg(
                n_total_tasks=("shared_success", "size"),
                n_shared_tasks=("shared_success", "sum"),
            )
        )
    else:
        summary = pd.DataFrame(
            {
                "n_total_tasks": [int(status_df["shared_success"].size)],
                "n_shared_tasks": [int(status_df["shared_success"].sum())],
            }
        )
    summary["retention_pct"] = 100.0 * summary["n_shared_tasks"] / summary["n_total_tasks"]
    return summary


def annotate_common_success_subset(
    df: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("query_idx",),
    method_col: str = "method_label",
    label_col: str = "subset_group",
    common_label: str = "Common-success subset",
    outside_label: str = "Outside common subset",
) -> pd.DataFrame:
    """Annotate successful rows as inside the common-success subset and all others as outside.

    A row belongs to the common-success subset only if:
    - its task key is shared-success across all compared methods, and
    - the row itself is successful.
    """
    status_df = _shared_success_status(df, by=by, method_col=method_col)
    if status_df.empty:
        out = df.copy()
        out[label_col] = outside_label
        return out

    merged = df.merge(status_df, on=list(by), how="left")
    out = merged.copy()
    row_in_common_subset = out["success"].fillna(False) & out["shared_success"].fillna(False)
    out[label_col] = row_in_common_subset.map(
        {True: common_label, False: outside_label}
    )
    return out.drop(columns="shared_success")
