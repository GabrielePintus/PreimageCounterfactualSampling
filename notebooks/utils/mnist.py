"""MNIST-specific benchmark analysis helpers."""

from __future__ import annotations

import pandas as pd


def mnist_task_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return source-vs-target task coverage counts."""
    return (
        df[["query_idx", "source_class", "target_class"]]
        .drop_duplicates()
        .groupby(["source_class", "target_class"])
        .size()
        .unstack(fill_value=0)
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
