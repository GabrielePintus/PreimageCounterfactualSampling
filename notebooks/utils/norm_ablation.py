"""Paired analysis helpers for the CertCF L1-vs-L2 query-norm ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .io import normalize_benchmark_df


NORM_ABLATION_RUNS = (
    "certcf_query_l1_no_sparsity",
    "certcf_query_l2_no_sparsity",
)
NORM_ABLATION_LABELS = {
    "certcf_query_l1_no_sparsity": "L1 query objective",
    "certcf_query_l2_no_sparsity": "L2 query objective",
}
NORM_ABLATION_DATASETS = (
    "adult",
    "compas",
    "german_credit",
    "give_me_some_credit",
    "heloc",
    "lending_club",
    "wisconsin_breast_cancer",
)
NORM_ABLATION_EXPECTED_TASKS = {
    "adult": 1000,
    "compas": 617,
    "german_credit": 100,
    "give_me_some_credit": 1000,
    "heloc": 1000,
    "lending_club": 1000,
    "wisconsin_breast_cancer": 56,
}
NORM_ABLATION_TASK_KEY = ("dataset", "query_idx", "target_class")


def _expected_tasks_for_dataset(
    dataset_name: str,
    expected_tasks: int | Mapping[str, int] | None,
) -> int:
    if expected_tasks is None:
        if dataset_name not in NORM_ABLATION_EXPECTED_TASKS:
            raise KeyError(f"No expected task count configured for {dataset_name!r}.")
        return int(NORM_ABLATION_EXPECTED_TASKS[dataset_name])
    if isinstance(expected_tasks, Mapping):
        if dataset_name not in expected_tasks:
            raise KeyError(f"No expected task count supplied for {dataset_name!r}.")
        return int(expected_tasks[dataset_name])
    return int(expected_tasks)


def load_norm_ablation_results(
    result_path: str | Path,
    *,
    datasets: Iterable[str] = NORM_ABLATION_DATASETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the combined result and any readable per-dataset progress files.

    Per-dataset files are included deliberately: the benchmark writes them as it
    progresses, before the combined parquet is complete. Duplicate rows are
    resolved in favor of the later per-dataset frame.
    """
    combined_path = Path(result_path)
    dataset_names = tuple(str(name) for name in datasets)
    candidates: list[tuple[str, Path]] = [("combined", combined_path)]
    candidates.extend(
        (
            dataset_name,
            combined_path.with_name(
                f"{combined_path.stem}_{dataset_name}{combined_path.suffix}"
            ),
        )
        for dataset_name in dataset_names
    )

    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    for source, path in candidates:
        if not path.exists():
            status_rows.append(
                {
                    "source": source,
                    "path": str(path),
                    "available": False,
                    "readable": False,
                    "rows": 0,
                    "error": None,
                }
            )
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # file may be observed during atomic persistence
            status_rows.append(
                {
                    "source": source,
                    "path": str(path),
                    "available": True,
                    "readable": False,
                    "rows": 0,
                    "error": str(exc),
                }
            )
            continue
        frames.append(frame)
        status_rows.append(
            {
                "source": source,
                "path": str(path),
                "available": True,
                "readable": True,
                "rows": int(len(frame)),
                "error": None,
            }
        )

    status_df = pd.DataFrame(status_rows)
    if not frames:
        return pd.DataFrame(), status_df

    result = pd.concat(frames, ignore_index=True, sort=False)
    required_identity = [*NORM_ABLATION_TASK_KEY, "run_name"]
    missing = [column for column in required_identity if column not in result]
    if missing:
        raise ValueError(f"Norm-ablation result is missing columns: {missing}")
    result = result.drop_duplicates(required_identity, keep="last").reset_index(drop=True)
    result = normalize_benchmark_df(
        result,
        method_labels=NORM_ABLATION_LABELS,
    )
    return result, status_df


def norm_ablation_completion_table(
    df: pd.DataFrame,
    *,
    datasets: Iterable[str] = NORM_ABLATION_DATASETS,
    runs: Iterable[str] = NORM_ABLATION_RUNS,
    expected_tasks: int | Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Return one completion row for every expected dataset/run combination."""
    rows: list[dict[str, object]] = []
    dataset_names = tuple(str(name) for name in datasets)
    run_names = tuple(str(name) for name in runs)
    for dataset_name in dataset_names:
        dataset_expected_tasks = _expected_tasks_for_dataset(
            dataset_name,
            expected_tasks,
        )
        for run_name in run_names:
            if df.empty:
                part = df
            else:
                part = df[
                    df["dataset"].astype(str).eq(dataset_name)
                    & df["run_name"].astype(str).eq(run_name)
                ]
            unique_tasks = (
                part[list(NORM_ABLATION_TASK_KEY)].drop_duplicates().shape[0]
                if len(part)
                else 0
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "run_name": run_name,
                    "run_label": NORM_ABLATION_LABELS.get(run_name, run_name),
                    "rows": int(len(part)),
                    "unique_tasks": int(unique_tasks),
                    "successes": (
                        int(part["success"].fillna(False).astype(bool).sum())
                        if len(part)
                        else 0
                    ),
                    "expected_tasks": dataset_expected_tasks,
                    "complete": bool(
                        len(part) == dataset_expected_tasks
                        and unique_tasks == dataset_expected_tasks
                    ),
                }
            )
    return pd.DataFrame(rows)


def validate_norm_ablation_results(
    df: pd.DataFrame,
    *,
    datasets: Iterable[str] = NORM_ABLATION_DATASETS,
    runs: Iterable[str] = NORM_ABLATION_RUNS,
    expected_tasks: int | Mapping[str, int] | None = None,
) -> None:
    """Validate a completed paired ablation, raising on any mismatch."""
    if df.empty:
        raise ValueError("Norm-ablation result is empty.")
    required = {
        *NORM_ABLATION_TASK_KEY,
        "run_name",
        "success",
        "l1_distance",
        "l2_distance",
        "l0_sparsity",
        "runtime_s",
        "build_time_s",
        "meta__distance_norm",
        "meta__sparsity_penalty",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Norm-ablation result is missing columns: {missing}")

    completion = norm_ablation_completion_table(
        df,
        datasets=datasets,
        runs=runs,
        expected_tasks=expected_tasks,
    )
    if not bool(completion["complete"].all()):
        incomplete = completion.loc[
            ~completion["complete"], ["dataset", "run_name", "unique_tasks"]
        ]
        raise ValueError(
            "Norm-ablation result is incomplete:\n" + incomplete.to_string(index=False)
        )

    duplicate_key = [*NORM_ABLATION_TASK_KEY, "run_name"]
    if bool(df.duplicated(duplicate_key).any()):
        raise ValueError("Norm-ablation result contains duplicate task/run rows.")

    expected_runs = tuple(str(name) for name in runs)
    expected_task_sets: dict[str, set[tuple[object, ...]]] = {}
    for run_name in expected_runs:
        part = df[df["run_name"].astype(str).eq(run_name)]
        expected_task_sets[run_name] = set(
            map(tuple, part[list(NORM_ABLATION_TASK_KEY)].itertuples(index=False, name=None))
        )
    first_tasks = expected_task_sets[expected_runs[0]]
    for run_name in expected_runs[1:]:
        if expected_task_sets[run_name] != first_tasks:
            raise ValueError(f"Task set for {run_name!r} does not match the L1 control.")

    expected_norm = {
        "certcf_query_l1_no_sparsity": 1.0,
        "certcf_query_l2_no_sparsity": 2.0,
    }
    for run_name, norm_value in expected_norm.items():
        part = df[df["run_name"].astype(str).eq(run_name)]
        observed_norms = set(
            pd.to_numeric(part["meta__distance_norm"], errors="coerce")
            .dropna()
            .astype(float)
        )
        if observed_norms != {norm_value}:
            raise ValueError(
                f"{run_name} has distance norms {observed_norms}; expected {norm_value}."
            )
        penalties = set(part["meta__sparsity_penalty"].dropna().astype(str))
        if penalties != {"none"}:
            raise ValueError(
                f"{run_name} has sparsity penalties {penalties}; expected only 'none'."
            )


def attach_l0_count(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the changed-feature count using each dataset's encoded dimension."""
    out = df.copy()
    out["n_features"] = np.nan
    for dataset_name, group in out.groupby("dataset", observed=True):
        feature_columns = [
            column
            for column in group.columns
            if column.startswith("x_orig_") and group[column].notna().any()
        ]
        if not feature_columns:
            raise ValueError(
                f"Cannot infer encoded feature count for dataset {dataset_name!r}."
            )
        out.loc[group.index, "n_features"] = len(feature_columns)
    out["n_features"] = out["n_features"].astype(int)
    out["l0_count"] = (
        pd.to_numeric(out["l0_sparsity"], errors="coerce") * out["n_features"]
    )
    return out


def shared_success_norm_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep tasks present and successful under both query objectives."""
    success_grid = df.pivot_table(
        index=list(NORM_ABLATION_TASK_KEY),
        columns="run_name",
        values="success",
        aggfunc="first",
    ).reindex(columns=list(NORM_ABLATION_RUNS))
    shared_keys = success_grid.index[
        success_grid.notna().all(axis=1)
        & success_grid.fillna(False).astype(bool).all(axis=1)
    ]
    if not len(shared_keys):
        return df.iloc[0:0].copy()
    shared_key_df = shared_keys.to_frame(index=False)
    return df.merge(shared_key_df, on=list(NORM_ABLATION_TASK_KEY), how="inner")


def norm_ablation_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize validity on all tasks and quality on paired successes."""
    data = attach_l0_count(df)
    validity = (
        data.groupby(["dataset", "run_name"], observed=True)
        .agg(
            n_total=("success", "size"),
            n_success=("success", "sum"),
            validity_pct=("success", lambda values: 100.0 * values.mean()),
            query_time_mean_s=("runtime_s", "mean"),
            query_time_median_s=("runtime_s", "median"),
            build_time_s=("build_time_s", "first"),
        )
        .reset_index()
    )
    shared = shared_success_norm_subset(data)
    quality = (
        shared.groupby(["dataset", "run_name"], observed=True)
        .agg(
            n_shared_success=("success", "size"),
            l1_mean=("l1_distance", "mean"),
            l1_median=("l1_distance", "median"),
            l2_mean=("l2_distance", "mean"),
            l2_median=("l2_distance", "median"),
            l0_count_mean=("l0_count", "mean"),
            l0_count_median=("l0_count", "median"),
            l0_fraction_mean=("l0_sparsity", "mean"),
        )
        .reset_index()
    )
    summary = validity.merge(quality, on=["dataset", "run_name"], how="left")
    summary["run_label"] = summary["run_name"].map(NORM_ABLATION_LABELS)
    return summary.sort_values(["dataset", "run_name"], ignore_index=True)


def norm_ablation_aggregate_summary(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the seven dataset-level values with mean and sample std."""
    metrics = [
        "validity_pct",
        "l1_mean",
        "l2_mean",
        "l0_count_mean",
        "l0_fraction_mean",
        "query_time_mean_s",
        "query_time_median_s",
    ]
    rows: list[dict[str, object]] = []
    for run_name, group in dataset_summary.groupby("run_name", observed=True):
        row: dict[str, object] = {
            "run_name": run_name,
            "run_label": NORM_ABLATION_LABELS.get(str(run_name), str(run_name)),
            "n_datasets": int(group["dataset"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("run_name", ignore_index=True)


def norm_ablation_paired_deltas(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return query-level and dataset-level L2-minus-L1 paired differences."""
    data = attach_l0_count(shared_success_norm_subset(df))
    value_columns = ["l1_distance", "l2_distance", "l0_count", "runtime_s"]
    pivot = data.pivot_table(
        index=list(NORM_ABLATION_TASK_KEY),
        columns="run_name",
        values=value_columns,
        aggfunc="first",
    )
    rows: dict[str, np.ndarray] = {}
    for metric in value_columns:
        l1_values = pivot[(metric, NORM_ABLATION_RUNS[0])].to_numpy(dtype=float)
        l2_values = pivot[(metric, NORM_ABLATION_RUNS[1])].to_numpy(dtype=float)
        rows[f"{metric}_l1"] = l1_values
        rows[f"{metric}_l2"] = l2_values
        rows[f"{metric}_delta_l2_minus_l1"] = l2_values - l1_values
        rows[f"{metric}_pct_change"] = np.divide(
            100.0 * (l2_values - l1_values),
            l1_values,
            out=np.full_like(l1_values, np.nan),
            where=np.abs(l1_values) > 1.0e-12,
        )
    query_deltas = pivot.index.to_frame(index=False).reset_index(drop=True)
    for column, values in rows.items():
        query_deltas[column] = values

    absolute_delta_columns = [
        column
        for column in query_deltas
        if column.endswith("_delta_l2_minus_l1")
    ]
    dataset_deltas = (
        query_deltas.groupby("dataset", observed=True)[absolute_delta_columns]
        .mean()
        .reset_index()
    )
    for metric in value_columns:
        means = query_deltas.groupby("dataset", observed=True)[
            [f"{metric}_l1", f"{metric}_l2"]
        ].mean()
        dataset_deltas[f"{metric}_pct_change"] = np.divide(
            100.0 * (means[f"{metric}_l2"] - means[f"{metric}_l1"]),
            means[f"{metric}_l1"],
            out=np.full(len(means), np.nan, dtype=float),
            where=np.abs(means[f"{metric}_l1"]) > 1.0e-12,
        ).to_numpy()
    return query_deltas, dataset_deltas


__all__ = [
    "NORM_ABLATION_DATASETS",
    "NORM_ABLATION_EXPECTED_TASKS",
    "NORM_ABLATION_LABELS",
    "NORM_ABLATION_RUNS",
    "NORM_ABLATION_TASK_KEY",
    "attach_l0_count",
    "load_norm_ablation_results",
    "norm_ablation_aggregate_summary",
    "norm_ablation_completion_table",
    "norm_ablation_dataset_summary",
    "norm_ablation_paired_deltas",
    "shared_success_norm_subset",
    "validate_norm_ablation_results",
]
