#!/usr/bin/env python3
"""Compare the parallel CertCF re-run with the selected original run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OLD_RUN = (
    "certcf_backward_shrink_sparsity_"
    "eps_alpha=0.2-sparsity_lambda=1.0"
)
KEY_COLUMNS = ["dataset", "query_idx", "source_class", "target_class"]


def _numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda column: int(column.removeprefix(prefix)))


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


def _rowwise_vector_comparison(
    old_values: np.ndarray,
    new_values: np.ndarray,
    *,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-wise equality and maximum difference, including NaN layout."""
    old_finite = np.isfinite(old_values)
    new_finite = np.isfinite(new_values)
    same_layout = np.all(old_finite == new_finite, axis=1)
    comparable = old_finite & new_finite
    abs_diff = np.where(comparable, np.abs(old_values - new_values), np.nan)
    max_abs_diff = np.asarray([_finite_max(row) for row in abs_diff], dtype=float)
    return same_layout & (max_abs_diff <= float(atol)), max_abs_diff


def compare(
    old_path: Path,
    new_path: Path,
    old_run: str,
    new_run: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    old = pd.read_parquet(old_path)
    new = pd.read_parquet(new_path)
    old = old.loc[old["run_name"].eq(old_run)].copy()
    if new_run is not None:
        new = new.loc[new["run_name"].eq(new_run)].copy()

    if old.empty:
        raise ValueError(f"Original run {old_run!r} was not found in {old_path}")
    if new.empty:
        selected = f" {new_run!r}" if new_run is not None else ""
        raise ValueError(f"New run{selected} was not found in {new_path}")
    for label, frame in (("old", old), ("new", new)):
        duplicates = int(frame.duplicated(KEY_COLUMNS).sum())
        if duplicates:
            raise ValueError(f"{label} artifact has {duplicates} duplicate query keys")

    old_keys = old[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    new_keys = new[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    keys_identical = old_keys.equals(new_keys)

    merged = old.merge(
        new,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    matched = merged.loc[merged["_merge"].eq("both")].copy()

    old_cf_columns = _numbered_columns(old, "x_cf_")
    new_cf_columns = _numbered_columns(new, "x_cf_")
    shared_dimensions = min(len(old_cf_columns), len(new_cf_columns))
    old_cf_columns = old_cf_columns[:shared_dimensions]
    new_cf_columns = new_cf_columns[:shared_dimensions]
    old_cf = matched[[f"{column}_old" for column in old_cf_columns]].to_numpy(float)
    new_cf = matched[[f"{column}_new" for column in new_cf_columns]].to_numpy(float)
    cf_allclose_1e6, cf_max_abs_diff = _rowwise_vector_comparison(
        old_cf,
        new_cf,
        atol=1.0e-6,
    )
    cf_allclose_1e5, _ = _rowwise_vector_comparison(
        old_cf,
        new_cf,
        atol=1.0e-5,
    )
    matched["cf_max_abs_diff"] = cf_max_abs_diff
    matched["cf_allclose_1e-6"] = cf_allclose_1e6
    matched["cf_allclose_1e-5"] = cf_allclose_1e5
    matched["same_target_result"] = matched["target_reached_old"].eq(
        matched["target_reached_new"]
    )
    # ``meta__anchor_idx`` is the region that produced the returned result.
    # ``meta__nearest_anchor_candidate_idx`` only describes initialization and
    # must not be used to assess equivalence of the final selection.
    old_anchor_column = "meta__anchor_idx_old"
    new_anchor_column = "meta__anchor_idx_new"
    if old_anchor_column in matched and new_anchor_column in matched:
        matched["anchor_indices_comparable"] = matched[old_anchor_column].notna() & matched[
            new_anchor_column
        ].notna()
        matched["same_anchor_index"] = matched["anchor_indices_comparable"] & matched[
            old_anchor_column
        ].eq(matched[new_anchor_column])
    else:
        # Older benchmark artifacts did not persist the final selected region.
        # The nearest initialization anchor is not an equivalent substitute.
        matched["anchor_indices_comparable"] = False
        matched["same_anchor_index"] = False
    for metric in ("l0_sparsity", "l1_distance", "l2_distance"):
        difference = np.abs(
            matched[f"{metric}_old"].astype(float)
            - matched[f"{metric}_new"].astype(float)
        )
        matched[f"{metric}_abs_diff"] = difference
        matched[f"same_{metric}_1e-6"] = difference.le(1.0e-6)

    rows: list[dict] = []
    for dataset, group in matched.groupby("dataset", sort=False):
        old_runtime = group["runtime_s_old"].astype(float)
        new_runtime = group["runtime_s_new"].astype(float)
        paired_speedup = old_runtime / new_runtime
        old_mean = float(old_runtime.mean())
        new_mean = float(new_runtime.mean())
        old_median = float(old_runtime.median())
        new_median = float(new_runtime.median())
        rows.append(
            {
                "dataset": dataset,
                "queries": int(len(group)),
                "old_target_successes": int(group["target_reached_old"].sum()),
                "new_target_successes": int(group["target_reached_new"].sum()),
                "old_mean_runtime_s": old_mean,
                "new_mean_runtime_s": new_mean,
                "mean_runtime_speedup": old_mean / new_mean,
                "old_median_runtime_s": old_median,
                "new_median_runtime_s": new_median,
                "median_runtime_speedup": old_median / new_median,
                "old_p95_runtime_s": float(old_runtime.quantile(0.95)),
                "new_p95_runtime_s": float(new_runtime.quantile(0.95)),
                "paired_speedup_median": float(paired_speedup.median()),
                "paired_speedup_p05": float(paired_speedup.quantile(0.05)),
                "paired_speedup_p95": float(paired_speedup.quantile(0.95)),
                "queries_faster_pct": float(100.0 * paired_speedup.gt(1.0).mean()),
                "counterfactuals_allclose_1e-6": int(group["cf_allclose_1e-6"].sum()),
                "counterfactuals_allclose_1e-5": int(group["cf_allclose_1e-5"].sum()),
                "anchor_indices_comparable": int(
                    group["anchor_indices_comparable"].sum()
                ),
                "same_anchor_index": int(group["same_anchor_index"].sum()),
                "maximum_counterfactual_abs_diff": float(group["cf_max_abs_diff"].max()),
                "same_l0_1e-6": int(group["same_l0_sparsity_1e-6"].sum()),
                "same_l1_1e-6": int(group["same_l1_distance_1e-6"].sum()),
                "same_l2_1e-6": int(group["same_l2_distance_1e-6"].sum()),
                "maximum_l0_abs_diff": float(group["l0_sparsity_abs_diff"].max()),
                "maximum_l1_abs_diff": float(group["l1_distance_abs_diff"].max()),
                "maximum_l2_abs_diff": float(group["l2_distance_abs_diff"].max()),
                "mean_l0_old": float(group["l0_sparsity_old"].mean()),
                "mean_l0_new": float(group["l0_sparsity_new"].mean()),
                "mean_l1_old": float(group["l1_distance_old"].mean()),
                "mean_l1_new": float(group["l1_distance_new"].mean()),
                "mean_l2_old": float(group["l2_distance_old"].mean()),
                "mean_l2_new": float(group["l2_distance_new"].mean()),
            }
        )
    summary = pd.DataFrame(rows)

    old_total = float(matched["runtime_s_old"].sum())
    new_total = float(matched["runtime_s_new"].sum())
    paired_speedup = matched["runtime_s_old"].astype(float) / matched[
        "runtime_s_new"
    ].astype(float)
    report = {
        "old_path": str(old_path.resolve()),
        "new_path": str(new_path.resolve()),
        "old_run": old_run,
        "new_run": new_run,
        "old_rows": int(len(old)),
        "new_rows": int(len(new)),
        "matched_rows": int(len(matched)),
        "keys_identical": bool(keys_identical),
        "old_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "new_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "old_target_successes": int(matched["target_reached_old"].sum()),
        "new_target_successes": int(matched["target_reached_new"].sum()),
        "same_target_result": int(matched["same_target_result"].sum()),
        "counterfactuals_allclose_1e-6": int(matched["cf_allclose_1e-6"].sum()),
        "counterfactuals_allclose_1e-5": int(matched["cf_allclose_1e-5"].sum()),
        "anchor_indices_comparable": int(matched["anchor_indices_comparable"].sum()),
        "same_anchor_index": int(matched["same_anchor_index"].sum()),
        "maximum_counterfactual_abs_diff": float(matched["cf_max_abs_diff"].max()),
        "same_l0_1e-6": int(matched["same_l0_sparsity_1e-6"].sum()),
        "same_l1_1e-6": int(matched["same_l1_distance_1e-6"].sum()),
        "same_l2_1e-6": int(matched["same_l2_distance_1e-6"].sum()),
        "maximum_l0_abs_diff": float(matched["l0_sparsity_abs_diff"].max()),
        "maximum_l1_abs_diff": float(matched["l1_distance_abs_diff"].max()),
        "maximum_l2_abs_diff": float(matched["l2_distance_abs_diff"].max()),
        "old_total_runtime_s": old_total,
        "new_total_runtime_s": new_total,
        "aggregate_runtime_speedup": old_total / new_total,
        "old_mean_runtime_s": float(matched["runtime_s_old"].mean()),
        "new_mean_runtime_s": float(matched["runtime_s_new"].mean()),
        "old_median_runtime_s": float(matched["runtime_s_old"].median()),
        "new_median_runtime_s": float(matched["runtime_s_new"].median()),
        "old_p95_runtime_s": float(matched["runtime_s_old"].quantile(0.95)),
        "new_p95_runtime_s": float(matched["runtime_s_new"].quantile(0.95)),
        "paired_speedup_median": float(paired_speedup.median()),
        "paired_speedup_p05": float(paired_speedup.quantile(0.05)),
        "paired_speedup_p95": float(paired_speedup.quantile(0.95)),
        "queries_faster_pct": float(100.0 * paired_speedup.gt(1.0).mean()),
        "datasets": summary.to_dict(orient="records"),
    }
    return summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old",
        type=Path,
        default=Path("results/final_benchmark_noface_certcf_shrink_sparsity.parquet"),
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=Path("results/final_benchmark_certcf_parallel.parquet"),
    )
    parser.add_argument("--old-run", default=DEFAULT_OLD_RUN)
    parser.add_argument(
        "--new-run",
        default=None,
        help="Optionally select one run_name from a multi-run new artifact.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/final_benchmark_certcf_parallel_comparison.json"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/final_benchmark_certcf_parallel_comparison.parquet"),
    )
    args = parser.parse_args()

    summary, report = compare(args.old, args.new, args.old_run, args.new_run)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary.to_parquet(args.output_summary, index=False)

    print(summary.to_string(index=False))
    print(json.dumps({key: value for key, value in report.items() if key != "datasets"}, indent=2))


if __name__ == "__main__":
    main()
