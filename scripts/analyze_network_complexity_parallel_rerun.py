#!/usr/bin/env python3
"""Compare serial and parallel CertCF network-complexity query artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["architecture_id", "query_position", "query_idx"]


def _numbered_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda column: int(column.removeprefix(prefix)))


def _anchor_index(value: object) -> float:
    if not isinstance(value, str) or not value:
        return float("nan")
    try:
        metadata = json.loads(value)
    except json.JSONDecodeError:
        return float("nan")
    result = metadata.get("anchor_idx", float("nan"))
    return float(result) if result is not None else float("nan")


def compare(old_path: Path, new_path: Path, *, atol: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    old = pd.read_parquet(old_path)
    new = pd.read_parquet(new_path)
    for label, frame in (("serial", old), ("parallel", new)):
        duplicates = int(frame.duplicated(KEY_COLUMNS).sum())
        if duplicates:
            raise ValueError(f"{label} artifact has {duplicates} duplicate query keys")

    old_cf_columns = _numbered_columns(old, "x_cf_")
    new_cf_columns = _numbered_columns(new, "x_cf_")
    if old_cf_columns != new_cf_columns:
        raise ValueError("Serial and parallel artifacts have different counterfactual columns")

    merged = old.merge(
        new,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_serial", "_parallel"),
        indicator=True,
        validate="one_to_one",
    )
    paired = merged.loc[merged["_merge"].eq("both")].copy()
    serial_cf = paired[[f"{column}_serial" for column in old_cf_columns]].to_numpy(float)
    parallel_cf = paired[[f"{column}_parallel" for column in new_cf_columns]].to_numpy(float)
    same_nan_layout = np.all(np.isnan(serial_cf) == np.isnan(parallel_cf), axis=1)
    differences = np.abs(serial_cf - parallel_cf)
    finite_differences = np.where(np.isfinite(differences), differences, 0.0)
    paired["counterfactual_max_abs_diff"] = finite_differences.max(axis=1)
    paired["counterfactual_allclose"] = same_nan_layout & np.all(
        finite_differences <= float(atol), axis=1
    )
    paired["anchor_idx_serial"] = paired["metadata_json_serial"].map(_anchor_index)
    paired["anchor_idx_parallel"] = paired["metadata_json_parallel"].map(_anchor_index)
    paired["anchor_indices_comparable"] = paired["anchor_idx_serial"].notna() & paired[
        "anchor_idx_parallel"
    ].notna()
    paired["same_anchor_index"] = paired["anchor_indices_comparable"] & paired[
        "anchor_idx_serial"
    ].eq(paired["anchor_idx_parallel"])
    paired["same_success"] = paired["success_serial"].eq(paired["success_parallel"])
    paired["same_predicted_class"] = paired["y_cf_serial"].eq(paired["y_cf_parallel"])
    paired["l1_abs_diff"] = np.abs(
        paired["l1_distance_serial"].astype(float)
        - paired["l1_distance_parallel"].astype(float)
    )
    paired["paired_speedup"] = (
        paired["runtime_s_serial"].astype(float)
        / paired["runtime_s_parallel"].astype(float)
    )

    summaries: list[dict] = []
    for architecture_id, group in paired.groupby("architecture_id", sort=True):
        serial_runtime = group["runtime_s_serial"].astype(float)
        parallel_runtime = group["runtime_s_parallel"].astype(float)
        serial_total = float(serial_runtime.sum())
        parallel_total = float(parallel_runtime.sum())
        summaries.append(
            {
                "architecture_id": architecture_id,
                "depth": int(group["depth_serial"].iloc[0]),
                "width": int(group["width_serial"].iloc[0]),
                "parameter_count": int(group["parameter_count_serial"].iloc[0]),
                "queries": int(len(group)),
                "serial_successes": int(group["success_serial"].sum()),
                "parallel_successes": int(group["success_parallel"].sum()),
                "serial_total_runtime_s": serial_total,
                "parallel_total_runtime_s": parallel_total,
                "total_runtime_speedup": serial_total / parallel_total,
                "serial_mean_runtime_s": float(serial_runtime.mean()),
                "parallel_mean_runtime_s": float(parallel_runtime.mean()),
                "serial_median_runtime_s": float(serial_runtime.median()),
                "parallel_median_runtime_s": float(parallel_runtime.median()),
                "serial_p95_runtime_s": float(serial_runtime.quantile(0.95)),
                "parallel_p95_runtime_s": float(parallel_runtime.quantile(0.95)),
                "paired_speedup_median": float(group["paired_speedup"].median()),
                "queries_faster_pct": float(100.0 * group["paired_speedup"].gt(1.0).mean()),
                "same_success": int(group["same_success"].sum()),
                "same_predicted_class": int(group["same_predicted_class"].sum()),
                "anchor_indices_comparable": int(
                    group["anchor_indices_comparable"].sum()
                ),
                "same_anchor_index": int(group["same_anchor_index"].sum()),
                "counterfactuals_allclose": int(group["counterfactual_allclose"].sum()),
                "maximum_counterfactual_abs_diff": float(
                    group["counterfactual_max_abs_diff"].max()
                ),
                "maximum_l1_abs_diff": float(group["l1_abs_diff"].max()),
            }
        )
    summary = pd.DataFrame(summaries)

    serial_total = float(paired["runtime_s_serial"].sum())
    parallel_total = float(paired["runtime_s_parallel"].sum())
    report = {
        "serial_path": str(old_path.resolve()),
        "parallel_path": str(new_path.resolve()),
        "serial_rows": int(len(old)),
        "parallel_rows": int(len(new)),
        "matched_rows": int(len(paired)),
        "serial_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "parallel_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "architectures": int(paired["architecture_id"].nunique()),
        "serial_successes": int(paired["success_serial"].sum()),
        "parallel_successes": int(paired["success_parallel"].sum()),
        "same_success": int(paired["same_success"].sum()),
        "same_predicted_class": int(paired["same_predicted_class"].sum()),
        "anchor_indices_comparable": int(paired["anchor_indices_comparable"].sum()),
        "same_anchor_index": int(paired["same_anchor_index"].sum()),
        "counterfactuals_allclose": int(paired["counterfactual_allclose"].sum()),
        "comparison_atol": float(atol),
        "maximum_counterfactual_abs_diff": float(
            paired["counterfactual_max_abs_diff"].max()
        ),
        "maximum_l1_abs_diff": float(paired["l1_abs_diff"].max()),
        "serial_total_runtime_s": serial_total,
        "parallel_total_runtime_s": parallel_total,
        "aggregate_runtime_speedup": serial_total / parallel_total,
        "serial_mean_runtime_s": float(paired["runtime_s_serial"].mean()),
        "parallel_mean_runtime_s": float(paired["runtime_s_parallel"].mean()),
        "serial_median_runtime_s": float(paired["runtime_s_serial"].median()),
        "parallel_median_runtime_s": float(paired["runtime_s_parallel"].median()),
        "serial_p95_runtime_s": float(paired["runtime_s_serial"].quantile(0.95)),
        "parallel_p95_runtime_s": float(paired["runtime_s_parallel"].quantile(0.95)),
        "paired_speedup_median": float(paired["paired_speedup"].median()),
        "paired_speedup_p05": float(paired["paired_speedup"].quantile(0.05)),
        "paired_speedup_p95": float(paired["paired_speedup"].quantile(0.95)),
        "queries_faster_pct": float(100.0 * paired["paired_speedup"].gt(1.0).mean()),
        "per_architecture": summary.to_dict(orient="records"),
    }
    return paired, summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial",
        type=Path,
        default=Path(
            "results/network_complexity/archive/serial_before_parallel_2026-09-05/"
            "network_complexity_queries.parquet"
        ),
    )
    parser.add_argument(
        "--parallel",
        type=Path,
        default=Path("results/network_complexity/network_complexity_queries.parquet"),
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=Path("results/network_complexity/parallel_rerun_pairs.parquet"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/network_complexity/parallel_rerun_summary.parquet"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/network_complexity/parallel_rerun_comparison.json"),
    )
    parser.add_argument("--atol", type=float, default=1.0e-5)
    args = parser.parse_args()

    pairs, summary, report = compare(args.serial, args.parallel, atol=args.atol)
    for path in (args.output_pairs, args.output_summary, args.output_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.output_pairs, index=False)
    summary.to_parquet(args.output_summary, index=False)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(json.dumps({key: value for key, value in report.items() if key != "per_architecture"}, indent=2))


if __name__ == "__main__":
    main()
