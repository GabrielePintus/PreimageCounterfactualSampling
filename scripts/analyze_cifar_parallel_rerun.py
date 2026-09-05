#!/usr/bin/env python3
"""Compare saved serial and parallel CIFAR ResNet query artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


NETWORKS = ("resnet20", "resnet32", "resnet56")


def _load(root: Path, networks: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict] = []
    for network in networks:
        query_dir = root / network / "queries"
        for json_path in sorted(query_dir.glob("query_*.json")):
            record = json.loads(json_path.read_text(encoding="utf-8"))
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.exists():
                raise FileNotFoundError(npz_path)
            arrays = np.load(npz_path)
            rows.append(
                {
                    **record,
                    "network": network,
                    "query_position": int(record["query_position"]),
                    "x_cf": np.asarray(arrays["x_cf"], dtype=float),
                }
            )
    return pd.DataFrame(rows)


def compare(
    serial_root: Path,
    parallel_root: Path,
    networks: tuple[str, ...],
    *,
    atol: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    serial = _load(serial_root, networks)
    parallel = _load(parallel_root, networks)
    keys = ["network", "query_position"]
    for label, frame in (("serial", serial), ("parallel", parallel)):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} artifacts contain duplicate query keys")

    paired = serial.merge(
        parallel,
        on=keys,
        how="outer",
        suffixes=("_serial", "_parallel"),
        indicator=True,
        validate="one_to_one",
    )
    matched = paired.loc[paired["_merge"].eq("both")].copy()
    differences: list[float] = []
    allclose: list[bool] = []
    for serial_cf, parallel_cf in zip(
        matched["x_cf_serial"], matched["x_cf_parallel"], strict=True
    ):
        if serial_cf.shape != parallel_cf.shape:
            differences.append(float("inf"))
            allclose.append(False)
            continue
        same_nan = np.array_equal(np.isnan(serial_cf), np.isnan(parallel_cf))
        finite_diff = np.where(
            np.isfinite(serial_cf - parallel_cf),
            np.abs(serial_cf - parallel_cf),
            0.0,
        )
        maximum = float(finite_diff.max(initial=0.0))
        differences.append(maximum)
        allclose.append(bool(same_nan and maximum <= atol))
    matched["counterfactual_max_abs_diff"] = differences
    matched["counterfactual_allclose"] = allclose
    matched["same_success"] = matched["success_serial"].eq(matched["success_parallel"])
    matched["same_validity"] = matched["validity_serial"].eq(
        matched["validity_parallel"]
    )
    matched["same_anchor_index"] = matched["anchor_idx_serial"].eq(
        matched["anchor_idx_parallel"]
    )
    matched["paired_speedup"] = (
        matched["runtime_s_serial"].astype(float)
        / matched["runtime_s_parallel"].astype(float)
    )

    summaries: list[dict] = []
    for network, group in matched.groupby("network", sort=True):
        serial_runtime = group["runtime_s_serial"].astype(float)
        parallel_runtime = group["runtime_s_parallel"].astype(float)
        summaries.append(
            {
                "network": network,
                "queries": int(len(group)),
                "serial_successes": int(group["success_serial"].sum()),
                "parallel_successes": int(group["success_parallel"].sum()),
                "serial_total_runtime_s": float(serial_runtime.sum()),
                "parallel_total_runtime_s": float(parallel_runtime.sum()),
                "total_runtime_speedup": float(
                    serial_runtime.sum() / parallel_runtime.sum()
                ),
                "serial_mean_runtime_s": float(serial_runtime.mean()),
                "parallel_mean_runtime_s": float(parallel_runtime.mean()),
                "serial_median_runtime_s": float(serial_runtime.median()),
                "parallel_median_runtime_s": float(parallel_runtime.median()),
                "serial_p95_runtime_s": float(serial_runtime.quantile(0.95)),
                "parallel_p95_runtime_s": float(parallel_runtime.quantile(0.95)),
                "paired_speedup_median": float(group["paired_speedup"].median()),
                "queries_faster_pct": float(
                    100.0 * group["paired_speedup"].gt(1.0).mean()
                ),
                "same_success": int(group["same_success"].sum()),
                "same_validity": int(group["same_validity"].sum()),
                "same_anchor_index": int(group["same_anchor_index"].sum()),
                "counterfactuals_allclose": int(
                    group["counterfactual_allclose"].sum()
                ),
                "maximum_counterfactual_abs_diff": float(
                    group["counterfactual_max_abs_diff"].max()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    serial_total = float(matched["runtime_s_serial"].sum())
    parallel_total = float(matched["runtime_s_parallel"].sum())
    report = {
        "serial_root": str(serial_root.resolve()),
        "parallel_root": str(parallel_root.resolve()),
        "serial_rows": int(len(serial)),
        "parallel_rows": int(len(parallel)),
        "matched_rows": int(len(matched)),
        "serial_only_rows": int(paired["_merge"].eq("left_only").sum()),
        "parallel_only_rows": int(paired["_merge"].eq("right_only").sum()),
        "serial_successes": int(matched["success_serial"].sum()),
        "parallel_successes": int(matched["success_parallel"].sum()),
        "same_success": int(matched["same_success"].sum()),
        "same_validity": int(matched["same_validity"].sum()),
        "same_anchor_index": int(matched["same_anchor_index"].sum()),
        "counterfactuals_allclose": int(matched["counterfactual_allclose"].sum()),
        "comparison_atol": float(atol),
        "maximum_counterfactual_abs_diff": float(
            matched["counterfactual_max_abs_diff"].max()
        ),
        "serial_total_runtime_s": serial_total,
        "parallel_total_runtime_s": parallel_total,
        "aggregate_runtime_speedup": serial_total / parallel_total,
        "serial_mean_runtime_s": float(matched["runtime_s_serial"].mean()),
        "parallel_mean_runtime_s": float(matched["runtime_s_parallel"].mean()),
        "serial_median_runtime_s": float(matched["runtime_s_serial"].median()),
        "parallel_median_runtime_s": float(matched["runtime_s_parallel"].median()),
        "serial_p95_runtime_s": float(matched["runtime_s_serial"].quantile(0.95)),
        "parallel_p95_runtime_s": float(matched["runtime_s_parallel"].quantile(0.95)),
        "paired_speedup_median": float(matched["paired_speedup"].median()),
        "queries_faster_pct": float(100.0 * matched["paired_speedup"].gt(1.0).mean()),
        "per_network": summary.to_dict(orient="records"),
    }
    return matched, summary, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial-root",
        type=Path,
        default=Path(
            "results/cifar_resnet_scaling/archive/"
            "serial_before_parallel_2026-09-05/networks"
        ),
    )
    parser.add_argument(
        "--parallel-root",
        type=Path,
        default=Path("results/cifar_resnet_scaling/networks"),
    )
    parser.add_argument("--networks", nargs="+", choices=NETWORKS, default=list(NETWORKS))
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/cifar_resnet_scaling/parallel_rerun"),
    )
    args = parser.parse_args()

    pairs, summary, report = compare(
        args.serial_root,
        args.parallel_root,
        tuple(args.networks),
        atol=args.atol,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pairs.drop(columns=["x_cf_serial", "x_cf_parallel"]).to_parquet(
        args.output_prefix.with_name(args.output_prefix.name + "_pairs.parquet"),
        index=False,
    )
    summary.to_parquet(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.parquet"),
        index=False,
    )
    args.output_prefix.with_name(args.output_prefix.name + "_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_network"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
