#!/usr/bin/env python3
"""Update only CertCF query times in the VeriX comparison from the parallel grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["architecture_id", "query_index"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired",
        type=Path,
        default=Path(
            "results/verix_certcf_synthetic32_grid/verix_certcf_queries.parquet"
        ),
    )
    parser.add_argument(
        "--parallel-grid",
        type=Path,
        default=Path("results/network_complexity/network_complexity_queries.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/verix_certcf_synthetic32_grid/"
            "verix_certcf_queries_parallel.parquet"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "results/verix_certcf_synthetic32_grid/"
            "certcf_parallel_timing_comparison.json"
        ),
    )
    parser.add_argument("--l1-atol", type=float, default=1.0e-5)
    args = parser.parse_args()

    paired = pd.read_parquet(args.paired)
    grid = pd.read_parquet(args.parallel_grid)
    certcf_mask = paired["method"].astype(str).str.lower().eq("certcf")
    certcf = paired.loc[certcf_mask].copy()
    if certcf.duplicated(KEYS).any():
        raise ValueError("Paired CertCF rows contain duplicate architecture/query keys")

    grid = grid.rename(columns={"query_idx": "query_index"})
    if grid.duplicated(KEYS).any():
        raise ValueError("Parallel grid contains duplicate architecture/query keys")
    selected = certcf.merge(
        grid[
            [
                *KEYS,
                "success",
                "y_cf",
                "l1_distance",
                "runtime_s",
            ]
        ],
        on=KEYS,
        how="left",
        suffixes=("_paired", "_parallel"),
        validate="one_to_one",
        indicator=True,
    )
    if not selected["_merge"].eq("both").all():
        missing = selected.loc[~selected["_merge"].eq("both"), KEYS]
        raise ValueError(f"Missing parallel grid rows:\n{missing.to_string(index=False)}")
    same_success = selected["success_paired"].eq(selected["success_parallel"])
    same_prediction = selected["predicted_class"].eq(selected["y_cf"])
    l1_difference = np.abs(
        selected["l1_distance_paired"].astype(float)
        - selected["l1_distance_parallel"].astype(float)
    )
    if not same_success.all():
        raise ValueError("Serial comparison and parallel grid disagree on success")
    if not same_prediction.all():
        raise ValueError("Serial comparison and parallel grid disagree on target prediction")
    if not l1_difference.le(float(args.l1_atol)).all():
        raise ValueError(
            "Counterfactual L1 distances differ beyond tolerance; a timing-only update "
            "would not be valid"
        )

    runtime_by_key = selected.set_index(KEYS)["runtime_s"]
    updated = paired.copy()
    certcf_keys = pd.MultiIndex.from_frame(updated.loc[certcf_mask, KEYS])
    updated.loc[certcf_mask, "runtime_seconds"] = runtime_by_key.reindex(
        certcf_keys
    ).to_numpy(dtype=float)

    verix_success_keys = pd.MultiIndex.from_frame(
        paired.loc[
            paired["method"].astype(str).str.lower().eq("verix")
            & paired["success"].astype(bool),
            KEYS,
        ]
    )
    original_runtime = certcf.set_index(KEYS)["runtime_seconds"].astype(float)
    parallel_runtime = runtime_by_key.astype(float)
    shared_original = original_runtime.reindex(verix_success_keys).dropna()
    shared_parallel = parallel_runtime.reindex(verix_success_keys).dropna()
    report = {
        "paired_path": str(args.paired.resolve()),
        "parallel_grid_path": str(args.parallel_grid.resolve()),
        "output_path": str(args.output.resolve()),
        "certcf_rows": int(len(certcf)),
        "same_success": int(same_success.sum()),
        "same_predicted_class": int(same_prediction.sum()),
        "l1_atol": float(args.l1_atol),
        "maximum_l1_abs_diff": float(l1_difference.max()),
        "all_certcf": {
            "serial_mean_runtime_s": float(original_runtime.mean()),
            "parallel_mean_runtime_s": float(parallel_runtime.mean()),
            "serial_median_runtime_s": float(original_runtime.median()),
            "parallel_median_runtime_s": float(parallel_runtime.median()),
            "serial_total_runtime_s": float(original_runtime.sum()),
            "parallel_total_runtime_s": float(parallel_runtime.sum()),
            "total_runtime_speedup": float(
                original_runtime.sum() / parallel_runtime.sum()
            ),
        },
        "shared_success": {
            "rows": int(len(shared_original)),
            "serial_mean_runtime_s": float(shared_original.mean()),
            "parallel_mean_runtime_s": float(shared_parallel.mean()),
            "serial_median_runtime_s": float(shared_original.median()),
            "parallel_median_runtime_s": float(shared_parallel.median()),
            "serial_total_runtime_s": float(shared_original.sum()),
            "parallel_total_runtime_s": float(shared_parallel.sum()),
            "total_runtime_speedup": float(
                shared_original.sum() / shared_parallel.sum()
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
