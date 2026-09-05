"""Aggregate completed end-to-end top-k stress-test seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from end_to_end import HERE


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "outputs" / "seeds")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "aggregate")
    args = parser.parse_args()

    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input.glob("seed_*/summary.json"))
    ]
    if not summaries:
        raise RuntimeError(f"No seed summaries found under {args.input}")

    rows = []
    for run in summaries:
        query = run["queries"]
        atlas = run["atlas"]
        rows.append(
            {
                "seed": run["configuration"]["seed"],
                "queries": query["n_queries"],
                "failure_rate": query["failure_rate"],
                "fallback_rate": query["fallback_rate"],
                "top_k_validity": query["top_k_target_validity"],
                "exhaustive_validity": query["exhaustive_target_validity"],
                "mean_top_k_distance": query["mean_top_k_distance"],
                "mean_exhaustive_distance": query["mean_exhaustive_distance"],
                "mean_gap": query["mean_gap"],
                "minimum_failure_gap": query["minimum_gap_on_failure"],
                "mean_ratio": query["mean_ratio"],
                "mean_exhaustive_rank": query["exhaustive_rank_mean"],
                "useful_anchor_count": atlas["role_counts"]["useful"],
                "decoy_initial_epsilon": atlas["role_epsilon"]["decoy_1"]["mean_initial"],
                "decoy_final_epsilon": atlas["role_epsilon"]["decoy_1"]["mean_final"],
                "decoy_shrinks": atlas["role_epsilon"]["decoy_1"]["mean_shrinks"],
                "useful_initial_epsilon": atlas["role_epsilon"]["useful"]["mean_initial"],
                "useful_final_epsilon": atlas["role_epsilon"]["useful"]["mean_final"],
                "useful_shrinks": atlas["role_epsilon"]["useful"]["mean_shrinks"],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "seed_summary.csv", rows)
    weights = [row["queries"] for row in rows]
    aggregate = {
        "runs": len(rows),
        "total_queries": int(sum(weights)),
        "top_k_failures": int(round(sum(row["failure_rate"] * row["queries"] for row in rows))),
        "failure_rate": float(np.average([row["failure_rate"] for row in rows], weights=weights)),
        "fallback_rate": float(np.average([row["fallback_rate"] for row in rows], weights=weights)),
        "top_k_target_validity": float(np.average([row["top_k_validity"] for row in rows], weights=weights)),
        "exhaustive_target_validity": float(np.average([row["exhaustive_validity"] for row in rows], weights=weights)),
        "mean_top_k_distance": float(np.mean([row["mean_top_k_distance"] for row in rows])),
        "mean_exhaustive_distance": float(np.mean([row["mean_exhaustive_distance"] for row in rows])),
        "mean_gap": float(np.mean([row["mean_gap"] for row in rows])),
        "minimum_failure_gap": float(np.min([row["minimum_failure_gap"] for row in rows])),
        "mean_distance_ratio": float(np.mean([row["mean_ratio"] for row in rows])),
        "mean_exhaustive_anchor_rank": float(np.mean([row["mean_exhaustive_rank"] for row in rows])),
        "mean_decoy_initial_epsilon": float(np.mean([row["decoy_initial_epsilon"] for row in rows])),
        "mean_decoy_final_epsilon": float(np.mean([row["decoy_final_epsilon"] for row in rows])),
        "mean_useful_initial_epsilon": float(np.mean([row["useful_initial_epsilon"] for row in rows])),
        "mean_useful_final_epsilon": float(np.mean([row["useful_final_epsilon"] for row in rows])),
    }
    (args.output / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
