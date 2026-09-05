#!/usr/bin/env python3
"""CLI for the CertCF nearest-anchor top-k heuristic ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.topk_heuristic_ablation import TopKHeuristicAblationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "build",
            "benchmark",
            "parallel-benchmark",
            "parallel-analyze",
            "aggregate",
            "analyze",
            "status",
            "all",
        ],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/topk_heuristic_ablation.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the selected stage instead of reusing valid artifacts.",
    )
    parser.add_argument(
        "--parallel-k-values",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated k values for the isolated parallel latency benchmark.",
    )
    parser.add_argument(
        "--candidate-workers",
        default="1,2,4,8",
        help="Comma-separated candidate worker counts; the serial value 1 is required.",
    )
    parser.add_argument(
        "--parallel-queries",
        type=int,
        default=None,
        help="Optional prefix of the fixed queries used by the parallel latency benchmark.",
    )
    parser.add_argument(
        "--candidate-backend",
        choices=["thread", "process"],
        default="process",
        help="Backend used for projections from the same query.",
    )
    args = parser.parse_args()

    runner = TopKHeuristicAblationRunner.from_yaml(args.config)
    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "build":
        result = runner.build(force=args.force)
    elif args.stage == "benchmark":
        frame = runner.benchmark(force=args.force)
        result = {
            "queries": int(frame["query_idx"].nunique()),
            "k_values": sorted(frame["k"].astype(int).unique().tolist()),
            "rows": int(len(frame)),
        }
    elif args.stage == "parallel-benchmark":
        k_values = [int(value) for value in args.parallel_k_values.split(",") if value.strip()]
        worker_values = [int(value) for value in args.candidate_workers.split(",") if value.strip()]
        frame = runner.benchmark_parallelism(
            k_values=k_values,
            worker_values=worker_values,
            n_queries=args.parallel_queries,
            backend=args.candidate_backend,
            force=args.force,
        )
        result = {
            "queries": int(frame["query_idx"].nunique()),
            "k_values": sorted(frame["k"].astype(int).unique().tolist()),
            "candidate_workers": sorted(
                frame["candidate_workers_requested"].astype(int).unique().tolist()
            ),
            "rows": int(len(frame)),
            "path": str(runner.paths.parallel_queries),
        }
    elif args.stage == "parallel-analyze":
        frame = runner.analyze_parallelism()
        result = {
            "rows": int(len(frame)),
            "path": str(runner.paths.parallel_summary),
        }
    elif args.stage == "aggregate":
        frame = runner.aggregate()
        result = {
            "queries": int(frame["query_idx"].nunique()),
            "rows": int(len(frame)),
            "path": str(runner.paths.combined_queries),
        }
    elif args.stage == "analyze":
        frame = runner.analyze()
        result = {"rows": int(len(frame)), "path": str(runner.paths.summary)}
    elif args.stage == "status":
        result = runner.status()
    else:
        frame = runner.all(force=args.force)
        result = {"rows": int(len(frame)), "path": str(runner.paths.summary)}

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
