#!/usr/bin/env python3
"""CLI for top-k versus exhaustive CertCF search on tabular datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.tabular_topk_ablation import TabularTopKAblationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "pilot", "build", "benchmark", "aggregate", "analyze", "status", "all"],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/tabular_topk_ablation.yaml",
    )
    parser.add_argument("--cases", nargs="+", help="Restrict to configured dataset IDs.")
    parser.add_argument("--query-positions", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    runner = TabularTopKAblationRunner.from_yaml(args.config)
    cases = runner.resolve_cases(args.cases)
    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "pilot":
        result = runner.pilot(force=args.force)
    elif args.stage == "build":
        builds = runner.build(cases, force=args.force)
        result = {"cases": len(builds), "builds": builds}
    elif args.stage == "benchmark":
        frame = runner.benchmark(
            cases,
            query_positions=args.query_positions,
            force=args.force,
        )
        result = {"rows": len(frame)}
    elif args.stage == "aggregate":
        frame = runner.aggregate(allow_partial=args.allow_partial)
        result = {"rows": len(frame), "path": str(runner.paths.combined)}
    elif args.stage == "analyze":
        per_dataset, macro = runner.analyze(allow_partial=args.allow_partial)
        result = {
            "per_dataset_rows": len(per_dataset),
            "macro_rows": len(macro),
            "per_dataset_path": str(runner.paths.per_dataset_summary),
            "macro_path": str(runner.paths.macro_summary),
        }
    elif args.stage == "status":
        result = runner.status()
    else:
        per_dataset, macro = runner.all(force=args.force)
        result = {"per_dataset_rows": len(per_dataset), "macro_rows": len(macro)}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
