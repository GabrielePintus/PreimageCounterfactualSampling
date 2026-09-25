#!/usr/bin/env python3
"""CLI for tabular shrinkage and LiRPA-backend ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.tabular_atlas_ablation import TabularAtlasAblationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "pilot",
            "shrinkage",
            "backend",
            "aggregate-shrinkage",
            "aggregate-backend",
            "status",
        ],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/tabular_atlas_ablation.yaml",
    )
    parser.add_argument("--cases", nargs="+", help="Restrict to configured dataset IDs.")
    parser.add_argument("--alphas", nargs="+", type=float)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["without_shrinkage", "with_shrinkage"],
    )
    parser.add_argument(
        "--backend-methods",
        nargs="+",
        choices=["crown", "optimized_crown", "alpha_crown"],
    )
    parser.add_argument("--repetitions", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    runner = TabularAtlasAblationRunner.from_yaml(args.config)
    cases = runner.resolve_cases(args.cases)
    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "pilot":
        result = runner.pilot(force=args.force)
    elif args.stage == "shrinkage":
        paths = runner.run_shrinkage(
            cases,
            alphas=args.alphas,
            variants=args.variants,
            force=args.force,
        )
        result = {"runs": len(paths)}
    elif args.stage == "backend":
        paths = runner.run_backend(
            cases,
            methods=args.backend_methods,
            repetitions=args.repetitions,
            force=args.force,
        )
        result = {"runs": len(paths)}
    elif args.stage == "aggregate-shrinkage":
        regions, summary, macro = runner.aggregate_shrinkage(
            allow_partial=args.allow_partial
        )
        result = {
            "regions": len(regions),
            "per_dataset_rows": len(summary),
            "macro_rows": len(macro),
            "macro_path": str(runner.paths.shrinkage_macro),
        }
    elif args.stage == "aggregate-backend":
        summary, macro = runner.aggregate_backend(allow_partial=args.allow_partial)
        result = {
            "per_dataset_rows": len(summary),
            "macro_rows": len(macro),
            "macro_path": str(runner.paths.backend_macro),
        }
    else:
        result = runner.status()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
