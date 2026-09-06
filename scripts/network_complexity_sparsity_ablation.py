#!/usr/bin/env python3
"""CLI for the full-grid CertCF query sparsity-refinement ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.network_complexity import architecture_id
from experiments.network_complexity_sparsity_ablation import (
    NetworkComplexitySparsityAblation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["benchmark", "analyze", "status", "all"])
    parser.add_argument(
        "--config",
        default="configs/experiments/network_complexity_grid_no_sparsity_parallel.yaml",
    )
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runner = NetworkComplexitySparsityAblation(args.config)
    selected = None
    if args.architectures:
        by_id = {architecture_id(*cell): cell for cell in runner.grid}
        unknown = sorted(set(args.architectures) - set(by_id))
        if unknown:
            raise SystemExit(f"Unknown architecture IDs: {unknown}")
        selected = [by_id[label] for label in args.architectures]

    if args.stage == "status":
        result = runner.status()
    elif args.stage == "benchmark":
        frame = runner.benchmark(selected, force=args.force)
        result = {"rows": len(frame), "architectures": frame["architecture_id"].nunique()}
    elif args.stage == "analyze":
        summary, comparison, report = runner.analyze()
        result = {
            "summary": str(runner.paths.summary),
            "comparison": str(runner.paths.comparison),
            "rows": len(summary),
            "report": report,
        }
    else:
        frame = runner.benchmark(force=args.force)
        summary, comparison, report = runner.analyze()
        result = {
            "query_rows": len(frame),
            "architectures": len(summary),
            "comparison": str(runner.paths.comparison),
            "report": report,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
