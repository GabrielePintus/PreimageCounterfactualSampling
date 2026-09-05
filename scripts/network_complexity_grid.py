#!/usr/bin/env python3
"""CLI for the CertCF network-complexity scaling grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.network_complexity import NetworkComplexityRunner, architecture_id


def _parse_architectures(values: list[str] | None, runner: NetworkComplexityRunner):
    if not values:
        return None
    by_id = {architecture_id(*cell): cell for cell in runner.grid}
    unknown = sorted(set(values) - set(by_id))
    if unknown:
        raise SystemExit(f"Unknown architecture IDs: {unknown}")
    return [by_id[value] for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "train", "benchmark", "analyze", "status", "all"],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/network_complexity_grid.yaml",
    )
    parser.add_argument("--architectures", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--candidate-parallelism",
        type=int,
        help="Override the number of workers used for independent top-k projections.",
    )
    parser.add_argument(
        "--candidate-parallel-backend",
        choices=["thread", "process"],
        help="Override the intra-query projection backend.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow analysis of only currently complete architecture artifacts.",
    )
    args = parser.parse_args()
    runner = NetworkComplexityRunner.from_yaml(args.config)
    runner.configure_candidate_parallelism(
        workers=args.candidate_parallelism,
        backend=args.candidate_parallel_backend,
    )
    selected = _parse_architectures(args.architectures, runner)

    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "train":
        result = runner.train(selected, force=args.force)
    elif args.stage == "benchmark":
        result = runner.benchmark(
            selected,
            force=args.force,
            calibrate_largest=selected is None,
        )
        result = {"rows": len(result), "architectures": result["architecture_id"].nunique()}
    elif args.stage == "analyze":
        result = runner.analyze(require_complete=not args.allow_partial)
        result = {"rows": len(result), "path": str(runner.paths.summary)}
    elif args.stage == "status":
        result = runner.status()
    else:
        result = runner.all(force=args.force)
        result = {"rows": len(result), "path": str(runner.paths.summary)}

    if isinstance(result, list):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
