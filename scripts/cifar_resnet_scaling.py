#!/usr/bin/env python3
"""CLI for the pretrained CIFAR-10 ResNet CertCF scaling benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.cifar_resnet_scaling import CifarResNetScalingRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "pilot", "build", "query", "benchmark", "aggregate", "analyze", "status", "all"],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/cifar_resnet_scaling.yaml",
    )
    parser.add_argument("--network", choices=["resnet20", "resnet32", "resnet56"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--pilot-artifacts",
        action="store_true",
        help=(
            "For build/query/benchmark, use the 10-query pilot artifact directory "
            "instead of the full 100-query directory."
        ),
    )
    parser.add_argument(
        "--candidate-parallelism",
        type=int,
        help="Execution-only number of workers for concurrent candidate projections.",
    )
    parser.add_argument(
        "--candidate-parallel-backend",
        choices=["thread", "process"],
        help="Execution-only backend for concurrent candidate projections.",
    )
    parser.add_argument(
        "--build-parallelism",
        type=int,
        help="Execution-only number of concurrent LiRPA class workers.",
    )
    parser.add_argument(
        "--epsilon-parallelism",
        type=int,
        help="Execution-only workers for initial-radius computation.",
    )
    args = parser.parse_args()

    runner = CifarResNetScalingRunner.from_yaml(args.config)
    runner.configure_candidate_parallelism(
        workers=args.candidate_parallelism,
        backend=args.candidate_parallel_backend,
    )
    runner.configure_build_parallelism(args.build_parallelism)
    runner.configure_epsilon_parallelism(args.epsilon_parallelism)
    if args.stage in {"build", "query", "benchmark"} and args.network is None:
        parser.error(f"{args.stage} requires --network")

    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "pilot":
        frame = runner.benchmark("resnet20", force=args.force, pilot=True)
        result = {"network": "resnet20", "pilot": True, "rows": len(frame)}
    elif args.stage == "build":
        result = runner.build(args.network, force=args.force, pilot=args.pilot_artifacts)
    elif args.stage == "query":
        frame = runner.query(
            args.network,
            force=args.force,
            pilot=args.pilot_artifacts,
        )
        result = {
            "network": args.network,
            "pilot": args.pilot_artifacts,
            "rows": len(frame),
        }
    elif args.stage == "benchmark":
        frame = runner.benchmark(
            args.network,
            force=args.force,
            pilot=args.pilot_artifacts,
        )
        result = {
            "network": args.network,
            "pilot": args.pilot_artifacts,
            "rows": len(frame),
        }
    elif args.stage == "aggregate":
        frame = runner.aggregate(allow_partial=args.allow_partial)
        result = {"networks": int(frame["network"].nunique()), "rows": len(frame), "path": str(runner.paths.combined)}
    elif args.stage == "analyze":
        frame = runner.analyze(allow_partial=args.allow_partial)
        result = {"networks": len(frame), "path": str(runner.paths.summary)}
    elif args.stage == "status":
        result = runner.status()
    else:
        frame = runner.all(force=args.force)
        result = {"networks": len(frame), "path": str(runner.paths.summary)}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
