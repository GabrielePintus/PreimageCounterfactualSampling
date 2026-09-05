#!/usr/bin/env python3
"""CLI for the Anchor-Ball/Anchor-PGD/CertCF LiRPA ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.lirpa_refinement_ablation import LiRPARefinementAblationRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "pilot", "build", "benchmark", "aggregate", "analyze", "status", "all"],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/lirpa_refinement_ablation.yaml",
    )
    parser.add_argument("--cases", nargs="+", help="Restrict to configured case IDs.")
    parser.add_argument(
        "--methods", nargs="+", choices=["anchor_ball", "anchor_pgd", "certcf"]
    )
    parser.add_argument("--query-positions", nargs="+", type=int)
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--skip-certification",
        action="store_true",
        help="Skip the expensive common post-hoc L1 certification during analyze/all.",
    )
    args = parser.parse_args()

    runner = LiRPARefinementAblationRunner.from_yaml(args.config)
    runner.configure_candidate_parallelism(
        workers=args.candidate_parallelism,
        backend=args.candidate_parallel_backend,
    )
    cases = runner.resolve_cases(args.cases)
    methods = runner.resolve_methods(args.methods)
    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "pilot":
        result = runner.pilot(force=args.force)
    elif args.stage == "build":
        builds = runner.build_all(cases, force=args.force)
        result = {"cases": len(builds), "builds": builds}
    elif args.stage == "benchmark":
        rows = runner.benchmark(
            cases, methods, query_positions=args.query_positions, force=args.force
        )
        result = {"rows": len(rows)}
    elif args.stage == "aggregate":
        frame = runner.aggregate(allow_partial=args.allow_partial)
        result = {"rows": len(frame), "path": str(runner.paths.combined)}
    elif args.stage == "analyze":
        frame = runner.analyze(
            allow_partial=args.allow_partial,
            skip_certification=args.skip_certification,
        )
        result = {"rows": len(frame), "path": str(runner.paths.summary)}
    elif args.stage == "status":
        result = runner.status()
    else:
        frame = runner.all(force=args.force, skip_certification=args.skip_certification)
        result = {"rows": len(frame), "path": str(runner.paths.summary)}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
