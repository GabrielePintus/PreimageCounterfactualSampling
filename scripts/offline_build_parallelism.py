#!/usr/bin/env python3
"""Benchmark serial and parallel CertCF atlas construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.offline_build_parallelism import OfflineBuildParallelismRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["run", "epsilon-sweep", "analyze", "status"],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/offline_build_parallelism.yaml",
    )
    parser.add_argument("--case")
    parser.add_argument("--variant")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers",
        default="1,2,4,8,12,16",
        help="Comma-separated worker counts for epsilon-sweep.",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    runner = OfflineBuildParallelismRunner.from_yaml(args.config)
    if args.stage == "run":
        if args.case is None or args.variant is None:
            parser.error("run requires --case and --variant")
        result = runner.run(args.case, args.variant, force=args.force)
    elif args.stage == "epsilon-sweep":
        frame = runner.run_epsilon_sweep(
            case_id=args.case or "cifar_resnet20",
            worker_counts=[int(value) for value in args.workers.split(",")],
            repetitions=args.repetitions,
            force=args.force,
        )
        result = {"rows": len(frame), "path": str(runner.paths.epsilon_sweep)}
    elif args.stage == "analyze":
        frame = runner.analyze()
        result = {"rows": len(frame), "path": str(runner.paths.summary)}
    else:
        result = runner.status()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
