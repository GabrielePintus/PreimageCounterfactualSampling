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
    parser.add_argument("stage", choices=["run", "analyze", "status"])
    parser.add_argument(
        "--config",
        default="configs/experiments/offline_build_parallelism.yaml",
    )
    parser.add_argument("--case")
    parser.add_argument("--variant")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runner = OfflineBuildParallelismRunner.from_yaml(args.config)
    if args.stage == "run":
        if args.case is None or args.variant is None:
            parser.error("run requires --case and --variant")
        result = runner.run(args.case, args.variant, force=args.force)
    elif args.stage == "analyze":
        frame = runner.analyze()
        result = {"rows": len(frame), "path": str(runner.paths.summary)}
    else:
        result = runner.status()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
