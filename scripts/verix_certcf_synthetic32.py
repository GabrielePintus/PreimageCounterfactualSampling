#!/usr/bin/env python3
"""CLI for the paired VERIX/CertCF benchmark on the synthetic 32D dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.verix_certcf_synthetic32 import Synthetic32Runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "verix", "certcf", "analyze", "status", "all"],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/verix_certcf_synthetic32.yaml",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        help="Configured IDs such as depth_01_width_016.",
    )
    parser.add_argument(
        "--query-indices",
        nargs="+",
        type=int,
        help="Restrict verix/certcf to prepared test-set indices.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    runner = Synthetic32Runner.from_yaml(args.config)
    cells = runner.resolve_architectures(args.architectures)
    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "verix":
        result = runner.run_verix(
            cells,
            args.query_indices,
            force=args.force,
        )
        result = {"completed_tasks": len(result)}
    elif args.stage == "certcf":
        result = runner.run_certcf(
            cells,
            args.query_indices,
            force=args.force,
        )
        result = {"completed_tasks": len(result)}
    elif args.stage == "analyze":
        frame = runner.analyze(allow_partial=args.allow_partial)
        result = {"rows": len(frame), "path": str(runner.paths.combined)}
    elif args.stage == "status":
        result = runner.status()
    else:
        frame = runner.all(force=args.force)
        result = {"rows": len(frame), "path": str(runner.paths.combined)}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
