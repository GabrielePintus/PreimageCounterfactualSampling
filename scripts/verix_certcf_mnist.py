#!/usr/bin/env python3
"""CLI for the paired paper-faithful VERIX/CertCF MNIST benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from experiments.verix_certcf_mnist import VeriXCertCFMNISTRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "pilot",
            "verix",
            "certcf",
            "robustness",
            "analyze",
            "status",
            "all",
        ],
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/verix_certcf_mnist.yaml",
    )
    parser.add_argument(
        "--query-indices",
        nargs="+",
        type=int,
        help="Restrict verix/certcf to prepared MNIST test indices.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Analyze only query pairs that are currently complete.",
    )
    args = parser.parse_args()
    runner = VeriXCertCFMNISTRunner.from_yaml(args.config)

    if args.stage == "prepare":
        result = runner.prepare(force=args.force)
    elif args.stage == "pilot":
        result = runner.pilot(force=args.force)
    elif args.stage == "verix":
        result = runner.run_verix(args.query_indices, force=args.force)
    elif args.stage == "certcf":
        result = runner.run_certcf(args.query_indices, force=args.force)
    elif args.stage == "robustness":
        result = runner.robustness(force=args.force)
        result = {"rows": len(result), "path": str(runner.paths.robustness)}
    elif args.stage == "analyze":
        result = runner.analyze(require_complete=not args.allow_partial)
        result = {"rows": len(result), "path": str(runner.paths.combined)}
    elif args.stage == "status":
        result = runner.status()
    else:
        result = runner.all(force=args.force)
        result = {"rows": len(result), "path": str(runner.paths.combined)}

    if isinstance(result, list):
        result = {"rows": len(result), "items": result}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
