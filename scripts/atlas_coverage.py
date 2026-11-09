#!/usr/bin/env python3
"""Measure held-out coverage using an existing certified atlas."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.atlas_coverage import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/atlas_coverage_synthetic32.yaml")
    parser.add_argument("--force", action="store_true", help="Recompute coverage cache, never rebuild the source atlas.")
    args = parser.parse_args()
    print(json.dumps(run(args.config, force=args.force), indent=2))


if __name__ == "__main__":
    main()
