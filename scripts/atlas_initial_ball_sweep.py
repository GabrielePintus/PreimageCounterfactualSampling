#!/usr/bin/env python3
"""Sweep initial-ball radii without constructing or certifying a new atlas."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.atlas_initial_ball_sweep import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/atlas_initial_ball_sweep_synthetic32.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
