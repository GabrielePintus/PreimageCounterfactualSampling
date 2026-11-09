#!/usr/bin/env python3
"""Experimental binary continuous atlas: require nonemptiness, not anchor inclusion."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.atlas_nonempty_policy import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/atlas_nonempty_synthetic32.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))
