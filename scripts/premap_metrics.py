#!/usr/bin/env python3
"""Controlled local-domain PREMAP/CertCF comparison on saved numerical datasets."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.premap_metrics import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/premap_metrics_synthetic32_alpha010.yaml")
    parser.add_argument("--premap-python", default=os.environ.get("CERTCF_PREMAP_PYTHON"))
    parser.add_argument("--premap-repository", default=os.environ.get("CERTCF_PREMAP_REPOSITORY"))
    args = parser.parse_args()
    if not args.premap_python or not args.premap_repository:
        parser.error("External PREMAP Python and repository paths are required")
    print(json.dumps(run(args.config, args.premap_python, args.premap_repository), indent=2))
