#!/usr/bin/env python3
"""Run a bounded PREMAP pilot on the existing synthetic-32 artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.atlas_premap import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/atlas_premap_synthetic32.yaml")
    parser.add_argument("--premap-python", default=os.environ.get("CERTCF_PREMAP_PYTHON"))
    parser.add_argument("--premap-repository", default=os.environ.get("CERTCF_PREMAP_REPOSITORY"))
    parser.add_argument("--diagnostics", action="store_true", help="Inspect saved global boxes and run small local-box positive controls.")
    args = parser.parse_args()
    if not args.premap_python or not args.premap_repository:
        parser.error("Specify the external PREMAP Python and repository paths (flags or CERTCF_PREMAP_* environment variables)")
    if args.diagnostics:
        from experiments.atlas_premap_diagnostics import run_diagnostics
        output = run_diagnostics(args.config, args.premap_python, args.premap_repository)
    else:
        output = run(args.config, args.premap_python, args.premap_repository)
    print(json.dumps(output, indent=2))
