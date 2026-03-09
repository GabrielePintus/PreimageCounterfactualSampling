"""Run a counterfactual benchmark from a YAML config file.

Usage:
    python run_experiment.py config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from counterfactuals.experiments.runner import run_from_config_path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_experiment.py <config_path>")
    config_path = sys.argv[1]
    summary = run_from_config_path(config_path)
    print("Experiment results:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
