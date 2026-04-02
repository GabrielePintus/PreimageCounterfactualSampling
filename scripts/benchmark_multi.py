#!/usr/bin/env python3
"""Benchmark counterfactual methods on multiple datasets, producing a single combined parquet.

Usage:
    python scripts/benchmark_multi.py --config configs/benchmarks/benchmark_meeting_all.yaml
    python scripts/benchmark_multi.py --config ... --datasets adult compas
    python scripts/benchmark_multi.py --config ... --output results/run2.parquet

Config schema:
    seed: 42
    output: results/benchmark_multi.parquet
    timeout_per_sample: 60

    methods:                         # shared across all datasets
      - name: nearest_neighbor
        run_name: nn
        params: { ... }
      - name: cpp
        run_name: cpp
        params:                      # checkpoint auto-inherited from model.params.checkpoint
          norm: 1
          ...

    datasets:
      - name: adult
        sampling: { n_queries: 50 }
        model:
          name: tabular_classifier_ckpt
          params:
            checkpoint: checkpoints/adult_classifier/best.ckpt
            device: cuda
            ...
        method_overrides:            # optional: per-dataset param overrides keyed by run_name
          face:
            epsilon: 3.0

Output:
    - One .parquet + .bmk per dataset:  <output_stem>_<dataset>.<ext>
    - One combined .parquet:            <output> (from config or --output)
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Allow running from repo root without installing as package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import run_single_dataset
from counterfactuals.utils.config import read_yaml


def _merge_methods(
    shared_methods: List[Dict[str, Any]],
    method_overrides: Dict[str, Dict[str, Any]],
    model_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build the final per-dataset method list.

    1. Deep-copy shared methods.
    2. Shallow-merge method_overrides[run_name] into each method's params.
    3. Auto-inherit ``checkpoint`` and ``device`` from model_params for cpp/my_method
       if not already set.
    """
    result = []
    for entry in shared_methods:
        m = deepcopy(entry)
        run_name = m.get("run_name", m["name"])

        # Apply explicit overrides if present.
        override = method_overrides.get(run_name) or method_overrides.get(m["name"]) or {}
        if override:
            m.setdefault("params", {}).update(override)

        # Auto-inherit checkpoint / device for cpp from the dataset's model config.
        if m["name"] in ("cpp", "my_method"):
            params = m.setdefault("params", {})
            if "checkpoint" not in params and "checkpoint" in model_params:
                params["checkpoint"] = model_params["checkpoint"]
            if "device" not in params and "device" in model_params:
                params["device"] = model_params["device"]

        result.append(m)
    return result


def _build_dataset_cfg(
    global_cfg: Dict[str, Any],
    ds_cfg: Dict[str, Any],
    global_output: Path,
) -> Dict[str, Any]:
    """Build a single-dataset config dict compatible with run_single_dataset()."""
    ds_name = ds_cfg["name"]
    model_params = ds_cfg.get("model", {}).get("params", {})
    method_overrides = ds_cfg.get("method_overrides", {})

    merged_methods = _merge_methods(
        shared_methods=global_cfg.get("methods", []),
        method_overrides=method_overrides,
        model_params=model_params,
    )

    # Per-dataset output: <stem>_<dataset>.<suffix>
    per_ds_output = global_output.parent / f"{global_output.stem}_{ds_name}{global_output.suffix}"

    return {
        "seed": int(global_cfg.get("seed", 42)),
        "dataset": {
            "name": ds_name,
            "params": ds_cfg.get("dataset_params", {"data_dir": "data/"}),
        },
        "model": ds_cfg["model"],
        "sampling": ds_cfg.get("sampling", global_cfg.get("sampling", {})),
        "preprocessing": ds_cfg.get("preprocessing", global_cfg.get("preprocessing")),
        "timeout_per_sample": ds_cfg.get(
            "timeout_per_sample", global_cfg.get("timeout_per_sample", 0)
        ),
        "output": {"path": str(per_ds_output)},
        "methods": merged_methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark counterfactual methods on multiple datasets."
    )
    parser.add_argument("--config", required=True, help="Path to multi-dataset YAML config.")
    parser.add_argument("--output", default=None, help="Override combined output parquet path.")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Run only these datasets by name (space-separated).",
    )
    args = parser.parse_args()

    global_cfg = read_yaml(args.config)
    global_output = Path(args.output or global_cfg.get("output", "results/benchmark_multi.parquet"))
    global_output.parent.mkdir(parents=True, exist_ok=True)

    datasets_cfg: List[Dict[str, Any]] = global_cfg.get("datasets", [])
    if args.datasets is not None:
        allowed = set(args.datasets)
        datasets_cfg = [d for d in datasets_cfg if d["name"] in allowed]

    if not datasets_cfg:
        print("[ERROR] No datasets to run. Check --datasets filter or config.")
        return

    all_dfs: List[pd.DataFrame] = []

    for i, ds_cfg in enumerate(datasets_cfg):
        ds_name = ds_cfg["name"]
        print(f"\n{'='*60}")
        print(f"[DATASET {i + 1}/{len(datasets_cfg)}]  {ds_name}")
        print(f"{'='*60}")

        cfg = _build_dataset_cfg(global_cfg, ds_cfg, global_output)
        result = run_single_dataset(cfg)
        df = result.to_dataframe()
        all_dfs.append(df)
        print(f"[INFO] {ds_name}: {len(df)} rows, {df['success'].mean():.1%} valid")

    if not all_dfs:
        print("[WARNING] No results produced.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(global_output, index=False, compression="gzip")
    print(f"\n[INFO] Combined results ({len(combined)} rows) saved to {global_output}")

    print("\n[SUMMARY PER DATASET]")
    summary = (
        combined.groupby(["dataset", "method"])["success"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "valid", "count": "total"})
    )
    summary["valid%"] = (summary["valid"] / summary["total"] * 100).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
