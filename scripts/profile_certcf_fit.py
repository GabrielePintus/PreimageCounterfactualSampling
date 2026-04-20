#!/usr/bin/env python3
"""Profile CertCF fit/build phases on a benchmark dataset.

This script reproduces the current benchmark-facing CertCF input-space fit path
while measuring wall time and memory separately for:

- model prediction relabeling
- optional prototype selection / clustering
- eps clearance computation
- LiRPA bound / certified polytope construction
- BVH index construction

Example:
    python scripts/profile_certcf_fit.py \
        --config configs/benchmarks/benchmark_meeting_certcf_input_vs_latent_200q.yaml \
        --dataset adult \
        --method certcf_input_kmedoids
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark import (  # noqa: E402
    _build_torch_model_from_checkpoint,
    _normalize_torch_device,
    subsample_train,
)
from certcf.atlas import CertCFAtlas  # noqa: E402
from certcf.eps_strategies import EpsStrategy, NearestOppositeClassClearanceStrategy  # noqa: E402
from certcf.indexing.bvh import BVHIndex  # noqa: E402
from counterfactuals.benchmarks import create_default_registries  # noqa: E402
from counterfactuals.methods.certcf import CertCF, _strip_dropout_modules  # noqa: E402
from counterfactuals.utils.clustering import select_prototype_indices  # noqa: E402
from counterfactuals.utils.config import read_yaml  # noqa: E402
from counterfactuals.utils.seed import seed_everything  # noqa: E402
from dataset_specs import get_tabular_dataset_spec  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal environments.
    psutil = None


class PrecomputedEpsStrategy(EpsStrategy):
    """Eps strategy that returns an already-computed array aligned to X."""

    def __init__(self, eps: np.ndarray):
        self.eps = np.asarray(eps, dtype=np.float64)

    def compute_eps(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(X) != len(self.eps):
            raise ValueError(
                f"Precomputed eps length mismatch: len(X)={len(X)}, len(eps)={len(self.eps)}"
            )
        return self.eps.copy()


def _rss_mb() -> float:
    if psutil is not None:
        return float(psutil.Process().memory_info().rss / 1024**2)

    import resource

    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


@dataclass
class PhaseRecord:
    phase: str
    wall_time_s: float
    cpu_rss_start_mb: float
    cpu_rss_end_mb: float
    cpu_rss_peak_mb: float
    cpu_rss_delta_mb: float
    cpu_rss_peak_delta_mb: float
    cuda_allocated_start_mb: float | None = None
    cuda_allocated_end_mb: float | None = None
    cuda_allocated_peak_mb: float | None = None
    cuda_reserved_start_mb: float | None = None
    cuda_reserved_end_mb: float | None = None
    cuda_reserved_peak_mb: float | None = None


@contextmanager
def profile_phase(name: str, records: list[PhaseRecord], device: torch.device, sample_interval_s: float):
    """Measure wall time, process RSS peak, and CUDA peak stats for one phase."""
    gc.collect()
    cuda_enabled = device.type == "cuda" and torch.cuda.is_available()
    if cuda_enabled:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_alloc_start = float(torch.cuda.memory_allocated(device) / 1024**2)
        cuda_reserved_start = float(torch.cuda.memory_reserved(device) / 1024**2)
    else:
        cuda_alloc_start = None
        cuda_reserved_start = None

    stop = threading.Event()
    rss_start = _rss_mb()
    rss_peak = rss_start

    def _sample() -> None:
        nonlocal rss_peak
        while not stop.is_set():
            rss_peak = max(rss_peak, _rss_mb())
            stop.wait(sample_interval_s)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
        sampler.join(timeout=max(0.1, sample_interval_s * 4))
        rss_end = _rss_mb()
        rss_peak = max(rss_peak, rss_end)

        if cuda_enabled:
            torch.cuda.synchronize(device)
            cuda_alloc_end = float(torch.cuda.memory_allocated(device) / 1024**2)
            cuda_reserved_end = float(torch.cuda.memory_reserved(device) / 1024**2)
            cuda_alloc_peak = float(torch.cuda.max_memory_allocated(device) / 1024**2)
            cuda_reserved_peak = float(torch.cuda.max_memory_reserved(device) / 1024**2)
        else:
            cuda_alloc_end = None
            cuda_reserved_end = None
            cuda_alloc_peak = None
            cuda_reserved_peak = None

        records.append(
            PhaseRecord(
                phase=name,
                wall_time_s=elapsed,
                cpu_rss_start_mb=rss_start,
                cpu_rss_end_mb=rss_end,
                cpu_rss_peak_mb=rss_peak,
                cpu_rss_delta_mb=rss_end - rss_start,
                cpu_rss_peak_delta_mb=rss_peak - rss_start,
                cuda_allocated_start_mb=cuda_alloc_start,
                cuda_allocated_end_mb=cuda_alloc_end,
                cuda_allocated_peak_mb=cuda_alloc_peak,
                cuda_reserved_start_mb=cuda_reserved_start,
                cuda_reserved_end_mb=cuda_reserved_end,
                cuda_reserved_peak_mb=cuda_reserved_peak,
            )
        )


def _find_dataset_cfg(cfg: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    if "datasets" not in cfg:
        ds_cfg = dict(cfg["dataset"])
        if ds_cfg["name"] != dataset_name:
            raise ValueError(f"Single-dataset config is for {ds_cfg['name']!r}, not {dataset_name!r}")
        return {
            "name": ds_cfg["name"],
            "dataset_params": ds_cfg.get("params", {"data_dir": "data/"}),
            "model": cfg["model"],
            "sampling": cfg.get("sampling", {}),
        }

    for ds_cfg in cfg["datasets"]:
        if ds_cfg["name"] == dataset_name:
            return ds_cfg
    available = ", ".join(ds["name"] for ds in cfg["datasets"])
    raise ValueError(f"Dataset {dataset_name!r} not found. Available: {available}")


def _find_method_cfg(cfg: dict[str, Any], method_run_name: str) -> dict[str, Any]:
    for method_cfg in cfg.get("methods", []):
        if method_cfg.get("run_name", method_cfg["name"]) == method_run_name:
            return method_cfg
    available = ", ".join(method.get("run_name", method["name"]) for method in cfg.get("methods", []))
    raise ValueError(f"Method {method_run_name!r} not found. Available: {available}")


def _resolve_data_dir(params: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(params)
    if "data_dir" in resolved:
        data_dir = Path(resolved["data_dir"])
        if not data_dir.is_absolute():
            resolved["data_dir"] = str(ROOT / data_dir)
    return resolved


def _parse_norm(value: Any) -> int | float:
    return np.inf if str(value).lower() in {"inf", "infinity"} else int(value)


def _records_as_dicts(records: list[PhaseRecord]) -> list[dict[str, Any]]:
    return [record.__dict__ for record in records]


def _print_phase_table(records: list[PhaseRecord]) -> None:
    headers = [
        "phase",
        "time_s",
        "rss_start",
        "rss_end",
        "rss_peak",
        "rss_peak_delta",
        "cuda_alloc_peak",
        "cuda_reserved_peak",
    ]
    rows = []
    for r in records:
        rows.append(
            [
                r.phase,
                f"{r.wall_time_s:.3f}",
                f"{r.cpu_rss_start_mb:.1f}",
                f"{r.cpu_rss_end_mb:.1f}",
                f"{r.cpu_rss_peak_mb:.1f}",
                f"{r.cpu_rss_peak_delta_mb:.1f}",
                "NA" if r.cuda_allocated_peak_mb is None else f"{r.cuda_allocated_peak_mb:.1f}",
                "NA" if r.cuda_reserved_peak_mb is None else f"{r.cuda_reserved_peak_mb:.1f}",
            ]
        )

    widths = [max(len(str(x)) for x in [h, *[row[i] for row in rows]]) for i, h in enumerate(headers)]
    print("\nPHASE PROFILE")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(x).ljust(widths[i]) for i, x in enumerate(row)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/benchmark_meeting_certcf_input_vs_latent_200q.yaml",
        help="Benchmark YAML containing the Adult CertCF method config.",
    )
    parser.add_argument("--dataset", default="adult", help="Dataset to profile. Default: adult.")
    parser.add_argument(
        "--method",
        default="certcf_input_kmedoids",
        help="CertCF run_name to profile. Default: certcf_input_kmedoids.",
    )
    parser.add_argument("--device", default=None, help="Override device from config, e.g. cpu, cuda, gpu:0.")
    parser.add_argument("--k-per-class", type=int, default=None, help="Override method k_per_class.")
    parser.add_argument(
        "--subsample-method",
        default=None,
        help="Override atlas_subsample_method/subsample_method.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path. Default: results/profile_certcf_fit_<dataset>_<method>.json",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=20.0,
        help="CPU RSS sampling interval during each phase.",
    )
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    ds_cfg = _find_dataset_cfg(cfg, args.dataset)
    method_cfg = _find_method_cfg(cfg, args.method)
    method_params = dict(method_cfg.get("params") or {})
    model_params = dict(ds_cfg.get("model", {}).get("params", {}))

    if args.k_per_class is not None:
        method_params["k_per_class"] = args.k_per_class
    if args.subsample_method is not None:
        method_params["atlas_subsample_method"] = args.subsample_method
    if "checkpoint" not in method_params and "checkpoint" in model_params:
        method_params["checkpoint"] = model_params["checkpoint"]
    if args.device is not None:
        method_params["device"] = args.device
    elif "device" not in method_params and "device" in model_params:
        method_params["device"] = model_params["device"]

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    requested_device = method_params.get("device", "cpu")
    device_str = _normalize_torch_device(requested_device)
    device = torch.device(device_str)
    if str(requested_device).strip().lower() != device_str:
        print(f"[INFO] device normalized: {requested_device!r} -> {device_str!r}")

    registries = create_default_registries()
    dataset_params = _resolve_data_dir(ds_cfg.get("dataset_params", {"data_dir": "data/"}))
    dataset = registries["dataset"].create(ds_cfg["name"], **dataset_params)
    dataset.load()
    x_train_full, y_train_full = dataset.get_train()

    sampling_cfg = ds_cfg.get("sampling", cfg.get("sampling", {}))
    n_train = int(sampling_cfg.get("n_train", len(x_train_full)))
    x_train, y_train = subsample_train(x_train_full, y_train_full, n_train, "random", rng)

    dataset_module = str(model_params.get("dataset_module", ds_cfg["name"]))
    hidden_dims = list(model_params.get("hidden_dims", [32, 8]))
    dropout = float(model_params.get("dropout", 0.2))
    checkpoint = method_params.get("checkpoint")
    if checkpoint is None:
        raise ValueError("CertCF fit profiling requires a checkpoint path.")

    model_wrapper = _build_torch_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device_str,
        dataset_module=dataset_module,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    spec = get_tabular_dataset_spec(dataset_module)
    cat_slices = list(spec.categorical_slices) or None

    eps_alpha = float(method_params.get("eps_alpha", 0.25))
    k_per_class_raw = method_params.get("k_per_class", 500)
    k_per_class = None if k_per_class_raw is None else int(k_per_class_raw)
    norm = _parse_norm(method_params.get("norm", 2))
    distance_norm = (
        norm
        if method_params.get("distance_norm") is None
        else _parse_norm(method_params.get("distance_norm"))
    )
    lirpa_method = str(method_params.get("lirpa_method", "backward"))
    batch_size = int(method_params.get("batch_size", 256))
    query_method = str(method_params.get("query_method", "sorted")).lower()
    query_k_candidates = int(method_params.get("query_k_candidates", 1))
    solver_maxiter = int(method_params.get("solver_maxiter", 500))
    boundary_beta = float(method_params.get("atlas_boundary_beta", method_params.get("boundary_beta", 0.5)))
    subsample_method = str(
        method_params.get("atlas_subsample_method", method_params.get("subsample_method", "kmedoids"))
    ).lower()
    subsample_space = str(method_params.get("atlas_subsample_space", "input")).lower()

    profiler_method = CertCF(
        model=model_wrapper,
        norm=norm,
        distance_norm=distance_norm,
        lirpa_method=lirpa_method,
        eps_strategy=NearestOppositeClassClearanceStrategy(alpha=eps_alpha),
        batch_size=batch_size,
        ohe_slices=cat_slices,
        cnn=False,
        default_query_method=query_method,
        query_k_candidates=query_k_candidates,
        solver_maxiter=solver_maxiter,
        k_per_class=k_per_class,
        subsample_method=subsample_method,
        subsample_space=subsample_space,
        boundary_beta=boundary_beta,
        random_seed=seed,
    )

    records: list[PhaseRecord] = []
    sample_interval_s = max(0.001, args.sample_interval_ms / 1000.0)

    print(f"[INFO] Dataset: {ds_cfg['name']} train={len(x_train)} dims={x_train.shape[1]}")
    print(f"[INFO] Method: {args.method} subsample_space={subsample_space} method={subsample_method} k_per_class={k_per_class}")
    print(f"[INFO] Device: {device}")

    with profile_phase("predict_training_labels", records, device, sample_interval_s):
        y_pred_full = profiler_method._predict_training_labels(x_train)

    predicted_classes = np.unique(y_pred_full)
    if predicted_classes.size < 2:
        raise ValueError("Predicted training support collapsed to fewer than 2 classes.")

    selection_data = x_train
    if subsample_space == "latent":
        with profile_phase("extract_latent_embeddings", records, device, sample_interval_s):
            selection_data = profiler_method._extract_penultimate_embeddings(x_train)

    class_selection: list[dict[str, Any]] = []
    with profile_phase("subsample_or_cluster", records, device, sample_interval_s):
        if k_per_class is None:
            x_selected = x_train.copy()
            y_selected = y_pred_full.copy()
            original_indices = np.arange(len(x_train), dtype=np.int64)
            for cls in predicted_classes:
                idx = np.where(y_pred_full == cls)[0]
                class_selection.append(
                    {
                        "class": int(cls),
                        "n_available": int(len(idx)),
                        "n_selected": int(len(idx)),
                    }
                )
        else:
            parts_x: list[np.ndarray] = []
            parts_y: list[np.ndarray] = []
            parts_indices: list[np.ndarray] = []
            boundary_scores_full = None
            if subsample_method == "boundary_random":
                boundary_scores_full = profiler_method._predict_training_boundary_scores(x_train)
            sampling_rng = np.random.default_rng(seed)
            for cls in predicted_classes:
                idx = np.where(y_pred_full == cls)[0]
                if subsample_method == "boundary_random":
                    if boundary_scores_full is None:
                        raise RuntimeError("Internal error: boundary scores were not computed for boundary_random sampling.")
                    proto = profiler_method._select_boundary_weighted_indices(
                        boundary_scores_full[idx],
                        k_per_class,
                        rng=sampling_rng,
                    )
                else:
                    proto = select_prototype_indices(
                        selection_data[idx],
                        k_per_class,
                        method=subsample_method,
                        random_state=seed,
                    )
                selected_idx = idx[proto]
                parts_x.append(x_train[selected_idx])
                parts_y.append(y_pred_full[selected_idx])
                parts_indices.append(selected_idx.astype(np.int64, copy=False))
                class_selection.append(
                    {
                        "class": int(cls),
                        "n_available": int(len(idx)),
                        "n_selected": int(len(selected_idx)),
                    }
                )
            x_selected = np.concatenate(parts_x)
            y_selected = np.concatenate(parts_y)
            original_indices = np.concatenate(parts_indices)

    eps_strategy = NearestOppositeClassClearanceStrategy(alpha=eps_alpha)
    with profile_phase("compute_eps_clearance", records, device, sample_interval_s):
        eps_array = eps_strategy.compute_eps(x_selected, y_selected)

    module, resolved_device = profiler_method._resolve_torch_module_and_device()
    clean_module = _strip_dropout_modules(module)
    tensor_dataset = TensorDataset(
        torch.from_numpy(x_selected).float(),
        torch.from_numpy(y_selected).long(),
    )
    atlas = CertCFAtlas(
        clean_module,
        tensor_dataset,
        resolved_device,
        cnn=False,
        norm=norm,
        distance_norm=distance_norm,
        lirpa_method=lirpa_method,
        eps_strategy=PrecomputedEpsStrategy(eps_array),
        batch_size=batch_size,
        ohe_slices=cat_slices,
        default_query_method=query_method,
        solver_maxiter=solver_maxiter,
    )
    atlas.eps_strategy = PrecomputedEpsStrategy(eps_array)

    with profile_phase("lirpa_bounds_polytopes", records, device, sample_interval_s):
        atlas.bounds = atlas._preimage.compute_all_bounds(
            eps=0.1,
            norm=atlas.norm,
            batch_size=atlas.batch_size,
            dtype=torch.float32,
            eps_array=eps_array,
            lirpa_method=atlas.lirpa_method,
        )

    with profile_phase("build_bvh_indices", records, device, sample_interval_s):
        atlas.bvh_indices = {}
        for label in atlas.class_labels:
            centers = atlas.bounds[label]["X"]
            eps_class = atlas.bounds[label]["eps"]
            atlas.bvh_indices[label] = BVHIndex(centers, eps_class)

    phase_dicts = _records_as_dicts(records)
    summary = {
        "dataset": ds_cfg["name"],
        "method": args.method,
        "device": str(device),
        "n_train_input": int(len(x_train)),
        "n_selected": int(len(x_selected)),
        "n_features": int(x_selected.shape[1]),
        "predicted_class_counts_full": {
            str(int(cls)): int(np.sum(y_pred_full == cls)) for cls in predicted_classes
        },
        "selected_class_counts": {
            str(int(cls)): int(np.sum(y_selected == cls)) for cls in np.unique(y_selected)
        },
        "class_selection": class_selection,
        "eps_min": float(np.min(eps_array)),
        "eps_median": float(np.median(eps_array)),
        "eps_mean": float(np.mean(eps_array)),
        "eps_max": float(np.max(eps_array)),
        "total_profiled_time_s": float(sum(record.wall_time_s for record in records)),
        "max_cpu_rss_peak_mb": float(max(record.cpu_rss_peak_mb for record in records)),
        "max_cuda_allocated_peak_mb": (
            None
            if all(record.cuda_allocated_peak_mb is None for record in records)
            else float(max(record.cuda_allocated_peak_mb or 0.0 for record in records))
        ),
        "max_cuda_reserved_peak_mb": (
            None
            if all(record.cuda_reserved_peak_mb is None for record in records)
            else float(max(record.cuda_reserved_peak_mb or 0.0 for record in records))
        ),
    }
    output = {
        "config_path": str(Path(args.config).resolve()),
        "method_params": method_params,
        "model_params": model_params,
        "summary": summary,
        "phases": phase_dicts,
        "selected_original_indices": original_indices.tolist(),
    }

    _print_phase_table(records)
    print("\nSUMMARY")
    for key, value in summary.items():
        if key == "class_selection":
            continue
        print(f"{key}: {value}")
    print("class_selection:", class_selection)

    output_path = Path(
        args.output
        or ROOT / "results" / f"profile_certcf_fit_{ds_cfg['name']}_{args.method}.json"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n[INFO] Wrote profile JSON to {output_path}")


if __name__ == "__main__":
    main()
