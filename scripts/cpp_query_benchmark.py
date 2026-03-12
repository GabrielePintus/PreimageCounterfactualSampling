#!/usr/bin/env python3
"""CPP query-time benchmark suite with per-query profiling and ablations.

Usage:
    python scripts/cpp_query_benchmark.py --config configs/cpp_query_benchmark_adult.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Allow running from repo root without installing as package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactuals.experiments.runner import create_default_registries
from counterfactuals.utils.config import read_yaml
from counterfactuals.utils.seed import seed_everything


def subsample_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a subsampled train set of size n."""
    if len(x_train) <= n:
        return x_train, y_train

    if method == "kmedoids":
        try:
            from sklearn_extra.cluster import KMedoids

            km = KMedoids(n_clusters=n, random_state=int(rng.integers(0, 2**31)))
            km.fit(x_train)
            idx = km.medoid_indices_
            print(f"[INFO] K-medoids subsampling: {len(x_train)} -> {len(idx)} samples.")
            return x_train[idx], y_train[idx]
        except ImportError:
            print("[WARNING] scikit-learn-extra not installed, falling back to random subsampling.")

    idx = rng.choice(len(x_train), size=n, replace=False)
    print(f"[INFO] Random subsampling: {len(x_train)} -> {n} samples.")
    return x_train[idx], y_train[idx]


def _build_cpp_atlas(
    params: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
):
    """Build a CertifiedAtlas for Adult tabular classifier checkpoint."""
    import torch
    from torch.utils.data import TensorDataset

    from models.classifiers import TabularClassifier
    from preimage_sampling import CertifiedAtlas, NearestOppositeClassClearanceStrategy
    from training.datamodules.adult import CARDINALITIES, INPUT_TYPES
    from training.lit_classifier import LitClassifier
    from counterfactuals.models.torch_model import TorchModelWrapper

    ckpt = params.get("checkpoint")
    if not ckpt:
        raise ValueError("atlas.params.checkpoint is required")

    device = params.get("device", "cpu")
    eps_alpha = float(params.get("eps_alpha", 0.25))
    _k_per_class = params.get("k_per_class", None)
    k_per_class = int(_k_per_class) if _k_per_class is not None else None
    _norm_raw = params.get("norm", 1)
    norm = np.inf if str(_norm_raw).lower() in ("inf", "infinity") else int(_norm_raw)
    batch_size = int(params.get("batch_size", 256))
    _msc = params.get("max_samples_per_class", None)
    max_samples_per_class = int(_msc) if _msc is not None else None
    cvxpy_solver_policy = str(params.get("cvxpy_solver_policy", "auto")).lower()
    if cvxpy_solver_policy not in {"auto", "legacy"}:
        raise ValueError(
            f"atlas.params.cvxpy_solver_policy must be one of {{'auto', 'legacy'}}, got {cvxpy_solver_policy!r}"
        )

    backbone = TabularClassifier(
        input_types=INPUT_TYPES,
        cardinalities=CARDINALITIES,
        hidden_dims=[32, 8],
        num_classes=2,
        dropout=0.2,
    )
    lit = LitClassifier.load_from_checkpoint(ckpt, model=backbone, map_location=device)
    full_model = lit.model.eval().to(device)

    net_for_atlas = torch.nn.Sequential(*[m for m in full_model.net if not isinstance(m, torch.nn.Dropout)])
    atlas_model = TorchModelWrapper(model=net_for_atlas, device=device)

    z_train = x_train

    # Class-wise anchor selection
    try:
        from sklearn_extra.cluster import KMedoids

        def _select_indices(X: np.ndarray, k: int) -> np.ndarray:
            km = KMedoids(n_clusters=min(k, len(X)), metric="euclidean", method="alternate", random_state=seed)
            km.fit(X)
            return km.medoid_indices_

    except ImportError:
        def _select_indices(X: np.ndarray, k: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return rng.choice(len(X), size=min(k, len(X)), replace=False)

    z_parts, y_parts = [], []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        if k_per_class is None:
            use_idx = np.arange(len(cls_idx))
        else:
            use_idx = _select_indices(z_train[cls_idx], k_per_class)
        z_parts.append(torch.from_numpy(z_train[cls_idx[use_idx]]).float())
        y_parts.append(torch.full((len(use_idx),), int(cls), dtype=torch.long))
        print(f"[INFO] atlas class {int(cls)}: {len(cls_idx)} -> {len(use_idx)} anchors")

    medoid_ds = TensorDataset(torch.cat(z_parts), torch.cat(y_parts))

    atlas = CertifiedAtlas(
        net_for_atlas,
        medoid_ds,
        device=device,
        cvxpy_solver_policy=cvxpy_solver_policy,
    )
    eps_strategy = NearestOppositeClassClearanceStrategy(alpha=eps_alpha)
    atlas.build(
        eps_strategy=eps_strategy,
        norm=norm,
        batch_size=batch_size,
        max_samples_per_class=max_samples_per_class,
    )

    cat_slices = [(s, e) for t, (s, e) in zip(INPUT_TYPES, full_model._slices) if t == "categorical"]
    atlas.ohe_slices = cat_slices if cat_slices else None

    return atlas, atlas_model, full_model, device


def _nearest_opposite_distance(x_query: np.ndarray, y_query: int, x_train: np.ndarray, y_train: np.ndarray) -> float:
    mask = y_train != y_query
    if not np.any(mask):
        return float("nan")
    return float(np.min(np.linalg.norm(x_train[mask] - x_query[None, :], axis=1)))


def _assign_difficulty_bins(values: np.ndarray) -> List[str]:
    q1 = np.nanquantile(values, 1.0 / 3.0)
    q2 = np.nanquantile(values, 2.0 / 3.0)
    out: List[str] = []
    for v in values:
        if np.isnan(v):
            out.append("unknown")
        elif v <= q1:
            out.append("easy")
        elif v <= q2:
            out.append("medium")
        else:
            out.append("hard")
    return out


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, grp in df.groupby("variant", sort=False):
        t = grp["total_time_ms"].to_numpy(dtype=float)
        qps = grp["n_qp_solved"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": variant,
                "n_queries": int(len(grp)),
                "success_rate": float(grp["success"].mean()),
                "validity_rate": float(grp["validity"].mean()),
                "time_ms_mean": float(np.mean(t)),
                "time_ms_p50": float(np.quantile(t, 0.50)),
                "time_ms_p95": float(np.quantile(t, 0.95)),
                "time_ms_p99": float(np.quantile(t, 0.99)),
                "n_qp_mean": float(np.mean(qps)),
                "n_qp_p50": float(np.quantile(qps, 0.50)),
                "n_qp_p95": float(np.quantile(qps, 0.95)),
                "distance_mean": float(grp["distance_eval_l2"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# CPP Query Benchmark Summary",
        "",
        "| Variant | Queries | Success | Validity | Mean ms | P50 ms | P95 ms | P99 ms | Mean QP | P50 QP | P95 QP | Mean L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['n_queries'])} | {row['success_rate']:.3f} | "
            f"{row['validity_rate']:.3f} | {row['time_ms_mean']:.2f} | {row['time_ms_p50']:.2f} | "
            f"{row['time_ms_p95']:.2f} | {row['time_ms_p99']:.2f} | {row['n_qp_mean']:.2f} | "
            f"{row['n_qp_p50']:.2f} | {row['n_qp_p95']:.2f} | {row['distance_mean']:.4f} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(cfg: Dict[str, Any]) -> None:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    registries = create_default_registries()

    ds_cfg = cfg["dataset"]
    dataset = registries["dataset"].create(ds_cfg["name"], **ds_cfg.get("params", {}))
    dataset.load()
    x_train_full, y_train_full = dataset.get_train()
    x_test_full, _ = dataset.get_test()

    sampling_cfg = cfg.get("sampling", {})
    n_train = int(sampling_cfg.get("n_train", len(x_train_full)))
    n_queries = int(sampling_cfg.get("n_queries", min(256, len(x_test_full))))
    train_method = str(sampling_cfg.get("train_method", "kmedoids"))

    x_train, y_train = subsample_train(x_train_full, y_train_full, n_train, train_method, rng)

    n_queries = min(n_queries, len(x_test_full))
    query_indices = np.sort(rng.choice(len(x_test_full), size=n_queries, replace=False))
    x_queries = x_test_full[query_indices]

    atlas_cfg = cfg["atlas"]
    atlas, atlas_model, full_model, device = _build_cpp_atlas(atlas_cfg.get("params", {}), x_train, y_train, seed=seed)

    y_orig = atlas_model.predict(x_queries)
    nop_dists = np.array([
        _nearest_opposite_distance(x_queries[i], int(y_orig[i]), x_train, y_train)
        for i in range(len(x_queries))
    ])
    difficulty = _assign_difficulty_bins(nop_dists)

    rows: List[Dict[str, Any]] = []
    variants = cfg.get("variants", [])

    import torch

    for variant in variants:
        variant_name = str(variant["name"])
        method = str(variant.get("method", "sorted"))
        params = dict(variant.get("params", {}))
        print(f"[RUN] {variant_name} (method={method}, params={params})")

        for i, q_idx in enumerate(query_indices):
            xq = x_queries[i]
            yq = int(y_orig[i])
            target_class = 1 - yq

            result = atlas.find_counterfactual(
                x_query=xq,
                target_class=target_class,
                method=method,
                k=int(params.get("k", 10)),
                delta=float(params.get("delta", 0.0)),
                robust_norm=params.get("robust_norm", None),
                solver_maxiter=params.get("solver_maxiter", None),
                solver_tol=params.get("solver_tol", None),
                fixed_dims=None,
            )

            if result.success and result.x_cf is not None:
                with torch.no_grad():
                    x_cf_eval = full_model.decode(torch.from_numpy(result.x_cf[None, :]).to(device)).cpu().numpy()[0]
                y_cf = int(atlas_model.predict(x_cf_eval[None, :])[0])
                validity = float(y_cf == target_class)
                distance_eval_l2 = float(np.linalg.norm(x_cf_eval - xq, ord=2))
            else:
                y_cf = -1
                validity = 0.0
                distance_eval_l2 = float("nan")

            row: Dict[str, Any] = {
                "variant": variant_name,
                "method": method,
                "query_idx": int(q_idx),
                "difficulty": difficulty[i],
                "nearest_opposite_dist": float(nop_dists[i]),
                "target_class": int(target_class),
                "y_orig": int(yq),
                "y_cf": int(y_cf),
                "success": float(result.success),
                "validity": validity,
                "distance_eval_l2": distance_eval_l2,
                "distance_solver": float(result.distance),
                "n_qp_solved": float(result.n_qp_solved),
                "anchor_idx": int(result.anchor_idx) if result.anchor_idx is not None else -1,
            }
            for key, value in result.profiling.items():
                row[key] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    summary_df = _summarize(df)

    output_cfg = cfg.get("output", {})
    per_query_path = Path(output_cfg.get("per_query_path", "results/cpp_query_benchmark.parquet"))
    summary_path = Path(output_cfg.get("summary_path", "results/cpp_query_benchmark_summary.parquet"))
    report_md_path = Path(output_cfg.get("report_md_path", "results/cpp_query_benchmark_summary.md"))

    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(per_query_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    _write_markdown_summary(summary_df, report_md_path)

    print(f"[DONE] per-query: {per_query_path}")
    print(f"[DONE] summary : {summary_path}")
    print(f"[DONE] report  : {report_md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPP query-time benchmark suite")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    run_suite(cfg)


if __name__ == "__main__":
    main()
