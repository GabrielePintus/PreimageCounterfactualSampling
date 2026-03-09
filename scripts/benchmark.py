#!/usr/bin/env python3
"""Benchmark all counterfactual methods on a single dataset.

Usage:
    python scripts/benchmark.py --config configs/benchmark_adult.yaml
    python scripts/benchmark.py --config configs/benchmark_adult.yaml --output results/run2.parquet
    python scripts/benchmark.py --config configs/benchmark_adult.yaml --seed 123 --n_queries 200
    python scripts/benchmark.py --config configs/benchmark_adult.yaml --methods dice face nearest_neighbor

Output:
    A Parquet file with one row per (method, query) containing raw (x_orig, x_cf)
    vectors plus immediate metrics (l2, l1, l0_sparsity). Failed attempts are
    logged as rows with success=False and NaN feature columns.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from repo root without installing as package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactuals.core.base_classes import CounterfactualExample
from counterfactuals.experiments.runner import create_default_registries
from counterfactuals.utils.config import read_yaml
from counterfactuals.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Timeout helpers (SIGALRM — Linux/macOS only)
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("generate() exceeded timeout")


def _call_with_timeout(fn, timeout_s: int):
    """Call fn() with a hard wall-clock timeout via SIGALRM.

    If timeout_s <= 0 the call is made without any timeout.
    Raises TimeoutError if the call exceeds the limit.
    """
    if timeout_s <= 0:
        return fn()
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Train subsampling
# ---------------------------------------------------------------------------

def subsample_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a subsampled train set of size n.

    method='kmedoids' uses sklearn_extra.cluster.KMedoids (falls back to random
    if the package is not installed).  method='random' always uses random sampling.
    If len(x_train) <= n, the full set is returned unchanged.
    """
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
            print(
                "[WARNING] scikit-learn-extra not installed. "
                "Falling back to random subsampling for the train set."
            )

    idx = rng.choice(len(x_train), size=n, replace=False)
    print(f"[INFO] Random subsampling: {len(x_train)} -> {n} samples.")
    return x_train[idx], y_train[idx]


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _failure_row(
    method_name: str,
    query_idx: int,
    x_orig: np.ndarray,
    y_orig: int,
    n_features: int,
    error: str,
    runtime_s: float,
    space: str = "raw",
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": method_name,
        "query_idx": int(query_idx),
        "space": space,
        "y_orig": int(y_orig),
        "y_cf": float("nan"),
        "success": False,
        "runtime_s": float(runtime_s),
        "error": error,
        "l2_distance": float("nan"),
        "l1_distance": float("nan"),
        "l0_sparsity": float("nan"),
    }
    for k in range(n_features):
        row[f"x_orig_{k}"] = float(x_orig[k])
        row[f"x_cf_{k}"] = float("nan")
    return row


def _success_row(
    method_name: str,
    query_idx: int,
    x_orig: np.ndarray,
    y_orig: int,
    x_cf: np.ndarray,
    y_cf: int,
    success: bool,
    runtime_s: float,
    n_features: int,
    space: str = "raw",
) -> Dict[str, Any]:
    diff = np.abs(x_cf.astype(np.float64) - x_orig.astype(np.float64))
    row: Dict[str, Any] = {
        "method": method_name,
        "query_idx": int(query_idx),
        "space": space,
        "y_orig": int(y_orig),
        "y_cf": int(y_cf),
        "success": bool(success),
        "runtime_s": float(runtime_s),
        "error": None,
        "l2_distance": float(np.linalg.norm(diff, ord=2)),
        "l1_distance": float(np.linalg.norm(diff, ord=1)),
        "l0_sparsity": float(np.mean(diff > 1e-6)),
    }
    for k in range(n_features):
        row[f"x_orig_{k}"] = float(x_orig[k])
        row[f"x_cf_{k}"] = float(x_cf[k])
    return row


# ---------------------------------------------------------------------------
# CertifiedAtlas builder
# ---------------------------------------------------------------------------

def _build_certified_atlas_method(
    params: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_queries: np.ndarray,
    seed: int,
):
    """Build CertifiedAtlasMethod from a checkpoint.

    Returns (atlas_method, atlas_model, embed_model, z_train, z_queries).

    ``generate()`` calls should pass z_queries and atlas_model (embedding-space
    model).  The returned ``embed_model`` (full TabularClassifier) exposes a
    ``decode()`` method to map embedding-space CFs back to raw feature space.
    """
    import torch
    from torch.utils.data import TensorDataset

    from models.classifiers import TabularClassifier
    from preimage_sampling import CertifiedAtlas, NearestOppositeClassClearanceStrategy
    from training.datamodules.adult import CARDINALITIES, INPUT_TYPES
    from training.lit_classifier import LitClassifier

    from counterfactuals.methods.my_method import CertifiedAtlasMethod
    from counterfactuals.models.torch_model import TorchModelWrapper

    ckpt = params.get("checkpoint")
    if not ckpt:
        raise ValueError("my_method requires a 'checkpoint' param pointing to the .ckpt file.")

    device = params.get("device", "cpu")
    eps_alpha = float(params.get("eps_alpha", 0.25))
    k_per_class = int(params.get("k_per_class", 500))
    norm = int(params.get("norm", 1))
    batch_size = int(params.get("batch_size", 256))

    # Load backbone + Lightning checkpoint.
    backbone = TabularClassifier(
        input_types=INPUT_TYPES,
        cardinalities=CARDINALITIES,
        embedding_dim=1,
        hidden_dims=[64, 32],
        num_classes=2,
    )
    lit = LitClassifier.load_from_checkpoint(ckpt, model=backbone, map_location=device)
    model = lit.model.eval().to(device)
    atlas_model = TorchModelWrapper(model=model.net, device=device)

    # Embed train and query points.
    with torch.no_grad():
        z_train = model.embed(torch.from_numpy(x_train).to(device)).cpu().numpy().astype(np.float32)
        z_queries = model.embed(torch.from_numpy(x_queries).to(device)).cpu().numpy().astype(np.float32)

    # Class-wise k-medoids on embedded train points (mirrors the notebook setup).
    try:
        from sklearn_extra.cluster import KMedoids
        def _medoid_indices(X: np.ndarray, k: int) -> np.ndarray:
            km = KMedoids(n_clusters=min(k, len(X)), metric="euclidean",
                          method="alternate", random_state=seed)
            km.fit(X)
            return km.medoid_indices_
    except ImportError:
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances as _pw
        def _medoid_indices(X: np.ndarray, k: int) -> np.ndarray:
            km = KMeans(n_clusters=min(k, len(X)), random_state=seed, n_init="auto")
            km.fit(X)
            return _pw(km.cluster_centers_, X).argmin(axis=1)

    z_parts, y_parts = [], []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        med_idx = _medoid_indices(z_train[cls_idx], k_per_class)
        z_parts.append(torch.from_numpy(z_train[cls_idx[med_idx]]).float())
        y_parts.append(torch.full((len(med_idx),), int(cls), dtype=torch.long))
        print(f"  [my_method] class {int(cls)}: {len(cls_idx)} -> {len(med_idx)} medoids")

    medoid_ds = TensorDataset(torch.cat(z_parts), torch.cat(y_parts))

    # Build atlas on the embedding head.
    eps_strategy = NearestOppositeClassClearanceStrategy(alpha=eps_alpha)
    atlas = CertifiedAtlas(model.net, medoid_ds, device=device)
    atlas.build(eps_strategy=eps_strategy, norm=norm, batch_size=batch_size)
    print(atlas.summary())

    atlas_method = CertifiedAtlasMethod(atlas=atlas, random_seed=seed)
    atlas_method.fit(x_train=z_train, y_train=y_train, model=atlas_model)

    return atlas_method, atlas_model, model, z_train, z_queries


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 85)
    print("BENCHMARK SUMMARY")
    print("=" * 85)
    fmt = f"{'method':<22} {'validity%':>10} {'l2_mean':>9} {'l1_mean':>9} {'sparsity%':>10} {'runtime_s':>10} {'n_failed':>9}"
    print(fmt)
    print("-" * 85)
    for method_name, grp in df.groupby("method", sort=False):
        n_total = len(grp)
        n_ok = int(grp["success"].sum())
        n_failed = n_total - n_ok
        ok = grp[grp["success"]]
        validity = 100.0 * n_ok / n_total if n_total > 0 else float("nan")
        l2 = float(ok["l2_distance"].mean()) if n_ok > 0 else float("nan")
        l1 = float(ok["l1_distance"].mean()) if n_ok > 0 else float("nan")
        sp = 100.0 * float(ok["l0_sparsity"].mean()) if n_ok > 0 else float("nan")
        rt = float(grp["runtime_s"].mean())
        print(
            f"{method_name:<22} {validity:>9.1f}% {l2:>9.3f} {l1:>9.3f}"
            f" {sp:>9.1f}% {rt:>10.3f} {n_failed:>9d}"
        )
    print("=" * 85)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides on top of the YAML config dict (mutates cfg)."""
    if args.output is not None:
        cfg.setdefault("output", {})["path"] = args.output
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_queries is not None:
        cfg.setdefault("sampling", {})["n_queries"] = args.n_queries
    if args.methods is not None:
        allowed = set(args.methods)
        cfg["methods"] = [m for m in cfg.get("methods", []) if m["name"] in allowed]
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark counterfactual methods on a single dataset."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", default=None, help="Override output Parquet path.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--n_queries", type=int, default=None, help="Override number of test queries.")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Run only these methods (space-separated names).",
    )
    args = parser.parse_args()

    cfg = _apply_cli_overrides(read_yaml(args.config), args)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    # --- Dataset ---
    registries = create_default_registries()
    ds_cfg = cfg["dataset"]
    dataset = registries["dataset"].create(ds_cfg["name"], **ds_cfg.get("params", {}))
    dataset.load()
    x_train_full, y_train_full = dataset.get_train()
    x_test_full, _ = dataset.get_test()
    n_features = x_train_full.shape[1]

    # --- Train subsampling (once, shared by all methods) ---
    sampling_cfg = cfg.get("sampling", {})
    n_train = int(sampling_cfg.get("n_train", len(x_train_full)))
    n_queries = int(sampling_cfg.get("n_queries", len(x_test_full)))
    train_method = str(sampling_cfg.get("train_method", "random"))

    x_train, y_train = subsample_train(x_train_full, y_train_full, n_train, train_method, rng)

    # --- Test query selection (fixed set, shared by all methods) ---
    n_queries = min(n_queries, len(x_test_full))
    query_indices = np.sort(rng.choice(len(x_test_full), size=n_queries, replace=False))
    x_queries = x_test_full[query_indices]
    print(f"[INFO] {n_queries} test queries selected from {len(x_test_full)} test samples.")

    # --- Model ---
    model_cfg = cfg["model"]
    model = registries["model"].create(model_cfg["name"], **model_cfg.get("params", {}))
    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "fit"):
        print(f"[INFO] Fitting model ({model_cfg['name']}) on {len(x_train)} train samples...")
        estimator.fit(x_train, y_train)
        acc = float(np.mean(model.predict(x_train) == y_train))
        print(f"[INFO] Train accuracy: {acc:.3f}")

    # Pre-compute original class predictions for all queries (shared across methods).
    y_orig_all = model.predict(x_queries)

    timeout_s = int(cfg.get("timeout_per_sample", 0))
    output_path = Path(cfg.get("output", {}).get("path", "results/benchmark.parquet"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods_cfg = cfg.get("methods", [])
    all_rows: List[Dict[str, Any]] = []

    for method_cfg in methods_cfg:
        method_name: str = method_cfg["name"]
        method_params: Dict[str, Any] = method_cfg.get("params", {})
        print(f"\n[METHOD] {method_name}")

        # --- Fit ---
        # my_method (CertifiedAtlas) operates in embedding space internally but
        # we decode CFs back to raw feature space for fair comparison.
        embed_model = None   # full TabularClassifier (has .decode()); set for my_method only
        embed_device = "cpu"
        if method_name == "my_method":
            try:
                method, active_model, embed_model, _, active_queries = _build_certified_atlas_method(
                    params=method_params,
                    x_train=x_train_full,
                    y_train=y_train_full,
                    x_queries=x_queries,
                    seed=seed,
                )
                embed_device = method_params.get("device", "cpu")
                active_y_orig = active_model.predict(active_queries)
                space = "raw"  # CFs are decoded back to raw feature space
            except Exception as exc:
                print(f"  [ERROR] build failed: {exc}")
                for q_idx, x_orig, y_orig in zip(query_indices, x_queries, y_orig_all):
                    all_rows.append(
                        _failure_row(method_name, q_idx, x_orig, int(y_orig), n_features,
                                     f"build_error: {exc}", 0.0, space="raw")
                    )
                continue
        else:
            try:
                method = registries["method"].create(method_name, random_seed=seed, **method_params)
                method.fit(x_train=x_train, y_train=y_train, model=model)
            except Exception as exc:
                print(f"  [ERROR] fit() failed: {exc}")
                for q_idx, x_orig, y_orig in zip(query_indices, x_queries, y_orig_all):
                    all_rows.append(
                        _failure_row(method_name, q_idx, x_orig, int(y_orig), n_features,
                                     f"fit_error: {exc}", 0.0)
                    )
                continue
            active_model = model
            active_queries = x_queries
            active_y_orig = y_orig_all
            space = "raw"

        # --- Generate ---
        n_ok = 0
        n_failed = 0
        # raw_queries is always x_queries; used for row storage regardless of
        # whether the method operates in embedding space internally.
        pbar = tqdm(
            enumerate(zip(query_indices, active_queries, active_y_orig)),
            total=n_queries,
            desc=f"  {method_name}",
            unit="query",
        )
        for pos, (q_idx, x_active, y_orig) in pbar:
            x_orig_raw = x_queries[pos]   # raw feature vector, always
            target_class = 1 - int(y_orig)
            example = CounterfactualExample(x=x_active, target_class=target_class)

            t0 = time.perf_counter()
            try:
                result = _call_with_timeout(
                    lambda: method.generate(example=example, model=active_model),
                    timeout_s,
                )
                runtime_s = time.perf_counter() - t0
                x_cf_active = np.asarray(result.x_cf, dtype=np.float32)
                y_cf = int(active_model.predict(x_cf_active[None, :])[0])

                # For my_method: decode embedding-space CF to raw feature space.
                if embed_model is not None:
                    import torch
                    with torch.no_grad():
                        x_cf_row = embed_model.decode(
                            torch.from_numpy(x_cf_active[None, :]).to(embed_device)
                        ).cpu().numpy()[0].astype(np.float32)
                else:
                    x_cf_row = x_cf_active

                all_rows.append(
                    _success_row(
                        method_name, q_idx, x_orig_raw, int(y_orig),
                        x_cf_row, y_cf, result.success, runtime_s, n_features, space,
                    )
                )
                n_ok += int(result.success)
            except TimeoutError:
                runtime_s = time.perf_counter() - t0
                all_rows.append(
                    _failure_row(method_name, q_idx, x_orig_raw, int(y_orig),
                                 n_features, "timeout", runtime_s, space)
                )
                n_failed += 1
            except Exception as exc:
                runtime_s = time.perf_counter() - t0
                all_rows.append(
                    _failure_row(method_name, q_idx, x_orig_raw, int(y_orig),
                                 n_features, str(exc), runtime_s, space)
                )
                n_failed += 1

            pbar.set_postfix(valid=n_ok, failed=n_failed)

    df = pd.DataFrame(all_rows)
    df.to_parquet(output_path, index=False)
    print(f"\n[INFO] Saved {len(df)} rows to {output_path}")

    _print_summary(df)


if __name__ == "__main__":
    main()
