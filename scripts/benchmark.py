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
from counterfactuals.preprocessing import (
    IdentityTransform,
    InverseTransformModel,
    PCATransform,
    adult_ohe_blocks,
    snap_ohe_blocks,
)
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
        "mad_l1_distance": float("nan"),
        "redundancy": float("nan"),
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
    mad_weights: Optional[np.ndarray] = None,
    input_types: Optional[List[str]] = None,
    redundancy: Optional[float] = None,
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
    # MAD-normalized L1: numerical features scaled by per-feature MAD,
    # categorical features contribute a binary mismatch term.
    if mad_weights is not None and input_types is not None:
        per_feat = np.empty(n_features)
        for i, t in enumerate(input_types):
            if t == "numerical":
                per_feat[i] = diff[i] / mad_weights[i]
            else:
                per_feat[i] = float(diff[i] > 1e-6)
        row["mad_l1_distance"] = float(per_feat.mean())
    else:
        row["mad_l1_distance"] = float("nan")
    row["redundancy"] = float(redundancy) if redundancy is not None else float("nan")
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
    _k_per_class = params.get("k_per_class", 500)
    k_per_class = int(_k_per_class) if _k_per_class is not None else None
    _norm_raw = params.get("norm", 2)
    norm = np.inf if str(_norm_raw).lower() in ("inf", "infinity") else int(_norm_raw)
    query_method = str(params.get("query_method", "sorted")).lower()
    if query_method not in {"sorted", "bvh", "knn"}:
        raise ValueError(
            f"my_method.query_method must be one of {{'sorted', 'bvh', 'knn'}}, got {query_method!r}"
        )
    cvxpy_solver_policy = str(params.get("cvxpy_solver_policy", "auto")).lower()
    if cvxpy_solver_policy not in {"auto", "legacy"}:
        raise ValueError(
            f"my_method.cvxpy_solver_policy must be one of {{'auto', 'legacy'}}, got {cvxpy_solver_policy!r}"
        )

    batch_size = int(params.get("batch_size", 256))
    _msc = params.get("max_samples_per_class", None)
    max_samples_per_class = int(_msc) if _msc is not None else None

    # Load backbone + Lightning checkpoint.
    backbone = TabularClassifier(
        input_types=INPUT_TYPES,
        cardinalities=CARDINALITIES,
        hidden_dims=[32, 8],
        num_classes=2,
        dropout=0.2,
    )
    lit = LitClassifier.load_from_checkpoint(ckpt, model=backbone, map_location=device)
    model = lit.model.eval().to(device)

    # Strip Dropout layers before LiRPA certification (identity in eval mode).
    # Numerical features are already StandardScaler-normalised by the datamodule;
    # the network has no BatchNorm, so the QP operates directly in the input space.
    net_for_atlas = torch.nn.Sequential(
        *[m for m in model.net if not isinstance(m, torch.nn.Dropout)]
    )
    atlas_model = TorchModelWrapper(model=net_for_atlas, device=device)

    # Input space = embedding space; data is already standardized by the datamodule.
    z_train   = x_train
    z_queries = x_queries

    # Class-wise k-medoids on raw OHE train points.
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
        if k_per_class is not None:
            med_idx = _medoid_indices(z_train[cls_idx], k_per_class)
        else:
            med_idx = np.arange(len(cls_idx))
        z_parts.append(torch.from_numpy(z_train[cls_idx[med_idx]]).float())
        y_parts.append(torch.full((len(med_idx),), int(cls), dtype=torch.long))
        print(f"  [my_method] class {int(cls)}: {len(cls_idx)} -> {len(med_idx)} medoids")

    medoid_ds = TensorDataset(torch.cat(z_parts), torch.cat(y_parts))

    # Build atlas in input space (standardized numerical + raw OHE categorical).
    eps_strategy = NearestOppositeClassClearanceStrategy(alpha=eps_alpha)
    atlas = CertifiedAtlas(
        net_for_atlas,
        medoid_ds,
        device=device,
        default_query_method=query_method,
        cvxpy_solver_policy=cvxpy_solver_policy,
    )
    atlas.build(
        eps_strategy=eps_strategy,
        norm=norm,
        batch_size=batch_size,
        max_samples_per_class=max_samples_per_class,
    )

    # OHE simplex constraints: sum(block)==1 is valid in raw OHE space.
    cat_slices = [
        (s, e)
        for t, (s, e) in zip(INPUT_TYPES, model._slices)
        if t == "categorical"
    ]
    atlas.ohe_slices = cat_slices if cat_slices else None

    print(atlas.summary())

    atlas_method = CertifiedAtlasMethod(atlas=atlas, random_seed=seed)
    atlas_method.fit(x_train=z_train, y_train=y_train, model=atlas_model)

    return atlas_method, atlas_model, model, z_train, z_queries, None


# ---------------------------------------------------------------------------
# PyTorch checkpoint model loader
# ---------------------------------------------------------------------------

def _build_torch_model_from_checkpoint(checkpoint: str, device: str = "cpu") -> 'TorchModelWrapper':
    """Load TabularClassifier from a Lightning checkpoint and return a TorchModelWrapper.

    Dropout layers are stripped so the model is deterministic at inference time.
    This wrapper can be used as the shared benchmark model so that all methods
    (DiCE, FACE, NN, growing_spheres, CPP) target the same classifier.
    """
    import torch
    from models.classifiers import TabularClassifier
    from training.datamodules.adult import CARDINALITIES, INPUT_TYPES
    from training.lit_classifier import LitClassifier
    from counterfactuals.models.torch_model import TorchModelWrapper

    backbone = TabularClassifier(
        input_types=INPUT_TYPES,
        cardinalities=CARDINALITIES,
        hidden_dims=[32, 8],
        num_classes=2,
        dropout=0.2,
    )
    lit = LitClassifier.load_from_checkpoint(checkpoint, model=backbone, map_location=device)
    net = lit.model.eval().to(device)
    net_no_dropout = torch.nn.Sequential(
        *[m for m in net.net if not isinstance(m, torch.nn.Dropout)]
    )
    return TorchModelWrapper(model=net_no_dropout, device=device)


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


def _build_shared_preprocessing(cfg: Dict[str, Any], seed: int):
    prep_cfg = cfg.get("preprocessing")
    if prep_cfg is None:
        return IdentityTransform(), "identity"
    if isinstance(prep_cfg, str):
        name = prep_cfg
        params: Dict[str, Any] = {}
        enabled = True
    else:
        enabled = bool(prep_cfg.get("enabled", True))
        if not enabled:
            return IdentityTransform(), "identity"
        name = str(prep_cfg.get("name", "identity"))
        params = dict(prep_cfg.get("params", {}))

    if name == "identity":
        return IdentityTransform(), "identity"
    if name == "pca":
        pca_params = {
            "n_components": params.get("n_components", 0.99),
            "svd_solver": params.get("svd_solver", "full"),
            "whiten": bool(params.get("whiten", False)),
            "random_state": int(params.get("random_state", seed)),
        }
        return PCATransform(**pca_params), "pca"

    raise ValueError(f"Unsupported preprocessing name: {name}")


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

    # --- MAD weights for MAD-normalized L1 proximity ---
    # Try to load feature type annotations from the dataset's datamodule.
    # Falls back to treating all features as numerical if not available.
    try:
        from training.datamodules.adult import OHE_FEATURE_TYPES as _input_types
    except ImportError:
        _input_types = ["numerical"] * n_features

    mad_weights = np.ones(n_features, dtype=np.float64)
    for i, t in enumerate(_input_types):
        if t == "numerical":
            col = x_train_full[:, i].astype(np.float64)
            mad = float(np.median(np.abs(col - np.median(col))))
            # For zero-inflated features (e.g. capital-gain/loss) MAD=0 because the
            # median equals most values.  These features are already StandardScaler-
            # normalised (std=1), so use 1.0 as the unit scale.
            mad_weights[i] = mad if mad > 0 else 1.0
    # Categorical features keep weight = 1.0 (binary mismatch is already in [0, 1]).

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

    # --- Shared preprocessing for generation space ---
    transform, transform_name = _build_shared_preprocessing(cfg, seed=seed)
    transform.fit(x_train)
    x_train_gen = transform.transform(x_train)
    x_queries_gen = transform.transform(x_queries)
    print(
        f"[INFO] Preprocessing: {transform_name} "
        f"({x_train.shape[1]} -> {x_train_gen.shape[1]} dims)."
    )

    ohe_blocks = None
    if ds_cfg["name"] == "adult":
        try:
            ohe_blocks = adult_ohe_blocks()
        except Exception as exc:
            print(f"[WARNING] Could not load Adult OHE block metadata for inverse snap: {exc}")

    # --- Model ---
    model_cfg = cfg["model"]
    if model_cfg["name"] == "tabular_classifier_ckpt":
        ckpt = model_cfg.get("params", {}).get("checkpoint")
        device = model_cfg.get("params", {}).get("device", "cpu")
        print(f"[INFO] Loading TabularClassifier from checkpoint: {ckpt}")
        model = _build_torch_model_from_checkpoint(ckpt, device=device)
        acc = float(np.mean(model.predict(x_train) == y_train))
        print(f"[INFO] Train accuracy: {acc:.3f}")
    else:
        model = registries["model"].create(model_cfg["name"], **model_cfg.get("params", {}))
        estimator = getattr(model, "estimator", None)
        if estimator is not None and hasattr(estimator, "fit"):
            print(f"[INFO] Fitting model ({model_cfg['name']}) on {len(x_train)} train samples...")
            estimator.fit(x_train, y_train)
            acc = float(np.mean(model.predict(x_train) == y_train))
            print(f"[INFO] Train accuracy: {acc:.3f}")

    model_for_methods = InverseTransformModel(base_model=model, transform=transform, ohe_blocks=ohe_blocks)

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
        if method_name in ("my_method", "cpp"):
            try:
                method, active_model, embed_model, _, active_queries, _ = _build_certified_atlas_method(
                    params=method_params,
                    x_train=x_train,
                    y_train=y_train,
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
                method.fit(x_train=x_train_gen, y_train=y_train, model=model_for_methods)
            except Exception as exc:
                print(f"  [ERROR] fit() failed: {exc}")
                for q_idx, x_orig, y_orig in zip(query_indices, x_queries, y_orig_all):
                    all_rows.append(
                        _failure_row(method_name, q_idx, x_orig, int(y_orig), n_features,
                                     f"fit_error: {exc}", 0.0)
                    )
                continue
            active_model = model_for_methods
            active_queries = x_queries_gen
            active_y_orig = y_orig_all
            space = "gen"

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

                # For cpp/my_method: QP returns a point in raw OHE space; apply
                # argmax-snap to produce a valid discrete OHE vector.
                # y_cf and success are re-evaluated on the decoded CF using the
                # shared full model, so all methods are judged in the same space.
                if embed_model is not None:
                    import torch
                    with torch.no_grad():
                        x_cf_row = embed_model.decode(
                            torch.from_numpy(x_cf_active[None, :]).to(embed_device)
                        ).cpu().numpy()[0].astype(np.float32)
                    y_cf = int(model.predict(x_cf_row[None, :])[0])
                    cf_success = (y_cf == target_class)
                else:
                    x_cf_row = np.asarray(transform.inverse_transform(x_cf_active), dtype=np.float32)
                    if ohe_blocks is not None:
                        x_cf_row = np.asarray(snap_ohe_blocks(x_cf_row, ohe_blocks), dtype=np.float32)
                    y_cf = int(model.predict(x_cf_row[None, :])[0])
                    cf_success = (y_cf == target_class)

                # Redundancy: fraction of changed features that can be individually
                # reverted without flipping the CF out of the target class.
                # Always uses the shared benchmark classifier (model) for consistency.
                redundancy_val = 0.0
                diff_raw = np.abs(
                    x_cf_row.astype(np.float64) - x_orig_raw.astype(np.float64)
                )
                changed_feats = np.where(diff_raw > 1e-6)[0]
                if len(changed_feats) > 0:
                    n_redundant = 0
                    for k in changed_feats:
                        x_test = x_cf_row.copy().astype(np.float32)
                        x_test[k] = x_orig_raw[k]
                        if int(model.predict(x_test[None, :])[0]) == target_class:
                            n_redundant += 1
                    redundancy_val = n_redundant / len(changed_feats)

                all_rows.append(
                    _success_row(
                        method_name, q_idx, x_orig_raw, int(y_orig),
                        x_cf_row, y_cf, cf_success, runtime_s, n_features, space,
                        mad_weights=mad_weights,
                        input_types=_input_types,
                        redundancy=redundancy_val,
                    )
                )
                n_ok += int(cf_success)
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
    df.to_parquet(output_path, index=False, compression="gzip")
    print(f"\n[INFO] Saved {len(df)} rows to {output_path}")

    _print_summary(df)


if __name__ == "__main__":
    main()
