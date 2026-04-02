#!/usr/bin/env python3
"""Benchmark counterfactual methods from a single- or multi-dataset config.

Usage:
    python scripts/benchmark.py --config configs/benchmarks/benchmark_adult.yaml
    python scripts/benchmark.py --config configs/benchmarks/benchmark_adult.yaml --output results/run2.parquet
    python scripts/benchmark.py --config configs/benchmarks/benchmark_adult.yaml --seed 123 --n_queries 200
    python scripts/benchmark.py --config configs/benchmarks/benchmark_adult.yaml --methods dice face nearest_neighbor
    python scripts/benchmark.py --config configs/benchmarks/benchmark_meeting_all.yaml
    python scripts/benchmark.py --config configs/benchmarks/benchmark_meeting_all.yaml --datasets adult compas

Output:
    Single-dataset config:
        A .parquet file (flat, for notebooks and analysis).
    Multi-dataset config:
        One per-dataset .parquet file plus one combined .parquet file.
"""

from __future__ import annotations

import argparse
import itertools
import signal
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress noisy DeprecationWarning from sklearn_extra (distutils.LooseVersion).
warnings.filterwarnings(
    "ignore",
    message="distutils Version classes are deprecated",
    category=DeprecationWarning,
    module=r"sklearn_extra",
)

# Allow running from repo root without installing as package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactuals.benchmarks import BenchmarkResult, MethodResult, QueryResult, create_default_registries
from counterfactuals.preprocessing import (
    IdentityTransform,
    InverseTransformModel,
    PCATransform,
    snap_ohe_blocks,
)
from counterfactuals.utils.config import read_yaml
from counterfactuals.utils.seed import seed_everything
from dataset_specs import get_tabular_dataset_spec


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


def _normalize_torch_device(device: Any) -> str:
    """Normalize user/config device aliases to valid torch device strings."""
    import torch

    raw = str(device or "cpu").strip().lower()
    if raw in {"cpu"}:
        return "cpu"
    if raw in {"auto", "gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw.startswith("gpu:"):
        idx = raw.split(":", 1)[1]
        return f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda"):
        return raw if torch.cuda.is_available() else "cpu"
    if raw.startswith("mps"):
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return raw if has_mps else "cpu"
    return raw


def _load_lit_checkpoint_resilient(lit_cls, checkpoint: str, backbone, map_location: str):
    """Load Lightning checkpoint with fallback for legacy/unknown storage tags."""
    try:
        return lit_cls.load_from_checkpoint(checkpoint, model=backbone, map_location=map_location)
    except RuntimeError as exc:
        msg = str(exc)
        if "tagged with gpu" in msg and map_location != "cpu":
            print(
                "[WARNING] Checkpoint uses legacy 'gpu' storage tag. "
                "Retrying load with map_location=cpu."
            )
            return lit_cls.load_from_checkpoint(checkpoint, model=backbone, map_location="cpu")
        raise


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
# Farthest Point Sampling
# ---------------------------------------------------------------------------

def _fps_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Greedy Farthest Point Sampling: return indices of k maximally spread points.

    Initialization: the point closest to the class mean (deterministic, no
    random seed needed).  Each subsequent step picks the point with the largest
    minimum distance to the already-selected subset.  Uses scipy.cdist for
    O(N·k) pairwise distance computation with no extra dependencies.
    """
    from scipy.spatial.distance import cdist

    k = min(k, len(X))
    if k == len(X):
        return np.arange(len(X))

    # Start from the point nearest to the class centroid.
    mean = X.mean(axis=0, keepdims=True)
    first = int(cdist(mean, X, metric="euclidean").argmin())

    selected = [first]
    # min_dists[i] = distance from X[i] to the closest selected point so far.
    min_dists = cdist(X[first : first + 1], X, metric="euclidean")[0]

    for _ in range(k - 1):
        farthest = int(np.argmax(min_dists))
        selected.append(farthest)
        new_dists = cdist(X[farthest : farthest + 1], X, metric="euclidean")[0]
        np.minimum(min_dists, new_dists, out=min_dists)

    return np.array(selected)


def _compute_query_metrics(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    n_features: int,
    mad_weights: Optional[np.ndarray],
    input_types: Optional[List[str]],
) -> tuple[float, float, float, float]:
    """Return (l2, l1, l0_sparsity, mad_l1) for a successful CF."""
    diff = np.abs(x_cf.astype(np.float64) - x_orig.astype(np.float64))
    l2 = float(np.linalg.norm(diff, ord=2))
    l1 = float(np.linalg.norm(diff, ord=1))
    l0 = float(np.mean(diff > 1e-6))
    if mad_weights is not None and input_types is not None:
        per_feat = np.empty(n_features)
        for i, t in enumerate(input_types):
            if t == "numerical":
                per_feat[i] = diff[i] / mad_weights[i]
            else:
                per_feat[i] = float(diff[i] > 1e-6)
        mad_l1 = float(per_feat.mean())
    else:
        mad_l1 = float("nan")
    return l2, l1, l0, mad_l1


# ---------------------------------------------------------------------------
# Dataset metadata helper
# ---------------------------------------------------------------------------

def _get_tabular_spec(dataset_name: str):
    """Return the shared tabular dataset spec when one exists."""
    try:
        return get_tabular_dataset_spec(dataset_name)
    except KeyError:
        return None


def _sample_balanced_indices(
    y: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample up to ``per_class`` items per class without replacement."""
    selected = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        selected.extend(chosen.tolist())
    return np.array(sorted(selected), dtype=np.int64)


def _make_failure_query_result(
    q_idx: int,
    runtime_s: float,
    error: str,
    target_class: Optional[int],
) -> QueryResult:
    return QueryResult(
        query_idx=int(q_idx),
        x_cf=None,
        y_cf=None,
        success=False,
        runtime_s=float(runtime_s),
        error=error,
        l2_distance=float("nan"),
        l1_distance=float("nan"),
        l0_sparsity=float("nan"),
        mad_l1_distance=float("nan"),
        redundancy=float("nan"),
        target_class=(None if target_class is None else int(target_class)),
    )


def _build_query_tasks(
    *,
    dataset_name: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model,
    sampling_cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (query_indices, y_true_tasks, y_orig_tasks, target_classes)."""
    n_queries = int(sampling_cfg.get("n_queries", len(x_test)))
    if dataset_name != "mnist":
        n_queries = min(n_queries, len(x_test))
        query_indices = np.sort(rng.choice(len(x_test), size=n_queries, replace=False))
        y_orig = model.predict(x_test[query_indices])
        return query_indices, y_test[query_indices], y_orig, None

    per_class = int(sampling_cfg.get("balanced_per_class", 10))
    target_policy = str(sampling_cfg.get("target_policy", "all_other_classes")).lower()
    source_indices = _sample_balanced_indices(y=y_test, per_class=per_class, rng=rng)
    if source_indices.size == 0:
        raise ValueError("No MNIST source queries were selected.")

    source_preds = model.predict(x_test[source_indices])
    n_classes = int(model.predict_proba(x_test[source_indices[:1]]).shape[1])
    if target_policy != "all_other_classes":
        raise ValueError(
            "MNIST benchmark currently supports only sampling.target_policy='all_other_classes'."
        )

    task_indices: List[int] = []
    task_true: List[int] = []
    task_orig: List[int] = []
    task_targets: List[int] = []
    for idx, source_pred in zip(source_indices, source_preds):
        for target_class in range(n_classes):
            if target_class == int(source_pred):
                continue
            task_indices.append(int(idx))
            task_true.append(int(y_test[idx]))
            task_orig.append(int(source_pred))
            task_targets.append(int(target_class))

    return (
        np.asarray(task_indices, dtype=np.int64),
        np.asarray(task_true, dtype=np.int64),
        np.asarray(task_orig, dtype=np.int64),
        np.asarray(task_targets, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# CertCFAtlas builder
# ---------------------------------------------------------------------------

def _build_certcf_method(
    params: Dict[str, Any],
    model_params: Dict[str, Any],
    dataset_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_queries: np.ndarray,
    seed: int,
):
    """Build CertCF from a checkpoint.

    Returns (atlas_method, atlas_model, embed_model, z_train, z_queries).

    ``generate()`` calls should pass z_queries and atlas_model (embedding-space
    model).  The returned ``embed_model`` (full TabularClassifier) exposes a
    ``decode()`` method to map embedding-space CFs back to raw feature space.
    """
    import torch

    from certcf import NearestOppositeClassClearanceStrategy
    from training.lit_classifier import LitClassifier

    from counterfactuals.methods.certcf import CertCF
    from counterfactuals.models.torch_model import TorchModelWrapper

    ckpt = params.get("checkpoint")
    if not ckpt:
        raise ValueError("certcf requires a 'checkpoint' param pointing to the .ckpt file.")

    requested_device = params.get("device", "cpu")
    device = _normalize_torch_device(requested_device)
    if str(requested_device).strip().lower() != device:
        print(f"[INFO] certcf device normalized: {requested_device!r} -> {device!r}")
    eps_alpha = float(params.get("eps_alpha", 0.25))
    _k_per_class = params.get("k_per_class", 500)
    k_per_class = int(_k_per_class) if _k_per_class is not None else None
    _norm_raw = params.get("norm", 2)
    norm = np.inf if str(_norm_raw).lower() in ("inf", "infinity") else int(_norm_raw)
    query_method = str(params.get("query_method", "sorted")).lower()
    if query_method not in {"sorted", "bvh"}:
        raise ValueError(
            f"certcf.query_method must be one of {{'sorted', 'bvh'}}, got {query_method!r}"
        )
    batch_size = int(params.get("batch_size", 256))
    atlas_subsample_method = str(params.get("atlas_subsample_method", "kmedoids")).lower()
    if atlas_subsample_method not in {"kmedoids", "bandit_kmedoids", "fps", "kmeans", "density_flat_kmedoids"}:
        raise ValueError(
            f"certcf.atlas_subsample_method must be one of {{'kmedoids', 'bandit_kmedoids', 'fps', 'kmeans', 'density_flat_kmedoids'}}, "
            f"got {atlas_subsample_method!r}"
        )
    solver_maxiter = int(params.get("solver_maxiter", 500))
    # Architecture params come from the shared model config, not method params.
    dataset_module = str(model_params.get("dataset_module", dataset_name))
    hidden_dims = list(model_params.get("hidden_dims", [32, 8]))
    dropout = float(model_params.get("dropout", 0.2))

    z_train = x_train
    z_queries = x_queries
    cat_slices = None

    if dataset_module == "mnist":
        from models.classifiers import MNISTClassifier

        num_classes = int(model_params.get("num_classes", 10))
        backbone = MNISTClassifier(num_classes=num_classes)
        lit = _load_lit_checkpoint_resilient(LitClassifier, ckpt, backbone, map_location=device)
        model = lit.model.eval().to(device)
        atlas_model = TorchModelWrapper(model=model, device=device)
        cnn = True
    else:
        from models.classifiers import TabularClassifier

        spec = get_tabular_dataset_spec(dataset_module)
        backbone = TabularClassifier(
            input_types=list(spec.input_types),
            cardinalities=list(spec.cardinalities),
            hidden_dims=hidden_dims,
            num_classes=2,
            dropout=dropout,
        )
        lit = _load_lit_checkpoint_resilient(LitClassifier, ckpt, backbone, map_location=device)
        model = lit.model.eval().to(device)

        # Wrap the full model (Dropout included — CertCF._fit strips it for LiRPA).
        atlas_model = TorchModelWrapper(model=model.net, device=device)

        # OHE simplex constraints: sum(block)==1 is valid in raw OHE space.
        cat_slices = list(spec.categorical_slices) or None
        cnn = False

    eps_strategy = NearestOppositeClassClearanceStrategy(alpha=eps_alpha)
    atlas_method = CertCF(
        model=atlas_model,
        norm=norm,
        eps_strategy=eps_strategy,
        batch_size=batch_size,
        ohe_slices=cat_slices,
        cnn=cnn,
        default_query_method=query_method,
        solver_maxiter=solver_maxiter,
        k_per_class=k_per_class,
        subsample_method=atlas_subsample_method,
        random_seed=seed,
    )
    atlas_method.fit(x_train=z_train, y_train=y_train)

    return atlas_method, atlas_model, model, z_train, z_queries, None


# ---------------------------------------------------------------------------
# PyTorch checkpoint model loader
# ---------------------------------------------------------------------------

def _build_torch_model_from_checkpoint(
    checkpoint: str,
    device: str = "cpu",
    dataset_module: str = "adult",
    hidden_dims: list = [32, 8],
    dropout: float = 0.2,
) -> 'TorchModelWrapper':
    """Load TabularClassifier from a Lightning checkpoint and return a TorchModelWrapper.

    Dropout layers are stripped so the model is deterministic at inference time.
    This wrapper can be used as the shared benchmark model so that all methods
    (DiCE, FACE, NN, growing_spheres, CertCF) target the same classifier.
    """
    import torch
    from models.classifiers import TabularClassifier
    from training.lit_classifier import LitClassifier
    from counterfactuals.models.torch_model import TorchModelWrapper

    spec = get_tabular_dataset_spec(dataset_module)

    requested_device = device
    device = _normalize_torch_device(device)
    if str(requested_device).strip().lower() != device:
        print(f"[INFO] model device normalized: {requested_device!r} -> {device!r}")

    backbone = TabularClassifier(
        input_types=list(spec.input_types),
        cardinalities=list(spec.cardinalities),
        hidden_dims=hidden_dims,
        num_classes=2,
        dropout=dropout,
    )
    lit = _load_lit_checkpoint_resilient(
        LitClassifier, checkpoint, backbone, map_location=device
    )
    net = lit.model.eval().to(device)
    net_no_dropout = torch.nn.Sequential(
        *[m for m in net.net if not isinstance(m, torch.nn.Dropout)]
    )
    return TorchModelWrapper(model=net_no_dropout, device=device)


def _build_mnist_model_from_checkpoint(
    checkpoint: str,
    device: str = "cpu",
    num_classes: int = 10,
    input_shape: tuple[int, ...] | None = (1, 28, 28),
) -> "TorchModelWrapper":
    """Load MNISTClassifier from checkpoint and optionally wrap flat inputs."""
    from counterfactuals.models.torch_model import TorchModelWrapper
    from models.classifiers import MNISTClassifier
    from training.lit_classifier import LitClassifier

    requested_device = device
    device = _normalize_torch_device(device)
    if str(requested_device).strip().lower() != device:
        print(f"[INFO] model device normalized: {requested_device!r} -> {device!r}")

    backbone = MNISTClassifier(num_classes=num_classes)
    lit = _load_lit_checkpoint_resilient(
        LitClassifier, checkpoint, backbone, map_location=device
    )
    net = lit.model.eval().to(device)
    return TorchModelWrapper(model=net, device=device, input_shape=input_shape)


# ---------------------------------------------------------------------------
# Grid search expansion
# ---------------------------------------------------------------------------

def _expand_grid(methods_cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand method configs with list-valued params into all combinations.

    Any param whose value is a list is treated as a sweep axis. The Cartesian
    product of all sweep axes is generated, and the run_name gets a
    ``_param=value`` suffix for each varied param. Fixed params (non-list) are
    passed through unchanged.

    Example::

        - name: face
          run_name: face_knn
          params:
            graph_mode: knn
            n_neighbors: [5, 15, 50]
            k_per_class: [200, 500, 1000]

    expands to 9 entries: face_knn_n_neighbors=5_k_per_class=200, ...
    """
    expanded: List[Dict[str, Any]] = []
    for entry in methods_cfg:
        name = entry["name"]
        run_name = entry.get("run_name", name)
        params = dict(entry.get("params") or {})

        sweep = {k: v for k, v in params.items() if isinstance(v, list)}
        fixed = {k: v for k, v in params.items() if not isinstance(v, list)}

        if not sweep:
            expanded.append(entry)
            continue

        keys = list(sweep.keys())
        for combo in itertools.product(*[sweep[k] for k in keys]):
            combo_dict = dict(zip(keys, combo))
            suffix = "-".join(f"{k}={v}" for k, v in combo_dict.items())
            expanded.append({
                "name": name,
                "run_name": f"{run_name}_{suffix}",
                "params": {**fixed, **combo_dict},
            })

    return expanded


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides on top of the YAML config dict (mutates cfg)."""
    if args.output is not None:
        if "datasets" in cfg:
            cfg["output"] = args.output
        else:
            cfg.setdefault("output", {})["path"] = args.output
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_queries is not None:
        if "datasets" in cfg:
            cfg.setdefault("sampling", {})["n_queries"] = args.n_queries
        else:
            cfg.setdefault("sampling", {})["n_queries"] = args.n_queries
    if args.methods is not None:
        allowed = set(args.methods)
        cfg["methods"] = [
            m for m in cfg.get("methods", [])
            if m.get("run_name", m["name"]) in allowed or m["name"] in allowed
        ]
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


def _merge_methods(
    shared_methods: List[Dict[str, Any]],
    method_overrides: Dict[str, Dict[str, Any]],
    model_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build the final per-dataset method list for a multi-dataset config."""
    result = []
    for entry in shared_methods:
        m = deepcopy(entry)
        run_name = m.get("run_name", m["name"])

        override = method_overrides.get(run_name) or method_overrides.get(m["name"]) or {}
        if override:
            m.setdefault("params", {}).update(override)

        if m["name"] == "certcf":
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
    """Build a single-dataset config dict from a multi-dataset config."""
    ds_name = ds_cfg["name"]
    model_params = ds_cfg.get("model", {}).get("params", {})
    method_overrides = ds_cfg.get("method_overrides", {})

    merged_methods = _merge_methods(
        shared_methods=global_cfg.get("methods", []),
        method_overrides=method_overrides,
        model_params=model_params,
    )

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


# ---------------------------------------------------------------------------
# Core per-dataset runner
# ---------------------------------------------------------------------------

def run_single_dataset(cfg: Dict[str, Any]) -> BenchmarkResult:
    """Run all configured methods on a single dataset and save results.

    Parameters
    ----------
    cfg:
        Fully resolved config dict (same schema as the single-dataset YAML files).

    Returns
    -------
    BenchmarkResult
        The structured benchmark result (already persisted to disk).
    """
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    # --- Dataset ---
    registries = create_default_registries()
    ds_cfg = cfg["dataset"]
    dataset = registries["dataset"].create(ds_cfg["name"], **ds_cfg.get("params", {}))
    dataset.load()
    x_train_full, y_train_full = dataset.get_train()
    x_test_full, y_test_full = dataset.get_test()
    n_features = x_train_full.shape[1]
    spec = getattr(dataset, "spec", None)

    # --- MAD weights for MAD-normalized L1 proximity ---
    # Use shared schema metadata when available; otherwise treat every dimension
    # as numerical (e.g. for image datasets).
    _input_types = list(spec.ohe_feature_types) if spec is not None else ["numerical"] * n_features

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

    # --- Train subsampling (shared by all methods EXCEPT certcf) ---
    # k-medoids is reserved for the CertCF method's own atlas construction (see
    # k_per_class inside _build_certcf_method). Other methods receive
    # either the full training set or a *random* subsample if n_train is set.
    sampling_cfg = cfg.get("sampling", {})
    n_train = int(sampling_cfg.get("n_train", len(x_train_full)))

    x_train, y_train = subsample_train(x_train_full, y_train_full, n_train, "random", rng)

    # --- Shared preprocessing for generation space ---
    transform, transform_name = _build_shared_preprocessing(cfg, seed=seed)
    transform.fit(x_train)
    x_train_gen = transform.transform(x_train)

    ohe_blocks = tuple(spec.ohe_blocks) if spec is not None and spec.ohe_blocks else None

    # --- Model ---
    model_cfg = cfg["model"]
    if model_cfg["name"] == "tabular_classifier_ckpt":
        _model_params = model_cfg.get("params", {})
        ckpt = _model_params.get("checkpoint")
        device = _model_params.get("device", "cpu")
        _ds_module = _model_params.get("dataset_module", ds_cfg["name"])
        _hidden_dims = list(_model_params.get("hidden_dims", [32, 8]))
        _dropout = float(_model_params.get("dropout", 0.2))
        print(f"[INFO] Loading TabularClassifier from checkpoint: {ckpt}")
        model = _build_torch_model_from_checkpoint(
            ckpt, device=device,
            dataset_module=_ds_module,
            hidden_dims=_hidden_dims,
            dropout=_dropout,
        )
        acc = float(np.mean(model.predict(x_train) == y_train))
        print(f"[INFO] Train accuracy: {acc:.3f}")
    elif model_cfg["name"] == "mnist_classifier_ckpt":
        _model_params = model_cfg.get("params", {})
        ckpt = _model_params.get("checkpoint")
        device = _model_params.get("device", "cpu")
        num_classes = int(_model_params.get("num_classes", 10))
        print(f"[INFO] Loading MNISTClassifier from checkpoint: {ckpt}")
        model = _build_mnist_model_from_checkpoint(
            ckpt,
            device=device,
            num_classes=num_classes,
            input_shape=(1, 28, 28),
        )
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

    # --- Test task selection (fixed set, shared across methods) ---
    query_indices, y_true_all, y_orig_all, target_classes_all = _build_query_tasks(
        dataset_name=ds_cfg["name"],
        x_test=x_test_full,
        y_test=y_test_full,
        model=model,
        sampling_cfg=sampling_cfg,
        rng=rng,
    )
    x_queries = x_test_full[query_indices]
    x_queries_gen = transform.transform(x_queries)
    print(f"[INFO] {len(query_indices)} benchmark tasks selected from {len(x_test_full)} test samples.")
    print(
        f"[INFO] Preprocessing: {transform_name} "
        f"({x_train.shape[1]} -> {x_train_gen.shape[1]} dims)."
    )

    timeout_s = int(cfg.get("timeout_per_sample", 0))
    output_path = Path(cfg.get("output", {}).get("path", "results/benchmark.parquet"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods_cfg = _expand_grid(cfg.get("methods", []))
    print(f"[INFO] {len(methods_cfg)} method runs after grid expansion.")

    benchmark_result = BenchmarkResult(
        dataset=ds_cfg["name"],
        seed=seed,
        x_queries=x_queries,
        y_orig=y_orig_all,
        y_true=y_true_all,
    )

    for method_cfg in methods_cfg:
        method_name: str = method_cfg["name"]        # registry key — selects the implementation
        run_name: str = method_cfg.get("run_name", method_name)  # label used in results
        method_params: Dict[str, Any] = method_cfg.get("params") or {}
        print(f"\n[METHOD] {run_name}" + (f" (impl: {method_name})" if run_name != method_name else ""))

        # --- Fit / Build ---
        # certcf (CertCFAtlas) operates in embedding space internally but
        # we decode CFs back to raw feature space for fair comparison.
        embed_model = None   # full TabularClassifier (has .decode()); set for certcf only
        embed_device = "cpu"
        build_time_s = 0.0
        if method_name == "certcf":
            try:
                _t_build = time.perf_counter()
                method, active_model, embed_model, _, active_queries, _ = _build_certcf_method(
                    params=method_params,
                    model_params=model_cfg.get("params", {}),
                    dataset_name=ds_cfg["name"],
                    x_train=x_train,
                    y_train=y_train,
                    x_queries=x_queries,
                    seed=seed,
                )
                build_time_s = time.perf_counter() - _t_build
                embed_device = method_params.get("device", "cpu")
                active_y_orig = y_orig_all
                space = "raw"  # CFs are decoded back to raw feature space
            except Exception as exc:
                print(f"  [ERROR] build failed: {exc}")
                qrs = [
                    _make_failure_query_result(
                        q_idx=int(q_idx),
                        runtime_s=0.0,
                        error=f"build_error: {exc}",
                        target_class=(
                            int(target_classes_all[pos])
                            if target_classes_all is not None
                            else 1 - int(y_orig_all[pos])
                        ),
                    )
                    for pos, q_idx in enumerate(query_indices)
                ]
                benchmark_result.method_results.append(MethodResult(
                    method=method_name, run_name=run_name, params=method_params,
                    build_time_s=0.0, space="raw", query_results=qrs,
                ))
                continue
        else:
            try:
                method = registries["method"].create(method_name, model=model_for_methods, random_seed=seed, **method_params)
                _t_build = time.perf_counter()
                method.fit(x_train=x_train_gen, y_train=y_train)
                build_time_s = time.perf_counter() - _t_build
            except Exception as exc:
                print(f"  [ERROR] fit() failed: {exc}")
                qrs = [
                    _make_failure_query_result(
                        q_idx=int(q_idx),
                        runtime_s=0.0,
                        error=f"fit_error: {exc}",
                        target_class=(
                            int(target_classes_all[pos])
                            if target_classes_all is not None
                            else 1 - int(y_orig_all[pos])
                        ),
                    )
                    for pos, q_idx in enumerate(query_indices)
                ]
                benchmark_result.method_results.append(MethodResult(
                    method=method_name, run_name=run_name, params=method_params,
                    build_time_s=0.0, space="gen", query_results=qrs,
                ))
                continue
            active_model = model_for_methods
            active_queries = x_queries_gen
            active_y_orig = y_orig_all
            space = "gen"
        print(f"  [build] {build_time_s:.2f}s")

        # --- Generate ---
        n_ok = 0
        n_failed = 0
        query_results: List[QueryResult] = []
        task_targets = (
            target_classes_all
            if target_classes_all is not None
            else np.asarray([1 - int(y) for y in active_y_orig], dtype=np.int64)
        )
        pbar = tqdm(
            enumerate(zip(query_indices, active_queries, active_y_orig, task_targets)),
            total=len(query_indices),
            desc=f"  {run_name}",
            unit="query",
        )
        for pos, (q_idx, x_active, y_orig, target_class) in pbar:
            x_orig_raw = x_queries[pos]   # raw feature vector, always
            target_class = int(target_class)

            t0 = time.perf_counter()
            try:
                result = _call_with_timeout(
                    lambda: method.generate(x=x_active, target_class=target_class),
                    timeout_s,
                )
                runtime_s = time.perf_counter() - t0
                if result.x_cf is None:
                    reason = str(result.metadata.get("reason", "no_counterfactual"))
                    query_results.append(_make_failure_query_result(
                        q_idx=int(q_idx),
                        runtime_s=runtime_s,
                        error=reason,
                        target_class=target_class,
                    ))
                    n_failed += 1
                    pbar.set_postfix(valid=n_ok, failed=n_failed)
                    continue
                x_cf_active = np.asarray(result.x_cf, dtype=np.float32)

                # For certcf: atlas outputs are already in raw/OHE space.
                if embed_model is not None:
                    x_cf_row = x_cf_active
                    y_cf = int(model.predict(x_cf_row[None, :])[0])
                else:
                    x_cf_row = np.asarray(transform.inverse_transform(x_cf_active), dtype=np.float32)
                    if ohe_blocks is not None:
                        x_cf_row = np.asarray(snap_ohe_blocks(x_cf_row, ohe_blocks), dtype=np.float32)
                    y_cf = int(model.predict(x_cf_row[None, :])[0])

                # All methods are evaluated the same way:
                # success means the returned CF flips the original model prediction.
                cf_success = (y_cf == target_class)

                # Redundancy: fraction of changed features that can be individually
                # reverted without flipping the CF out of the target class.
                redundancy_val = 0.0
                diff_raw = np.abs(x_cf_row.astype(np.float64) - x_orig_raw.astype(np.float64))
                changed_feats = np.where(diff_raw > 1e-6)[0]
                if len(changed_feats) > 0:
                    n_redundant = 0
                    for k in changed_feats:
                        x_test = x_cf_row.copy().astype(np.float32)
                        x_test[k] = x_orig_raw[k]
                        if int(model.predict(x_test[None, :])[0]) == target_class:
                            n_redundant += 1
                    redundancy_val = n_redundant / len(changed_feats)

                l2, l1, l0, mad_l1 = _compute_query_metrics(
                    x_orig_raw, x_cf_row, n_features, mad_weights, _input_types
                )
                query_results.append(QueryResult(
                    query_idx=int(q_idx),
                    x_cf=x_cf_row,
                    y_cf=y_cf,
                    success=cf_success,
                    runtime_s=runtime_s,
                    error=None,
                    l2_distance=l2,
                    l1_distance=l1,
                    l0_sparsity=l0,
                    mad_l1_distance=mad_l1,
                    redundancy=redundancy_val,
                    target_class=target_class,
                ))
                n_ok += int(cf_success)
            except TimeoutError:
                runtime_s = time.perf_counter() - t0
                query_results.append(_make_failure_query_result(
                    q_idx=int(q_idx),
                    runtime_s=runtime_s,
                    error="timeout",
                    target_class=target_class,
                ))
                n_failed += 1
            except Exception as exc:
                runtime_s = time.perf_counter() - t0
                query_results.append(_make_failure_query_result(
                    q_idx=int(q_idx),
                    runtime_s=runtime_s,
                    error=str(exc),
                    target_class=target_class,
                ))
                n_failed += 1

            pbar.set_postfix(valid=n_ok, failed=n_failed)

        benchmark_result.method_results.append(MethodResult(
            method=method_name,
            run_name=run_name,
            params=method_params,
            build_time_s=build_time_s,
            space=space,
            query_results=query_results,
        ))

    # --- Save ---
    df = benchmark_result.to_dataframe()
    df.to_parquet(output_path, index=False, compression="gzip")
    print(f"\n[INFO] Saved benchmark parquet to {output_path}")

    benchmark_result.summary()
    return benchmark_result


def run_multi_dataset(
    global_cfg: Dict[str, Any],
    dataset_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run a multi-dataset benchmark config and return the combined dataframe."""
    global_output = Path(global_cfg.get("output", "results/benchmark_multi.parquet"))
    global_output.parent.mkdir(parents=True, exist_ok=True)

    datasets_cfg: List[Dict[str, Any]] = global_cfg.get("datasets", [])
    if dataset_filter is not None:
        allowed = set(dataset_filter)
        datasets_cfg = [d for d in datasets_cfg if d["name"] in allowed]

    if not datasets_cfg:
        raise ValueError("No datasets to run. Check --datasets filter or config.")

    all_dfs: List[pd.DataFrame] = []

    for i, ds_cfg in enumerate(datasets_cfg):
        ds_name = ds_cfg["name"]
        print(f"\n{'=' * 60}")
        print(f"[DATASET {i + 1}/{len(datasets_cfg)}]  {ds_name}")
        print(f"{'=' * 60}")

        cfg = _build_dataset_cfg(global_cfg, ds_cfg, global_output)
        result = run_single_dataset(cfg)
        df = result.to_dataframe()
        all_dfs.append(df)
        print(f"[INFO] {ds_name}: {len(df)} rows, {df['success'].mean():.1%} valid")

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

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark counterfactual methods from a single- or multi-dataset config."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", default=None, help="Override output Parquet path.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--n_queries", type=int, default=None, help="Override number of test queries.")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Run only these methods (space-separated names).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="For multi-dataset configs, run only these datasets by name.",
    )
    args = parser.parse_args()
    cfg = _apply_cli_overrides(read_yaml(args.config), args)
    if "datasets" in cfg:
        run_multi_dataset(cfg, dataset_filter=args.datasets)
    else:
        if args.datasets is not None:
            raise SystemExit("--datasets is only supported for multi-dataset benchmark configs.")
        run_single_dataset(cfg)


if __name__ == "__main__":
    main()
