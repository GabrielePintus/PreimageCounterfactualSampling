#!/usr/bin/env python3
"""Benchmark counterfactual methods from a single- or multi-dataset config.

Usage:
    python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml
    python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml --output results/run2.parquet
    python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml --seed 123 --n_queries 200
    python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml --methods dice face nearest_neighbor
    python scripts/benchmark.py --config configs/benchmarks/final_benchmark.yaml --datasets adult compas

Output:
    Single-dataset config:
        A .parquet file (flat, for notebooks and analysis).
    Multi-dataset config:
        One per-dataset .parquet file plus one combined .parquet file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import sys
import threading
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil
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
from counterfactuals.core.base_classes import CounterfactualResult
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
    def _manual_load(target_map_location: str):
        import torch

        ckpt = torch.load(checkpoint, map_location=target_map_location, weights_only=False)
        hparams = dict(ckpt.get("hyper_parameters", {}))
        hparams.pop("_class_path", None)
        hparams.pop("_instantiator", None)
        hparams.pop("model", None)

        lit = lit_cls(model=backbone, **hparams)
        state_dict = ckpt.get("state_dict", {})
        lit.load_state_dict(state_dict, strict=True)
        return lit

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
    except TypeError as exc:
        msg = str(exc)
        if "object() takes no arguments" in msg:
            print(
                "[WARNING] Lightning checkpoint instantiation failed via CLI metadata. "
                "Falling back to direct state_dict load."
            )
            try:
                return _manual_load(map_location)
            except RuntimeError as manual_exc:
                manual_msg = str(manual_exc)
                if "tagged with gpu" in manual_msg and map_location != "cpu":
                    print(
                        "[WARNING] Manual checkpoint load hit legacy 'gpu' storage tag. "
                        "Retrying load with map_location=cpu."
                    )
                    return _manual_load("cpu")
                raise
        raise


def _infer_tabular_classifier_dims_from_checkpoint(checkpoint: str) -> tuple[list[int], int]:
    """Infer TabularClassifier hidden dims and class count from a checkpoint state dict."""
    import torch
    from models.classifiers import infer_tabular_classifier_dims_from_state_dict

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", {})
    return infer_tabular_classifier_dims_from_state_dict(state_dict)


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


def subsample_train_per_class(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly cap the training set to at most ``max_per_class`` rows per label."""
    max_per_class = int(max_per_class)
    if max_per_class <= 0:
        raise ValueError("sampling.n_train_per_class must be positive")

    y_arr = np.asarray(y_train, dtype=np.int64)
    selected_parts: list[np.ndarray] = []
    changed = False
    for cls in np.unique(y_arr):
        idx = np.where(y_arr == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
            changed = True
        selected_parts.append(idx)

    if not changed:
        return x_train, y_train

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)
    counts = {int(cls): int(np.sum(y_arr[selected] == cls)) for cls in np.unique(y_arr)}
    print(f"[INFO] Per-class random subsampling: {len(x_train)} -> {len(selected)} samples ({counts}).")
    return x_train[selected], y_train[selected]


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


class ResourceMonitor:
    """Background process RSS sampler for benchmark instrumentation."""

    def __init__(self, enabled: bool = False, interval_s: float = 1.0):
        self.enabled = bool(enabled)
        self.interval_s = float(interval_s)
        self.samples: List[Dict[str, float]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._process = psutil.Process(os.getpid()) if self.enabled else None
        self._start_t: Optional[float] = None

    def _sample_once(self) -> None:
        if not self.enabled or self._process is None:
            return
        rss_mb = float(self._process.memory_info().rss) / (1024.0 ** 2)
        t_s = 0.0 if self._start_t is None else float(time.monotonic() - self._start_t)
        self.samples.append({
            "t_s": round(t_s, 6),
            "rss_mb": round(rss_mb, 6),
        })

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._sample_once()

    def start(self) -> None:
        if not self.enabled:
            return
        self._stop_event.clear()
        self._start_t = time.monotonic()
        self._sample_once()
        self._thread = threading.Thread(target=self._run, name="benchmark-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 2.0, 0.1))
            self._thread = None
        self._sample_once()

    def summarize(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"ram_monitor_enabled": False}
        if not self.samples:
            self._sample_once()

        rss = np.asarray([sample["rss_mb"] for sample in self.samples], dtype=np.float64)
        return {
            "ram_monitor_enabled": True,
            "ram_monitor_interval_s": float(self.interval_s),
            "ram_rss_mb_start": float(rss[0]),
            "ram_rss_mb_end": float(rss[-1]),
            "ram_rss_mb_peak": float(np.max(rss)),
            "ram_rss_mb_mean": float(np.mean(rss)),
            "ram_rss_mb_min": float(np.min(rss)),
            "ram_rss_mb_n_samples": int(len(self.samples)),
            "ram_trace": list(self.samples),
        }


# ---------------------------------------------------------------------------
# Dataset metadata helper
# ---------------------------------------------------------------------------

def _get_tabular_spec(dataset_name: str):
    """Return the shared tabular dataset spec when one exists."""
    try:
        return get_tabular_dataset_spec(dataset_name)
    except KeyError:
        return None


def _normalize_feature_name_list(raw_features: Any, *, param_name: str) -> list[str]:
    if raw_features is None:
        return []
    if isinstance(raw_features, str):
        features = [raw_features]
    else:
        try:
            features = list(raw_features)
        except TypeError as exc:
            raise ValueError(
                f"{param_name} must be a string or a sequence of strings"
            ) from exc

    normalized: list[str] = []
    seen: set[str] = set()
    for feature in features:
        if not isinstance(feature, str):
            raise ValueError(f"{param_name} entries must be strings")
        name = feature.strip()
        if not name:
            raise ValueError(f"{param_name} entries must be non-empty strings")
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return normalized


def _normalize_immutable_features(raw_features: Any) -> list[str]:
    return _normalize_feature_name_list(raw_features, param_name="immutable_features")


def _resolve_immutable_feature_dims(
    dataset_name: str,
    immutable_features: Any,
    preprocessing_name: str = "identity",
) -> tuple[Optional[np.ndarray], list[str]]:
    features = _normalize_immutable_features(immutable_features)
    if not features:
        return None, []
    if preprocessing_name != "identity":
        raise ValueError(
            "immutable_features are only supported with identity preprocessing "
            f"(got preprocessing={preprocessing_name!r})."
        )

    spec = _get_tabular_spec(dataset_name)
    if spec is None:
        raise ValueError(
            f"immutable_features require a tabular dataset spec; none found for {dataset_name!r}."
        )

    feature_to_slice = dict(zip(spec.feature_names, spec.feature_slices))
    unknown = [feature for feature in features if feature not in feature_to_slice]
    if unknown:
        valid = ", ".join(spec.feature_names)
        missing = ", ".join(unknown)
        raise ValueError(
            f"Unknown immutable feature(s) for dataset {dataset_name!r}: {missing}. "
            f"Valid features are: {valid}."
        )

    dims: list[int] = []
    for feature in features:
        start, end = feature_to_slice[feature]
        dims.extend(range(int(start), int(end)))
    return np.unique(np.asarray(dims, dtype=np.int64)), features


def _normalize_directional_feature_lists(
    nondecreasing_features: Any,
    nonincreasing_features: Any,
) -> tuple[list[str], list[str]]:
    nondecreasing = _normalize_feature_name_list(
        nondecreasing_features,
        param_name="nondecreasing_features",
    )
    nonincreasing = _normalize_feature_name_list(
        nonincreasing_features,
        param_name="nonincreasing_features",
    )
    overlap = sorted(set(nondecreasing) & set(nonincreasing))
    if overlap:
        names = ", ".join(overlap)
        raise ValueError(
            "Directional constraints cannot require the same feature to be both "
            f"nondecreasing and nonincreasing: {names}."
        )
    return nondecreasing, nonincreasing


def _resolve_directional_feature_dims(
    dataset_name: str,
    nondecreasing_features: Any,
    nonincreasing_features: Any,
    preprocessing_name: str = "identity",
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], list[str], list[str]]:
    nondecreasing, nonincreasing = _normalize_directional_feature_lists(
        nondecreasing_features,
        nonincreasing_features,
    )
    if not nondecreasing and not nonincreasing:
        return None, None, [], []
    if preprocessing_name != "identity":
        raise ValueError(
            "nondecreasing_features/nonincreasing_features are only supported "
            f"with identity preprocessing (got preprocessing={preprocessing_name!r})."
        )

    spec = _get_tabular_spec(dataset_name)
    if spec is None:
        raise ValueError(
            "Directional feature constraints require a tabular dataset spec; "
            f"none found for {dataset_name!r}."
        )

    feature_to_slice = dict(zip(spec.feature_names, spec.feature_slices))
    feature_to_type = dict(zip(spec.feature_names, spec.input_types))
    requested = nondecreasing + nonincreasing
    unknown = [feature for feature in requested if feature not in feature_to_slice]
    if unknown:
        valid = ", ".join(spec.feature_names)
        missing = ", ".join(unknown)
        raise ValueError(
            f"Unknown directional feature(s) for dataset {dataset_name!r}: {missing}. "
            f"Valid features are: {valid}."
        )

    categorical = [feature for feature in requested if feature_to_type[feature] != "numerical"]
    if categorical:
        names = ", ".join(categorical)
        raise ValueError(
            "Directional constraints only support numerical original features. "
            f"Categorical feature(s) should use immutable_features instead: {names}."
        )

    def _dims_for(features: list[str]) -> Optional[np.ndarray]:
        dims: list[int] = []
        for feature in features:
            start, end = feature_to_slice[feature]
            if int(end) - int(start) != 1:
                raise ValueError(
                    "Directional constraints only support single-coordinate "
                    f"numerical features; {feature!r} maps to slice ({start}, {end})."
                )
            dims.append(int(start))
        if not dims:
            return None
        return np.unique(np.asarray(dims, dtype=np.int64))

    return _dims_for(nondecreasing), _dims_for(nonincreasing), nondecreasing, nonincreasing


def _attach_feature_constraints(
    *,
    method_name: str,
    method_params: Dict[str, Any],
    dataset_name: str,
    preprocessing_name: str,
) -> Dict[str, Any]:
    """Resolve YAML feature constraints into method-level encoded dimensions."""
    immutable_features = _normalize_immutable_features(method_params.get("immutable_features"))
    nondecreasing_features, nonincreasing_features = _normalize_directional_feature_lists(
        method_params.get("nondecreasing_features"),
        method_params.get("nonincreasing_features"),
    )
    if not immutable_features and not nondecreasing_features and not nonincreasing_features:
        return method_params

    supported_methods = {"certcf", "face", "nearest_neighbor", "growing_spheres"}
    if method_name not in supported_methods:
        supported = ", ".join(sorted(supported_methods))
        raise ValueError(
            f"Feature constraints are not supported for method {method_name!r}. "
            f"Supported methods are: {supported}."
        )

    constrained_params = dict(method_params)
    if immutable_features:
        fixed_dims, resolved_features = _resolve_immutable_feature_dims(
            dataset_name=dataset_name,
            immutable_features=immutable_features,
            preprocessing_name=preprocessing_name,
        )
        constrained_params["fixed_dims"] = fixed_dims
        constrained_params["immutable_features"] = resolved_features
    if nondecreasing_features or nonincreasing_features:
        inc_dims, dec_dims, inc_features, dec_features = _resolve_directional_feature_dims(
            dataset_name=dataset_name,
            nondecreasing_features=nondecreasing_features,
            nonincreasing_features=nonincreasing_features,
            preprocessing_name=preprocessing_name,
        )
        constrained_params["nondecreasing_dims"] = inc_dims
        constrained_params["nonincreasing_dims"] = dec_dims
        constrained_params["nondecreasing_features"] = inc_features
        constrained_params["nonincreasing_features"] = dec_features
    return constrained_params


def _attach_immutable_feature_constraints(
    *,
    method_name: str,
    method_params: Dict[str, Any],
    dataset_name: str,
    preprocessing_name: str,
) -> Dict[str, Any]:
    """Backward-compatible wrapper for tests and older internal callers."""
    return _attach_feature_constraints(
        method_name=method_name,
        method_params=method_params,
        dataset_name=dataset_name,
        preprocessing_name=preprocessing_name,
    )


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
    metadata: Optional[Dict[str, Any]] = None,
    method_success: Optional[bool] = None,
    target_reached: Optional[bool] = None,
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
        method_success=method_success,
        target_reached=target_reached,
        target_class=(None if target_class is None else int(target_class)),
        metadata=dict(metadata or {}),
    )


def _persist_completed_benchmark_result(
    benchmark_result: BenchmarkResult,
    output_path: Path,
    *,
    reason: str,
    base_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Persist completed method runs only.

    This helper is intentionally called only after a MethodResult has been fully
    appended.  If the process is interrupted during a later method/query, the
    parquet therefore contains only completed method runs, never the active
    partially computed one.
    """
    new_df = benchmark_result.to_dataframe()
    frames = []
    if base_df is not None and not base_df.empty:
        frames.append(base_df)
    if not new_df.empty:
        frames.append(new_df)
    if not frames:
        print(f"\n[INFO] No completed benchmark rows to save ({reason}).")
        return None
    df = pd.concat(frames, ignore_index=True)
    _write_parquet_atomic(df, output_path)
    print(f"\n[INFO] Saved benchmark parquet to {output_path} ({reason})")
    return df


def _write_parquet_atomic(df: pd.DataFrame, output_path: Path) -> None:
    tmp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    df.to_parquet(tmp_path, index=False, compression="gzip")
    os.replace(tmp_path, output_path)


def _json_normalized(value: Any) -> str:
    return json.dumps(BenchmarkResult._json_safe_value(value or {}), sort_keys=True)


def _task_counts_frame(query_indices: np.ndarray, target_classes: np.ndarray) -> pd.Series:
    task_df = pd.DataFrame({
        "query_idx": np.asarray(query_indices, dtype=np.int64),
        "target_class": np.asarray(target_classes, dtype=np.int64),
    })
    return task_df.value_counts(sort=False).sort_index()


def _existing_run_is_complete(
    grp: pd.DataFrame,
    *,
    expected_task_counts: pd.Series,
    expected_params: Dict[str, Any],
) -> bool:
    if len(grp) != int(expected_task_counts.sum()):
        return False
    required_cols = {"query_idx", "target_class", "params"}
    if not required_cols.issubset(grp.columns):
        return False
    try:
        observed = pd.DataFrame({
            "query_idx": grp["query_idx"].to_numpy(dtype=np.int64),
            "target_class": grp["target_class"].to_numpy(dtype=np.int64),
        })
    except (TypeError, ValueError):
        return False
    observed_task_counts = observed.value_counts(sort=False).sort_index()
    if not observed_task_counts.equals(expected_task_counts):
        return False

    existing_params_raw = grp["params"].iloc[0]
    try:
        existing_params = (
            json.loads(existing_params_raw)
            if isinstance(existing_params_raw, str)
            else existing_params_raw
        )
    except (json.JSONDecodeError, TypeError):
        return False
    return _json_normalized(existing_params) == _json_normalized(expected_params)


def _load_resume_base_df(
    output_path: Path,
    *,
    methods_cfg: List[Dict[str, Any]],
    query_indices: np.ndarray,
    target_classes: np.ndarray,
    force_redo: bool,
) -> tuple[Optional[pd.DataFrame], set[str]]:
    if force_redo:
        if output_path.exists():
            print(f"[INFO] --force set: ignoring existing benchmark parquet at {output_path}")
        return None, set()
    if not output_path.exists():
        return None, set()

    try:
        existing_df = pd.read_parquet(output_path)
    except Exception as exc:
        print(f"[WARNING] Could not read existing benchmark parquet {output_path}: {exc}")
        return None, set()
    if existing_df.empty:
        return None, set()

    run_column = "run_name" if "run_name" in existing_df.columns else "method"
    expected_task_counts = _task_counts_frame(query_indices, target_classes)
    completed_run_names: set[str] = set()
    current_runs = {
        str(method_cfg.get("run_name", method_cfg["name"])): dict(method_cfg.get("params") or {})
        for method_cfg in methods_cfg
    }

    for run_name, expected_params in current_runs.items():
        grp = existing_df[existing_df[run_column].astype(str).eq(run_name)]
        if grp.empty:
            continue
        if _existing_run_is_complete(
            grp,
            expected_task_counts=expected_task_counts,
            expected_params=expected_params,
        ):
            completed_run_names.add(run_name)

    if not completed_run_names:
        print(f"[INFO] Existing parquet found at {output_path}, but no complete matching runs can be resumed.")
        return None, set()

    base_df = existing_df[existing_df[run_column].astype(str).isin(completed_run_names)].copy()
    skipped = ", ".join(sorted(completed_run_names))
    print(
        f"[INFO] Resuming from {output_path}: "
        f"{len(completed_run_names)} complete method run(s) will be skipped ({skipped})."
    )
    return base_df, completed_run_names


def _print_flat_benchmark_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nBENCHMARK SUMMARY\n(no completed rows)")
        return
    try:
        from tabulate import tabulate
    except ImportError:
        print("[WARNING] tabulate not installed; skipping summary table.")
        return

    rows = []
    for run_name, grp in df.groupby("run_name", sort=False):
        ok = grp[grp["success"].astype(bool)]
        n_total = len(grp)
        n_ok = len(ok)
        validity = 100.0 * n_ok / n_total if n_total else float("nan")
        rows.append([
            run_name,
            f"{validity:.1f}%",
            f"{float(ok['l2_distance'].mean()):.3f}" if n_ok else "nan",
            f"{float(ok['l1_distance'].mean()):.3f}" if n_ok else "nan",
            f"{100.0 * float(ok['l0_sparsity'].mean()):.1f}%" if n_ok else "nan",
            f"{float(grp['build_time_s'].iloc[0]):.2f}" if "build_time_s" in grp else "nan",
            f"{float(grp['runtime_s'].mean()):.3f}" if "runtime_s" in grp else "nan",
            n_total - n_ok,
        ])

    print("\nBENCHMARK SUMMARY")
    print(tabulate(
        rows,
        headers=["method", "validity%", "l2_mean", "l1_mean", "sparsity%",
                 "build_s", "query_s", "n_failed"],
        tablefmt="simple",
    ))


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
    if "query_indices" in sampling_cfg:
        query_indices = np.asarray(sampling_cfg["query_indices"], dtype=np.int64)
        if query_indices.ndim != 1:
            raise ValueError("sampling.query_indices must be a one-dimensional list of test indices.")
        if len(query_indices) == 0:
            raise ValueError("sampling.query_indices must not be empty.")
        if np.any(query_indices < 0) or np.any(query_indices >= len(x_test)):
            raise ValueError(
                "sampling.query_indices contains indices outside the test-set range "
                f"[0, {len(x_test) - 1}]."
            )
        query_indices = np.asarray(sorted(dict.fromkeys(query_indices.tolist())), dtype=np.int64)
        y_orig = model.predict(x_test[query_indices])
        return query_indices, y_test[query_indices], y_orig, None

    n_queries = int(sampling_cfg.get("n_queries", len(x_test)))
    if dataset_name != "mnist":
        n_queries = min(n_queries, len(x_test))
        query_indices = np.sort(rng.choice(len(x_test), size=n_queries, replace=False))
        y_orig = model.predict(x_test[query_indices])
        return query_indices, y_test[query_indices], y_orig, None

    target_policy = str(sampling_cfg.get("target_policy", "all_other_classes")).lower()
    n_classes = int(model.predict_proba(x_test[:1]).shape[1])
    total_query_limit = int(sampling_cfg["n_queries"]) if "n_queries" in sampling_cfg else None
    if total_query_limit is not None and total_query_limit <= 0:
        raise ValueError("sampling.n_queries must be positive")

    per_class = int(sampling_cfg.get("balanced_per_class", 10))
    if total_query_limit is not None:
        n_source_classes = max(1, len(np.unique(y_test)))
        tasks_per_source = max(1, n_classes - 1)
        required_per_class = int(
            np.ceil(total_query_limit / (n_source_classes * tasks_per_source))
        )
        per_class = max(per_class, required_per_class)

    source_indices = _sample_balanced_indices(y=y_test, per_class=per_class, rng=rng)
    if source_indices.size == 0:
        raise ValueError("No MNIST source queries were selected.")

    source_preds = model.predict(x_test[source_indices])
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

    if total_query_limit is not None:
        if len(task_indices) < total_query_limit:
            raise ValueError(
                f"MNIST sampling produced only {len(task_indices)} source-target tasks, "
                f"fewer than sampling.n_queries={total_query_limit}."
            )

        # Balance the final task budget across source classes. For 1,000 MNIST
        # queries this selects exactly 100 tasks per source class.
        task_true_array = np.asarray(task_true, dtype=np.int64)
        source_classes = np.unique(task_true_array)
        base_quota, remainder = divmod(total_query_limit, len(source_classes))
        remainder_classes = set(
            rng.choice(source_classes, size=remainder, replace=False).tolist()
        )
        selected_parts: List[np.ndarray] = []
        for source_class in source_classes:
            candidates = np.where(task_true_array == source_class)[0]
            quota = base_quota + int(source_class in remainder_classes)
            take = min(quota, len(candidates))
            selected_parts.append(rng.choice(candidates, size=take, replace=False))

        selected = np.concatenate(selected_parts)
        if len(selected) < total_query_limit:
            remaining = np.setdiff1d(
                np.arange(len(task_indices), dtype=np.int64),
                selected,
                assume_unique=False,
            )
            extra = rng.choice(
                remaining,
                size=total_query_limit - len(selected),
                replace=False,
            )
            selected = np.concatenate([selected, extra])
        selected = np.sort(selected)
        task_indices = np.asarray(task_indices, dtype=np.int64)[selected].tolist()
        task_true = task_true_array[selected].tolist()
        task_orig = np.asarray(task_orig, dtype=np.int64)[selected].tolist()
        task_targets = np.asarray(task_targets, dtype=np.int64)[selected].tolist()

    return (
        np.asarray(task_indices, dtype=np.int64),
        np.asarray(task_true, dtype=np.int64),
        np.asarray(task_orig, dtype=np.int64),
        np.asarray(task_targets, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# CertCFAtlas builder
# ---------------------------------------------------------------------------

def _build_mnist_backbone(architecture: str | None = None, num_classes: int = 10):
    """Construct the configured MNIST classifier, preserving the legacy default."""
    from models.classifiers import LeNet5Classifier, MNISTClassifier

    normalized = "mnist" if architecture is None else str(architecture).strip().lower()
    if normalized in {"mnist", "default", "mnist_classifier"}:
        return MNISTClassifier(num_classes=num_classes)
    if normalized == "lenet5":
        return LeNet5Classifier(num_classes=num_classes)
    raise ValueError(
        "Unsupported MNIST architecture "
        f"{architecture!r}; expected one of {{'mnist', 'lenet5'}}."
    )


def _build_certcf_method(
    params: Dict[str, Any],
    model_params: Dict[str, Any],
    dataset_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_queries: np.ndarray,
    seed: int,
    preprocessing_name: str = "identity",
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
    _distance_norm_raw = params.get("distance_norm", None)
    if _distance_norm_raw is None:
        distance_norm = norm
    else:
        distance_norm = (
            np.inf if str(_distance_norm_raw).lower() in ("inf", "infinity") else int(_distance_norm_raw)
        )
    lirpa_method = str(params.get("lirpa_method", "backward"))
    delta = float(params.get("delta", 0.0))
    _robust_norm_raw = params.get("robust_norm", None)
    if _robust_norm_raw is None:
        robust_norm = None
    else:
        robust_norm = (
            np.inf if str(_robust_norm_raw).lower() in ("inf", "infinity") else int(_robust_norm_raw)
        )
    query_method = str(params.get("query_method", "sorted")).lower()
    if query_method not in {"sorted", "bvh", "nearest_anchor"}:
        raise ValueError(
            f"certcf.query_method must be one of {{'sorted', 'bvh', 'nearest_anchor'}}, got {query_method!r}"
        )
    query_k_candidates = int(params.get("query_k_candidates", 1))
    if query_k_candidates <= 0:
        raise ValueError("certcf.query_k_candidates must be positive")
    batch_size = int(params.get("batch_size", 256))
    atlas_subsample_method = str(
        params.get("atlas_subsample_method", params.get("subsample_method", "kmedoids"))
    ).lower()
    if atlas_subsample_method not in {"random", "boundary_random", "kmedoids", "bandit_kmedoids", "fps", "kmeans", "density_flat_kmedoids"}:
        raise ValueError(
            f"certcf.atlas_subsample_method must be one of {{'random', 'boundary_random', 'kmedoids', 'bandit_kmedoids', 'fps', 'kmeans', 'density_flat_kmedoids'}}, "
            f"got {atlas_subsample_method!r}"
        )
    boundary_beta = float(params.get("atlas_boundary_beta", params.get("boundary_beta", 0.5)))
    if not (0.0 <= boundary_beta <= 1.0):
        raise ValueError(f"certcf.boundary_beta must be in [0, 1], got {boundary_beta!r}")
    atlas_subsample_space = str(params.get("atlas_subsample_space", "input")).lower()
    if atlas_subsample_space not in {"input", "latent"}:
        raise ValueError(
            f"certcf.atlas_subsample_space must be one of {{'input', 'latent'}}, "
            f"got {atlas_subsample_space!r}"
        )
    solver_maxiter = int(params.get("solver_maxiter", 500))
    query_parallelism = int(params.get("query_parallelism", 1))
    if query_parallelism <= 0:
        raise ValueError("certcf.query_parallelism must be positive")
    candidate_parallelism = int(params.get("candidate_parallelism", 1))
    if candidate_parallelism <= 0:
        raise ValueError("certcf.candidate_parallelism must be positive")
    if query_parallelism > 1 and candidate_parallelism > 1:
        raise ValueError(
            "certcf.query_parallelism and certcf.candidate_parallelism cannot both exceed 1"
        )
    candidate_parallel_backend = str(
        params.get("candidate_parallel_backend", "thread")
    ).lower()
    if candidate_parallel_backend not in {"thread", "process"}:
        raise ValueError(
            "certcf.candidate_parallel_backend must be one of {'thread', 'process'}"
        )
    cvxpy_solvers = deepcopy(params["cvxpy_solvers"]) if "cvxpy_solvers" in params else None
    cvxpy_solver_options = deepcopy(params["cvxpy_solver_options"]) if "cvxpy_solver_options" in params else None
    cvxpy_accept_statuses = deepcopy(params["cvxpy_accept_statuses"]) if "cvxpy_accept_statuses" in params else None
    input_bounds = deepcopy(params["input_bounds"]) if "input_bounds" in params else None
    classification_margin = float(params.get("classification_margin", 0.0))
    adaptive_eps = bool(params.get("adaptive_eps", False))
    adaptive_eps_shrink_factor = float(params.get("adaptive_eps_shrink_factor", 0.5))
    adaptive_eps_max_shrinks = int(params.get("adaptive_eps_max_shrinks", 8))
    adaptive_eps_min = float(params.get("adaptive_eps_min", 1.0e-6))
    adaptive_eps_center_tol = float(params.get("adaptive_eps_center_tol", 1.0e-6))
    adaptive_eps_binary_search_steps = int(params.get("adaptive_eps_binary_search_steps", 0))
    sparsity_penalty = str(params.get("sparsity_penalty", "none"))
    sparsity_lambda = float(params.get("sparsity_lambda", 0.0))
    sparsity_reweight_iters = int(params.get("sparsity_reweight_iters", 0))
    sparsity_eps = float(params.get("sparsity_eps", 1.0e-3))
    sparsity_group_ohe = bool(params.get("sparsity_group_ohe", True))
    ohe_decode_mode = params["ohe_decode_mode"] if "ohe_decode_mode" in params else None
    decode_beam_width = params["decode_beam_width"] if "decode_beam_width" in params else None
    decode_beam_branch_top_k = params["decode_beam_branch_top_k"] if "decode_beam_branch_top_k" in params else None
    decode_beam_max_solver_calls = (
        params["decode_beam_max_solver_calls"] if "decode_beam_max_solver_calls" in params else None
    )
    # Architecture params come from the shared model config, not method params.
    dataset_module = str(model_params.get("dataset_module", dataset_name))
    hidden_dims = list(model_params.get("hidden_dims", [32, 8]))
    dropout = float(model_params.get("dropout", 0.2))
    fixed_dims, immutable_features = _resolve_immutable_feature_dims(
        dataset_name=dataset_module,
        immutable_features=params.get("immutable_features"),
        preprocessing_name=preprocessing_name,
    )
    (
        nondecreasing_dims,
        nonincreasing_dims,
        nondecreasing_features,
        nonincreasing_features,
    ) = _resolve_directional_feature_dims(
        dataset_name=dataset_module,
        nondecreasing_features=params.get("nondecreasing_features"),
        nonincreasing_features=params.get("nonincreasing_features"),
        preprocessing_name=preprocessing_name,
    )

    z_train = x_train
    z_queries = x_queries
    cat_slices = None

    if dataset_module == "mnist":
        num_classes = int(model_params.get("num_classes", 10))
        architecture = model_params.get("architecture")
        backbone = _build_mnist_backbone(
            architecture=architecture,
            num_classes=num_classes,
        )
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
    certcf_kwargs = dict(
        model=atlas_model,
        norm=norm,
        distance_norm=distance_norm,
        lirpa_method=lirpa_method,
        delta=delta,
        robust_norm=robust_norm,
        eps_strategy=eps_strategy,
        batch_size=batch_size,
        ohe_slices=cat_slices,
        cnn=cnn,
        default_query_method=query_method,
        query_k_candidates=query_k_candidates,
        solver_maxiter=solver_maxiter,
        query_parallelism=query_parallelism,
        candidate_parallelism=candidate_parallelism,
        candidate_parallel_backend=candidate_parallel_backend,
        k_per_class=k_per_class,
        subsample_method=atlas_subsample_method,
        subsample_space=atlas_subsample_space,
        boundary_beta=boundary_beta,
        classification_margin=classification_margin,
        fixed_dims=fixed_dims,
        immutable_features=immutable_features,
        nondecreasing_dims=nondecreasing_dims,
        nonincreasing_dims=nonincreasing_dims,
        nondecreasing_features=nondecreasing_features,
        nonincreasing_features=nonincreasing_features,
        adaptive_eps=adaptive_eps,
        adaptive_eps_shrink_factor=adaptive_eps_shrink_factor,
        adaptive_eps_max_shrinks=adaptive_eps_max_shrinks,
        adaptive_eps_min=adaptive_eps_min,
        adaptive_eps_center_tol=adaptive_eps_center_tol,
        adaptive_eps_binary_search_steps=adaptive_eps_binary_search_steps,
        sparsity_penalty=sparsity_penalty,
        sparsity_lambda=sparsity_lambda,
        sparsity_reweight_iters=sparsity_reweight_iters,
        sparsity_eps=sparsity_eps,
        sparsity_group_ohe=sparsity_group_ohe,
        random_seed=seed,
    )
    if cvxpy_solvers is not None:
        certcf_kwargs["cvxpy_solvers"] = cvxpy_solvers
    if cvxpy_solver_options is not None:
        certcf_kwargs["cvxpy_solver_options"] = cvxpy_solver_options
    if cvxpy_accept_statuses is not None:
        certcf_kwargs["cvxpy_accept_statuses"] = cvxpy_accept_statuses
    if input_bounds is not None:
        certcf_kwargs["input_bounds"] = input_bounds
    if ohe_decode_mode is not None:
        certcf_kwargs["ohe_decode_mode"] = ohe_decode_mode
    if decode_beam_width is not None:
        certcf_kwargs["decode_beam_width"] = decode_beam_width
    if decode_beam_branch_top_k is not None:
        certcf_kwargs["decode_beam_branch_top_k"] = decode_beam_branch_top_k
    if decode_beam_max_solver_calls is not None:
        certcf_kwargs["decode_beam_max_solver_calls"] = decode_beam_max_solver_calls

    atlas_method = CertCF(**certcf_kwargs)
    atlas_method.fit(x_train=z_train, y_train=y_train)

    return atlas_method, atlas_model, model, z_train, z_queries, None


# ---------------------------------------------------------------------------
# PyTorch checkpoint model loader
# ---------------------------------------------------------------------------

def _build_torch_model_from_checkpoint(
    checkpoint: str,
    device: str = "cpu",
    dataset_module: str = "adult",
    hidden_dims: list | None = None,
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

    inferred_hidden_dims, inferred_num_classes = _infer_tabular_classifier_dims_from_checkpoint(checkpoint)
    resolved_hidden_dims = list(hidden_dims) if hidden_dims is not None else inferred_hidden_dims

    def _make_backbone(current_hidden_dims: list[int]) -> TabularClassifier:
        return TabularClassifier(
            input_types=list(spec.input_types),
            cardinalities=list(spec.cardinalities),
            hidden_dims=current_hidden_dims,
            num_classes=inferred_num_classes,
            dropout=dropout,
        )

    backbone = _make_backbone(resolved_hidden_dims)
    try:
        lit = _load_lit_checkpoint_resilient(
            LitClassifier, checkpoint, backbone, map_location=device
        )
    except RuntimeError as exc:
        if resolved_hidden_dims != inferred_hidden_dims:
            print(
                "[INFO] Retrying checkpoint load with hidden_dims inferred from checkpoint: "
                f"{inferred_hidden_dims}"
            )
            backbone = _make_backbone(inferred_hidden_dims)
            lit = _load_lit_checkpoint_resilient(
                LitClassifier, checkpoint, backbone, map_location=device
            )
        else:
            raise exc
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
    architecture: str | None = None,
) -> "TorchModelWrapper":
    """Load a configured MNIST classifier and optionally wrap flat inputs."""
    from counterfactuals.models.torch_model import TorchModelWrapper
    from training.lit_classifier import LitClassifier

    requested_device = device
    device = _normalize_torch_device(device)
    if str(requested_device).strip().lower() != device:
        print(f"[INFO] model device normalized: {requested_device!r} -> {device!r}")

    backbone = _build_mnist_backbone(
        architecture=architecture,
        num_classes=num_classes,
    )
    lit = _load_lit_checkpoint_resilient(
        LitClassifier, checkpoint, backbone, map_location=device
    )
    net = lit.model.eval().to(device)
    return TorchModelWrapper(model=net, device=device, input_shape=input_shape)


# ---------------------------------------------------------------------------
# Grid search expansion
# ---------------------------------------------------------------------------

_NON_SWEEP_LIST_PARAMS_BY_METHOD: Dict[str, set[str]] = {
    "certcf": {
        "cvxpy_solvers",
        "input_bounds",
        "immutable_features",
        "nondecreasing_features",
        "nonincreasing_features",
    },
    "face": {
        "immutable_features",
        "nondecreasing_features",
        "nonincreasing_features",
    },
    "nearest_neighbor": {
        "immutable_features",
        "nondecreasing_features",
        "nonincreasing_features",
    },
    "growing_spheres": {
        "immutable_features",
        "nondecreasing_features",
        "nonincreasing_features",
    },
}


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
        non_sweep_list_keys = _NON_SWEEP_LIST_PARAMS_BY_METHOD.get(name, set())

        sweep = {
            k: v for k, v in params.items()
            if isinstance(v, list) and k not in non_sweep_list_keys
        }
        fixed = {
            k: v for k, v in params.items()
            if not isinstance(v, list) or k in non_sweep_list_keys
        }

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
    candidate_parallelism = getattr(args, "candidate_parallelism", None)
    candidate_parallel_backend = getattr(args, "candidate_parallel_backend", None)
    if candidate_parallelism is not None or candidate_parallel_backend is not None:
        for method in cfg.get("methods", []):
            if method.get("name") != "certcf":
                continue
            params = method.setdefault("params", {})
            if candidate_parallelism is not None:
                params["candidate_parallelism"] = int(candidate_parallelism)
            if candidate_parallel_backend is not None:
                params["candidate_parallel_backend"] = str(candidate_parallel_backend)
    if getattr(args, "force", False):
        cfg["force_redo"] = True
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
        "resource_monitor": deepcopy(global_cfg.get("resource_monitor", {})),
        "force_redo": bool(ds_cfg.get("force_redo", global_cfg.get("force_redo", False))),
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

    # --- Shared train subsampling ---
    # Dataset-level caps are applied before all method-specific fitting/building.
    # CertCF may still apply its own atlas support reduction via k_per_class.
    sampling_cfg = cfg.get("sampling", {})
    x_train, y_train = x_train_full, y_train_full
    if "n_train_per_class" in sampling_cfg:
        x_train, y_train = subsample_train_per_class(
            x_train, y_train, int(sampling_cfg["n_train_per_class"]), rng
        )
    n_train = int(sampling_cfg.get("n_train", len(x_train)))
    x_train, y_train = subsample_train(x_train, y_train, n_train, "random", rng)

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
        architecture = _model_params.get("architecture")
        architecture_label = "mnist" if architecture is None else str(architecture)
        print(
            f"[INFO] Loading MNIST classifier ({architecture_label}) "
            f"from checkpoint: {ckpt}"
        )
        model = _build_mnist_model_from_checkpoint(
            ckpt,
            device=device,
            num_classes=num_classes,
            input_shape=(1, 28, 28),
            architecture=architecture,
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
    y_train_true = np.asarray(y_train, dtype=np.int64)
    y_train_pred_raw = np.asarray(model.predict(x_train), dtype=np.int64)
    y_train_pred_gen = np.asarray(model_for_methods.predict(x_train_gen), dtype=np.int64)
    train_label_agreement = float(np.mean(y_train_pred_raw == y_train_true)) if len(y_train_true) else float("nan")

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
    force_redo = bool(cfg.get("force_redo", False))
    if force_redo and output_path.exists():
        output_path.unlink()
        print(f"[INFO] --force set: removed existing benchmark parquet at {output_path}")
    resource_cfg = cfg.get("resource_monitor") or {}
    ram_monitor_enabled = bool(resource_cfg.get("enabled", False))
    ram_monitor_interval_s = float(resource_cfg.get("interval_s", 1.0))

    methods_cfg = _expand_grid(cfg.get("methods", []))
    print(f"[INFO] {len(methods_cfg)} method runs after grid expansion.")

    benchmark_result = BenchmarkResult(
        dataset=ds_cfg["name"],
        seed=seed,
        x_queries=x_queries,
        y_orig=y_orig_all,
        y_true=y_true_all,
    )
    resume_target_classes = (
        target_classes_all
        if target_classes_all is not None
        else np.asarray([1 - int(y) for y in y_orig_all], dtype=np.int64)
    )
    resume_base_df, completed_run_names = _load_resume_base_df(
        output_path,
        methods_cfg=methods_cfg,
        query_indices=query_indices,
        target_classes=resume_target_classes,
        force_redo=force_redo,
    )

    for method_cfg in methods_cfg:
        method_name: str = method_cfg["name"]        # registry key — selects the implementation
        run_name: str = method_cfg.get("run_name", method_name)  # label used in results
        method_params_raw: Dict[str, Any] = dict(method_cfg.get("params") or {})
        method_params: Dict[str, Any] = dict(method_params_raw)
        if run_name in completed_run_names:
            print(f"\n[METHOD] {run_name} -- already complete, skipping")
            continue
        query_batch_size = 1
        if method_name == "dice":
            query_batch_size = max(1, int(method_params.pop("query_batch_size", 1)))
        elif method_name == "certcf":
            query_batch_size = max(1, int(method_params.get("query_parallelism", 1)))
        print(f"\n[METHOD] {run_name}" + (f" (impl: {method_name})" if run_name != method_name else ""))
        resource_monitor = ResourceMonitor(
            enabled=ram_monitor_enabled,
            interval_s=ram_monitor_interval_s,
        )
        method_metadata: Dict[str, Any] = {
            "ram_monitor_enabled": ram_monitor_enabled,
            "query_batch_size": query_batch_size,
        }
        if ram_monitor_enabled:
            method_metadata["ram_monitor_interval_s"] = ram_monitor_interval_s
        resource_monitor.start()

        # --- Fit / Build ---
        # certcf (CertCFAtlas) operates in embedding space internally but
        # we decode CFs back to raw feature space for fair comparison.
        embed_model = None   # full TabularClassifier (has .decode()); set for certcf only
        embed_device = "cpu"
        build_time_s = 0.0
        fit_label_metadata = {
            "train_label_source": "predicted",
            "train_label_agreement_true": train_label_agreement,
        }
        if method_name == "certcf":
            try:
                _t_build = time.perf_counter()
                method, active_model, embed_model, _, active_queries, _ = _build_certcf_method(
                    params=method_params,
                    model_params=model_cfg.get("params", {}),
                    dataset_name=ds_cfg["name"],
                    x_train=x_train,
                    y_train=y_train_pred_raw,
                    x_queries=x_queries,
                    seed=seed,
                    preprocessing_name=transform_name,
                )
                build_time_s = time.perf_counter() - _t_build
                embed_device = method_params.get("device", "cpu")
                active_y_orig = y_orig_all
                space = "raw"  # CFs are decoded back to raw feature space
                fit_label_metadata["train_label_space"] = "raw"
                fit_label_metadata["train_label_n_classes"] = int(np.unique(y_train_pred_raw).size)
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
                        metadata=dict(fit_label_metadata),
                    )
                    for pos, q_idx in enumerate(query_indices)
                ]
                resource_monitor.stop()
                method_metadata.update(resource_monitor.summarize())
                benchmark_result.method_results.append(MethodResult(
                    method=method_name, run_name=run_name, params=method_params_raw,
                    build_time_s=0.0, space="raw", metadata=method_metadata, query_results=qrs,
                ))
                _persist_completed_benchmark_result(
                    benchmark_result,
                    output_path,
                    reason=f"completed build-failure record for {run_name}",
                    base_df=resume_base_df,
                )
                continue
        else:
            try:
                method_params = _attach_feature_constraints(
                    method_name=method_name,
                    method_params=method_params,
                    dataset_name=ds_cfg["name"],
                    preprocessing_name=transform_name,
                )
                method = registries["method"].create(method_name, model=model_for_methods, random_seed=seed, **method_params)
                _t_build = time.perf_counter()
                method.fit(x_train=x_train_gen, y_train=y_train_pred_gen)
                build_time_s = time.perf_counter() - _t_build
                fit_label_metadata["train_label_space"] = "generation"
                fit_label_metadata["train_label_n_classes"] = int(np.unique(y_train_pred_gen).size)
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
                        metadata=dict(fit_label_metadata),
                    )
                    for pos, q_idx in enumerate(query_indices)
                ]
                resource_monitor.stop()
                method_metadata.update(resource_monitor.summarize())
                benchmark_result.method_results.append(MethodResult(
                    method=method_name, run_name=run_name, params=method_params_raw,
                    build_time_s=0.0, space="gen", metadata=method_metadata, query_results=qrs,
                ))
                _persist_completed_benchmark_result(
                    benchmark_result,
                    output_path,
                    reason=f"completed fit-failure record for {run_name}",
                    base_df=resume_base_df,
                )
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
        candidate_parallelism = int(method_params.get("candidate_parallelism", 1))
        candidate_parallel_backend = str(
            method_params.get("candidate_parallel_backend", "thread")
        ).lower()
        candidate_parallel_warmup = bool(
            method_params.get("candidate_parallel_warmup", True)
        )
        if (
            method_name == "certcf"
            and candidate_parallelism > 1
            and candidate_parallel_backend == "process"
            and candidate_parallel_warmup
            and len(query_indices) > 0
        ):
            warmup_started = time.perf_counter()
            warmup_result = method.generate(
                x=active_queries[0],
                target_class=int(task_targets[0]),
            )
            warmup_time_s = time.perf_counter() - warmup_started
            method_metadata["candidate_parallel_warmup_s"] = float(warmup_time_s)
            method_metadata["candidate_parallel_warmup_success"] = bool(
                warmup_result.success
            )
            print(
                "  [candidate pool warm-up] "
                f"{warmup_time_s:.3f}s, workers={candidate_parallelism}, "
                f"success={bool(warmup_result.success)}"
            )
        pbar = tqdm(total=len(query_indices), desc=f"  {run_name}", unit="query")

        def _append_query_result(
            pos: int,
            q_idx: int,
            target_class: int,
            result: CounterfactualResult,
            runtime_s: float,
        ) -> None:
            x_orig_raw = x_queries[pos]   # raw feature vector, always
            target_class = int(target_class)
            nonlocal n_ok, n_failed
            if result.x_cf is None:
                reason = str(result.metadata.get("reason", "no_counterfactual"))
                query_results.append(_make_failure_query_result(
                    q_idx=int(q_idx),
                    runtime_s=runtime_s,
                    error=reason,
                    target_class=target_class,
                    metadata=result.metadata,
                    method_success=bool(result.success),
                    target_reached=False,
                ))
                n_failed += 1
                return
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

            method_success = bool(result.success)
            target_reached = (y_cf == target_class)
            cf_success = method_success and target_reached
            result_metadata = dict(result.metadata or {})
            result_metadata.update(fit_label_metadata)
            result_metadata["method_success"] = method_success
            result_metadata["target_reached"] = target_reached
            result_metadata["benchmark_success"] = cf_success
            error = None
            if not method_success:
                error = str(
                    result_metadata.get("reason")
                    or result_metadata.get("error")
                    or "method_reported_failure"
                )

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
                method_success=method_success,
                target_reached=target_reached,
                runtime_s=runtime_s,
                error=error,
                l2_distance=l2,
                l1_distance=l1,
                l0_sparsity=l0,
                mad_l1_distance=mad_l1,
                redundancy=redundancy_val,
                target_class=target_class,
                metadata=result_metadata,
            ))
            n_ok += int(cf_success)

        for start in range(0, len(query_indices), query_batch_size):
            end = min(start + query_batch_size, len(query_indices))
            batch_len = end - start
            batch_query_indices = query_indices[start:end]
            batch_queries = active_queries[start:end]
            batch_targets = np.asarray(task_targets[start:end], dtype=np.int64)
            t0 = time.perf_counter()
            try:
                if method_name == "dice" and batch_len > 1:
                    results_batch = _call_with_timeout(
                        lambda: method.generate_batch(x=batch_queries, target_class=batch_targets),
                        timeout_s * batch_len,
                    )
                    runtime_s = time.perf_counter() - t0
                    if len(results_batch) != batch_len:
                        raise ValueError(
                            f"generate_batch returned {len(results_batch)} results for batch_len={batch_len}"
                        )
                    per_query_runtime_s = runtime_s / batch_len
                    for offset, result in enumerate(results_batch):
                        pos = start + offset
                        _append_query_result(
                            pos=pos,
                            q_idx=int(batch_query_indices[offset]),
                            target_class=int(batch_targets[offset]),
                            result=result,
                            runtime_s=per_query_runtime_s,
                        )
                elif method_name == "certcf" and batch_len > 1:
                    results_batch = method.generate_batch(
                        x=batch_queries,
                        target_class=batch_targets,
                        timeout_s_per_query=timeout_s,
                    )
                    runtime_s = time.perf_counter() - t0
                    if len(results_batch) != batch_len:
                        raise ValueError(
                            f"generate_batch returned {len(results_batch)} results for batch_len={batch_len}"
                        )
                    per_query_runtime_s = runtime_s / batch_len
                    for offset, result in enumerate(results_batch):
                        pos = start + offset
                        _append_query_result(
                            pos=pos,
                            q_idx=int(batch_query_indices[offset]),
                            target_class=int(batch_targets[offset]),
                            result=result,
                            runtime_s=per_query_runtime_s,
                        )
                else:
                    result = _call_with_timeout(
                        lambda: method.generate(x=batch_queries[0], target_class=int(batch_targets[0])),
                        timeout_s,
                    )
                    runtime_s = time.perf_counter() - t0
                    _append_query_result(
                        pos=start,
                        q_idx=int(batch_query_indices[0]),
                        target_class=int(batch_targets[0]),
                        result=result,
                        runtime_s=runtime_s,
                    )
            except TimeoutError:
                runtime_s = time.perf_counter() - t0
                per_query_runtime_s = runtime_s / batch_len
                for offset in range(batch_len):
                    query_results.append(_make_failure_query_result(
                        q_idx=int(batch_query_indices[offset]),
                        runtime_s=per_query_runtime_s,
                        error="timeout",
                        target_class=int(batch_targets[offset]),
                        metadata={**fit_label_metadata, "reason": "timeout"},
                    ))
                    n_failed += 1
            except Exception as exc:
                runtime_s = time.perf_counter() - t0
                per_query_runtime_s = runtime_s / batch_len
                for offset in range(batch_len):
                    query_results.append(_make_failure_query_result(
                        q_idx=int(batch_query_indices[offset]),
                        runtime_s=per_query_runtime_s,
                        error=str(exc),
                        target_class=int(batch_targets[offset]),
                        metadata={
                            **fit_label_metadata,
                            "reason": "exception",
                            "exception_type": type(exc).__name__,
                        },
                    ))
                    n_failed += 1

            pbar.update(batch_len)
            pbar.set_postfix(valid=n_ok, failed=n_failed)

        pbar.close()
        if method_name == "certcf":
            close_candidate_pool = getattr(
                getattr(method, "atlas", None),
                "close_candidate_process_pool",
                None,
            )
            if callable(close_candidate_pool):
                close_candidate_pool()

        resource_monitor.stop()
        method_metadata.update(resource_monitor.summarize())
        benchmark_result.method_results.append(MethodResult(
            method=method_name,
            run_name=run_name,
            params=method_params_raw,
            build_time_s=build_time_s,
            space=space,
            metadata=method_metadata,
            query_results=query_results,
        ))
        _persist_completed_benchmark_result(
            benchmark_result,
            output_path,
            reason=f"completed method {run_name}",
            base_df=resume_base_df,
        )

    # --- Save ---
    final_df = _persist_completed_benchmark_result(
        benchmark_result,
        output_path,
        reason="final",
        base_df=resume_base_df,
    )

    if final_df is not None:
        _print_flat_benchmark_summary(final_df)
    else:
        benchmark_result.summary()
    return benchmark_result


def run_multi_dataset(
    global_cfg: Dict[str, Any],
    dataset_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run a multi-dataset benchmark config and return the combined dataframe."""
    global_output = Path(global_cfg.get("output", "results/benchmark_multi.parquet"))
    global_output.parent.mkdir(parents=True, exist_ok=True)
    if bool(global_cfg.get("force_redo", False)) and global_output.exists():
        global_output.unlink()
        print(f"[INFO] --force set: removed existing combined parquet at {global_output}")

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
        per_dataset_output = Path(cfg["output"]["path"])
        if per_dataset_output.exists():
            df = pd.read_parquet(per_dataset_output)
        else:
            df = result.to_dataframe()
        all_dfs.append(df)
        print(f"[INFO] {ds_name}: {len(df)} rows, {df['success'].mean():.1%} valid")
        combined_so_far = pd.concat(all_dfs, ignore_index=True)
        _write_parquet_atomic(combined_so_far, global_output)
        print(
            f"[INFO] Combined results so far ({len(combined_so_far)} rows) "
            f"saved to {global_output}"
        )

    combined = pd.concat(all_dfs, ignore_index=True)
    _write_parquet_atomic(combined, global_output)
    print(f"\n[INFO] Combined results ({len(combined)} rows) saved to {global_output}")

    print("\n[SUMMARY PER DATASET]")
    summary = (
        combined.groupby(["dataset", "run_name"])["success"]
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing result parquet files and recompute all configured runs.",
    )
    parser.add_argument(
        "--candidate-parallelism",
        type=int,
        default=None,
        help="Override the number of concurrent CertCF candidate-projection workers.",
    )
    parser.add_argument(
        "--candidate-parallel-backend",
        choices=["thread", "process"],
        default=None,
        help="Override the CertCF candidate-projection backend.",
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
