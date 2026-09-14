"""Empirical ablation of CertCF's nearest-anchor top-k query heuristic.

The experiment reuses one trained classifier and the deterministic synthetic
32-dimensional dataset from the network-complexity grid.  It constructs and
serializes one certified atlas, evaluates strict nearest-anchor prefixes, and
compares every prefix with the exact optimum over the serialized atlas.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import TensorDataset
from tqdm import tqdm

from certcf import NearestOppositeClassClearanceStrategy
from certcf.atlas import CertCFAtlas
from counterfactuals.methods.certcf import _strip_dropout_modules
from counterfactuals.utils.clustering import select_prototype_indices
from experiments.network_complexity import (
    NetworkComplexityRunner,
    _normalize_device,
    architecture_id,
    parameter_count,
    sha256_file,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "source_config": "configs/experiments/network_complexity_grid.yaml",
        "depth": 3,
        "width": 128,
        "n_queries": 1_000,
        "k_values": [1, 2, 3, 4, 5, 6, 7],
        "absolute_tolerance": 1.0e-5,
        "relative_tolerance": 1.0e-6,
        "within_relative_tolerances": [0.01, 0.05],
        "confidence_level": 0.95,
        "partial_save_every": 25,
    },
    "support": {
        "points_per_true_class": 5_000,
        "anchors_per_predicted_class": 500,
        "sampling": "random",
    },
    "certcf": {
        "device": "auto",
        "eps_alpha": 0.20,
        "norm": 1,
        "distance_norm": 1,
        "lirpa_method": "backward",
        "lirpa_batch_size": 128,
        "classification_margin": 1.0e-4,
        "adaptive_eps": True,
        "adaptive_eps_shrink_factor": 0.5,
        "adaptive_eps_max_shrinks": 8,
        "adaptive_eps_min": 1.0e-6,
        "adaptive_eps_center_tol": 1.0e-6,
        "adaptive_eps_binary_search_steps": 0,
        # This ablation studies the L1 projection distance D_k directly.
        "sparsity_penalty": "none",
        "sparsity_lambda": 0.0,
        "sparsity_reweight_iters": 0,
        "sparsity_eps": 1.0e-3,
        "solver_maxiter": 500,
        "cvxpy_solvers": ["CLARABEL"],
        "cvxpy_solver_options": {
            "CLARABEL": {
                "max_iter": 100,
                "tol_gap_abs": 1.0e-3,
                "tol_gap_rel": 1.0e-5,
                "tol_feas": 1.0e-8,
            }
        },
        "cvxpy_accept_statuses": {"CLARABEL": ["optimal"]},
    },
    "artifacts": {
        "output_dir": "results/topk_heuristic_ablation",
        "notebook": "notebooks/TopKHeuristicAblation.ipynb",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    k_values = [int(value) for value in experiment["k_values"]]
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("experiment.k_values must be a non-empty list of positive integers")
    if k_values != sorted(set(k_values)):
        raise ValueError("experiment.k_values must be strictly increasing and unique")
    if int(experiment["n_queries"]) <= 0:
        raise ValueError("experiment.n_queries must be positive")
    if int(experiment["partial_save_every"]) <= 0:
        raise ValueError("experiment.partial_save_every must be positive")
    if float(experiment["absolute_tolerance"]) < 0.0:
        raise ValueError("experiment.absolute_tolerance must be non-negative")
    if float(experiment["relative_tolerance"]) < 0.0:
        raise ValueError("experiment.relative_tolerance must be non-negative")
    relative_tolerances = [float(value) for value in experiment["within_relative_tolerances"]]
    if any(value < 0.0 for value in relative_tolerances):
        raise ValueError("within_relative_tolerances must be non-negative")
    if relative_tolerances != [0.01, 0.05]:
        raise ValueError(
            "The current report fixes within_relative_tolerances to [0.01, 0.05]"
        )
    confidence = float(experiment["confidence_level"])
    if not 0.0 < confidence < 1.0:
        raise ValueError("experiment.confidence_level must lie in (0, 1)")

    support = config["support"]
    if int(support["points_per_true_class"]) <= 0:
        raise ValueError("support.points_per_true_class must be positive")
    if int(support["anchors_per_predicted_class"]) <= 0:
        raise ValueError("support.anchors_per_predicted_class must be positive")
    if str(support["sampling"]).lower() != "random":
        raise ValueError("This controlled ablation requires random anchor sampling")

    certcf = config["certcf"]
    if int(certcf["norm"]) != 1 or int(certcf["distance_norm"]) != 1:
        raise ValueError("The initial top-k ablation fixes atlas and projection norms to L1")
    if str(certcf["sparsity_penalty"]).lower() != "none":
        raise ValueError("The D_k ablation requires sparsity_penalty='none'")
    if int(certcf["lirpa_batch_size"]) <= 0:
        raise ValueError("certcf.lirpa_batch_size must be positive")


def config_fingerprint(config: dict[str, Any]) -> str:
    protocol = deepcopy(config)
    protocol.pop("artifacts", None)
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def prepare_metadata(self) -> Path:
        return self.root / "prepare.json"

    @property
    def atlas_dir(self) -> Path:
        return self.root / "atlas"

    @property
    def atlas_manifest(self) -> Path:
        return self.atlas_dir / "manifest.json"

    @property
    def atlas_support(self) -> Path:
        return self.atlas_dir / "selected_support.npz"

    @property
    def build_metadata(self) -> Path:
        return self.root / "build.json"

    @property
    def partial_queries(self) -> Path:
        return self.root / "benchmark_queries.partial.parquet"

    @property
    def benchmark_queries(self) -> Path:
        return self.root / "benchmark_queries.parquet"

    @property
    def benchmark_metadata(self) -> Path:
        return self.root / "benchmark.json"

    @property
    def combined_queries(self) -> Path:
        return self.root / "topk_queries.parquet"

    @property
    def minimal_k(self) -> Path:
        return self.root / "topk_minimal_k.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "topk_summary.parquet"

    @property
    def parallel_partial_queries(self) -> Path:
        return self.root / "topk_parallel_queries.partial.parquet"

    @property
    def parallel_queries(self) -> Path:
        return self.root / "topk_parallel_queries.parquet"

    @property
    def parallel_summary(self) -> Path:
        return self.root / "topk_parallel_summary.parquet"

    @property
    def parallel_metadata(self) -> Path:
        return self.root / "topk_parallel.json"


@dataclass
class PrefixState:
    """Cumulative strict top-k result after inspecting one anchor prefix."""

    k: int
    success: bool
    distance: float
    anchor_idx: int | None
    anchor_rank: int | None
    point: np.ndarray | None
    runtime_ms: float
    projection_time_ms: float
    n_projections: int
    n_pruned_by_bound: int
    initialization_rank: int | None


def strict_prefix_search(
    atlas: CertCFAtlas,
    x_query: np.ndarray,
    target_class: int,
    k_values: Iterable[int],
    *,
    fixed_dims: np.ndarray | None = None,
) -> list[PrefixState]:
    """Evaluate all strict nearest-anchor prefixes with shared projections.

    The function mirrors the fixed-budget part of ``nearest_anchor`` but never
    invokes its fallback.  Every candidate region is projected at most once;
    the best result is carried forward to derive all requested prefix sizes.
    """

    requested = sorted({int(value) for value in k_values})
    if not requested or requested[0] <= 0:
        raise ValueError("k_values must contain positive integers")
    if atlas.bounds is None:
        raise ValueError("Atlas bounds must be loaded")

    x_query = np.asarray(x_query, dtype=np.float64).reshape(-1)
    target_class = int(target_class)
    bounds = atlas.bounds[target_class]
    centers = np.asarray(bounds["X"], dtype=np.float64)
    n_polytopes = len(centers)
    maximum_k = requested[-1]
    if maximum_k > n_polytopes:
        raise ValueError(
            f"Requested k={maximum_k}, but target class {target_class} has only "
            f"{n_polytopes} atlas regions"
        )

    started = time.perf_counter()
    center_distances = np.linalg.norm(
        centers - x_query[None, :],
        ord=atlas.distance_norm,
        axis=1,
    )
    ordered_indices = np.argsort(center_distances)
    lower_bounds = atlas._anchor_bbox_lower_bounds(
        x_query,
        centers,
        np.asarray(bounds["eps"], dtype=np.float64),
        atlas.distance_norm,
    )

    best_point: np.ndarray | None = None
    best_distance = math.inf
    best_anchor_idx: int | None = None
    best_anchor_rank: int | None = None
    initialization_rank: int | None = None

    project_fn, projection_time_s, _ = atlas._make_project_fn_for_constraints(
        x_query,
        bounds,
        0.0,
        None,
        atlas.solver_maxiter,
        fixed_dims,
    )
    requested_set = set(requested)
    states: list[PrefixState] = []
    n_projections = 0
    n_pruned = 0
    for rank, raw_index in enumerate(ordered_indices[:maximum_k], start=1):
        candidate_index = int(raw_index)
        center_compatible = not (
            fixed_dims is not None
            and len(fixed_dims)
            and not np.allclose(
                centers[candidate_index][fixed_dims],
                x_query[fixed_dims],
                atol=1.0e-6,
            )
        )
        if center_compatible:
            _, center_certified = atlas._polytope_membership_for_anchor(
                centers[candidate_index],
                bounds,
                candidate_index,
                delta=0.0,
                robust_norm=atlas.norm,
            )
            if center_certified and float(center_distances[candidate_index]) < best_distance:
                best_point = centers[candidate_index].copy()
                best_distance = float(center_distances[candidate_index])
                best_anchor_idx = candidate_index
                best_anchor_rank = rank
                if initialization_rank is None:
                    initialization_rank = rank
        if float(lower_bounds[candidate_index]) >= best_distance:
            n_pruned += 1
        else:
            point, distance = project_fn(candidate_index, best_distance)
            n_projections += 1
            if point is not None and float(distance) < best_distance:
                best_point = np.asarray(point, dtype=np.float64).copy()
                best_distance = float(distance)
                best_anchor_idx = candidate_index
                best_anchor_rank = rank

        if rank in requested_set:
            states.append(
                PrefixState(
                    k=rank,
                    success=best_point is not None,
                    distance=float(best_distance),
                    anchor_idx=best_anchor_idx,
                    anchor_rank=best_anchor_rank,
                    point=None if best_point is None else best_point.copy(),
                    runtime_ms=1.0e3 * (time.perf_counter() - started),
                    projection_time_ms=1.0e3 * float(projection_time_s[0]),
                    n_projections=int(n_projections),
                    n_pruned_by_bound=int(n_pruned),
                    initialization_rank=initialization_rank,
                )
            )
    return states


def add_minimal_k_columns(frame: pd.DataFrame, maximum_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach per-query minimum recovery ranks and return their compact table."""

    result = frame.copy()
    keys = ["query_position", "query_idx"]
    records: list[dict[str, Any]] = []
    for key, group in result.groupby(keys, sort=True):
        exact_rows = group.loc[group["exact_recovery"].astype(bool)].sort_values("k")
        within_one_rows = group.loc[group["within_1pct"].astype(bool)].sort_values("k")
        within_five_rows = group.loc[group["within_5pct"].astype(bool)].sort_values("k")
        minimum_exact = int(exact_rows.iloc[0]["k"]) if len(exact_rows) else math.nan
        minimum_one = int(within_one_rows.iloc[0]["k"]) if len(within_one_rows) else math.nan
        minimum_five = int(within_five_rows.iloc[0]["k"]) if len(within_five_rows) else math.nan
        records.append(
            {
                "query_position": int(key[0]),
                "query_idx": int(key[1]),
                "minimum_k_exact": minimum_exact,
                "minimum_k_within_1pct": minimum_one,
                "minimum_k_within_5pct": minimum_five,
                "exact_recovered_within_sweep": bool(len(exact_rows)),
                "recovery_rank_censored": int(maximum_k + 1) if not len(exact_rows) else int(minimum_exact),
            }
        )
    ranks = pd.DataFrame(records)
    result = result.merge(ranks, on=keys, how="left", validate="many_to_one")
    return result, ranks


def one_sided_binomial_upper_bound(failures: int, trials: int, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper confidence bound."""

    failures = int(failures)
    trials = int(trials)
    if trials <= 0:
        return math.nan
    if failures <= 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / trials))
    if failures >= trials:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(confidence, failures + 1, trials - failures))


def summarize_results(frame: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Summarize strict top-k quality, cost, and empirical failure probability."""

    rows: list[dict[str, Any]] = []
    for k, group in frame.groupby("k", sort=True):
        shared = group.loc[group["strict_success"] & group["exact_success"]]
        gaps = shared["gap_absolute"].to_numpy(dtype=float)
        relative = shared["gap_relative"].to_numpy(dtype=float)
        failures = int((~group["exact_recovery"].astype(bool)).sum())
        trials = int(len(group))
        rows.append(
            {
                "k": int(k),
                "n_queries": trials,
                "strict_success_rate": float(group["strict_success"].mean()),
                "strict_target_validity": float(group["strict_target_valid"].mean()),
                "exact_success_rate": float(group["exact_success"].mean()),
                "exact_recovery_rate": float(group["exact_recovery"].mean()),
                "miss_probability_empirical": float(failures / trials) if trials else math.nan,
                "miss_probability_upper_confidence": one_sided_binomial_upper_bound(
                    failures, trials, confidence
                ),
                "within_1pct_rate": float(group["within_1pct"].mean()),
                "within_5pct_rate": float(group["within_5pct"].mean()),
                "validity_fallback_needed_rate": float(group["validity_fallback_needed"].mean()),
                "exhaustive_fallback_needed_rate": float(group["exhaustive_fallback_needed"].mean()),
                "gap_absolute_mean": float(np.mean(gaps)) if len(gaps) else math.nan,
                "gap_absolute_median": float(np.median(gaps)) if len(gaps) else math.nan,
                "gap_absolute_p95": float(np.percentile(gaps, 95)) if len(gaps) else math.nan,
                "gap_absolute_max": float(np.max(gaps)) if len(gaps) else math.nan,
                "gap_relative_mean": float(np.mean(relative)) if len(relative) else math.nan,
                "gap_relative_p95": float(np.percentile(relative, 95)) if len(relative) else math.nan,
                "strict_runtime_ms_mean": float(group["strict_runtime_ms"].mean()),
                "strict_runtime_ms_median": float(group["strict_runtime_ms"].median()),
                "strict_runtime_ms_p95": float(group["strict_runtime_ms"].quantile(0.95)),
                "strict_projection_time_ms_mean": float(group["strict_projection_time_ms"].mean()),
                "strict_projections_mean": float(group["strict_n_projections"].mean()),
                "exact_runtime_ms_mean": float(group["exact_runtime_ms"].mean()),
                "exact_projections_mean": float(group["exact_n_projections"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


class TopKHeuristicAblationRunner:
    """Prepare, build, run, aggregate, and analyze the top-k ablation."""

    def __init__(self, config: dict[str, Any], *, config_path: str | Path | None = None):
        validate_config(config)
        self.config = deepcopy(config)
        self.config_path = None if config_path is None else Path(config_path)
        self.fingerprint = config_fingerprint(self.config)
        self.paths = ArtifactPaths(Path(self.config["artifacts"]["output_dir"]).resolve())
        self.source_runner = NetworkComplexityRunner.from_yaml(
            self.config["experiment"]["source_config"]
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TopKHeuristicAblationRunner":
        return cls(load_config(path), config_path=path)

    @property
    def depth(self) -> int:
        return int(self.config["experiment"]["depth"])

    @property
    def width(self) -> int:
        return int(self.config["experiment"]["width"])

    @property
    def architecture_id(self) -> str:
        return architecture_id(self.depth, self.width)

    @property
    def k_values(self) -> list[int]:
        return [int(value) for value in self.config["experiment"]["k_values"]]

    @property
    def device(self) -> str:
        return _normalize_device(self.config["certcf"]["device"])

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _source_hashes(self) -> dict[str, str]:
        return {
            "dataset_sha256": sha256_file(self.source_runner.paths.dataset),
            "checkpoint_sha256": sha256_file(
                self.source_runner.paths.checkpoint(self.depth, self.width)
            ),
        }

    def _run_fingerprint(self) -> str:
        return _json_hash({"config_fingerprint": self.fingerprint, **self._source_hashes()})

    def _load_pool_and_queries(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.source_runner.paths.dataset, allow_pickle=False) as cached:
            x_train = cached["x_train"].astype(np.float32)
            y_train = cached["y_train"].astype(np.int64)
            x_test = cached["x_test"].astype(np.float32)
            y_test = cached["y_test"].astype(np.int64)
            query_indices = cached["query_indices"].astype(np.int64)

        rng = np.random.default_rng(int(self.config["experiment"]["seed"]))
        cap = int(self.config["support"]["points_per_true_class"])
        selected_parts: list[np.ndarray] = []
        for label in (0, 1):
            indices = np.flatnonzero(y_train == label)
            if len(indices) < cap:
                raise RuntimeError(
                    f"Class {label} contains {len(indices)} training points, fewer than requested {cap}"
                )
            if len(indices) > cap:
                indices = rng.choice(indices, size=cap, replace=False)
            selected_parts.append(indices)
        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)

        n_queries = int(self.config["experiment"]["n_queries"])
        if n_queries > len(query_indices):
            raise RuntimeError(
                f"Requested {n_queries} queries, but the shared dataset contains only {len(query_indices)}"
            )
        return x_train[selected], y_train[selected], x_test, y_test, query_indices[:n_queries]

    @staticmethod
    def _predict(model: torch.nn.Module, x: np.ndarray, device: str) -> np.ndarray:
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), 2048):
                tensor = torch.from_numpy(x[start : start + 2048]).to(device)
                predictions.append(model(tensor).argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions).astype(np.int64, copy=False)

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        del force  # Preparing this experiment must never regenerate the source dataset.
        self.source_runner.prepare(announce=False)
        if (self.depth, self.width) not in self.source_runner.grid:
            raise RuntimeError(
                f"Architecture {self.architecture_id} is not part of the source grid"
            )
        if not self.source_runner._valid_training_artifact(self.depth, self.width):
            raise RuntimeError(
                f"Missing or stale source checkpoint for {self.architecture_id}. "
                "Train it with scripts/network_complexity_grid.py first."
            )
        x_pool, y_pool, _, y_test, query_indices = self._load_pool_and_queries()
        metadata = {
            "status": "complete",
            "architecture_id": self.architecture_id,
            "depth": self.depth,
            "width": self.width,
            "parameter_count": parameter_count(32, [self.width] * self.depth, 2),
            "config_fingerprint": self.fingerprint,
            "run_fingerprint": self._run_fingerprint(),
            **self._source_hashes(),
            "support_points": int(len(x_pool)),
            "support_true_class_counts": {
                str(label): int(np.sum(y_pool == label)) for label in (0, 1)
            },
            "n_queries": int(len(query_indices)),
            "query_true_class_counts": {
                str(label): int(np.sum(y_test[query_indices] == label)) for label in (0, 1)
            },
            "k_values": self.k_values,
        }
        _atomic_json(metadata, self.paths.prepare_metadata)
        _log(
            f"[PREPARE] {self.architecture_id}: {len(x_pool)} supporti, "
            f"{len(query_indices)} query condivise, k={self.k_values}."
        )
        return metadata

    def _select_atlas_support(
        self,
        x_pool: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k_per_class = int(self.config["support"]["anchors_per_predicted_class"])
        seed = int(self.config["experiment"]["seed"])
        global_indices: list[np.ndarray] = []
        for label in (0, 1):
            class_indices = np.flatnonzero(predicted_labels == label)
            if len(class_indices) < k_per_class:
                raise RuntimeError(
                    f"Predicted class {label} contains only {len(class_indices)} support points; "
                    f"cannot sample {k_per_class} anchors"
                )
            local = select_prototype_indices(
                x_pool[class_indices],
                k_per_class,
                method="random",
                random_state=seed,
            )
            global_indices.append(class_indices[local])
        selected = np.concatenate(global_indices).astype(np.int64, copy=False)
        return x_pool[selected], predicted_labels[selected], selected

    def _new_atlas(
        self,
        model: torch.nn.Module,
        x_support: np.ndarray,
        y_support: np.ndarray,
        *,
        bounds_checkpoint_dir: Path | None = None,
    ) -> CertCFAtlas:
        cfg = self.config["certcf"]
        clean_model = _strip_dropout_modules(model.net).eval().to(self.device)
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(x_support, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_support, dtype=np.int64)),
        )
        return CertCFAtlas(
            clean_model,
            dataset,
            self.device,
            norm=int(cfg["norm"]),
            distance_norm=int(cfg["distance_norm"]),
            lirpa_method=str(cfg["lirpa_method"]),
            eps_strategy=NearestOppositeClassClearanceStrategy(alpha=float(cfg["eps_alpha"])),
            batch_size=int(cfg["lirpa_batch_size"]),
            bounds_checkpoint_dir=bounds_checkpoint_dir,
            default_query_method="sorted",
            solver_maxiter=int(cfg["solver_maxiter"]),
            query_parallelism=1,
            cvxpy_solvers=list(cfg["cvxpy_solvers"]),
            cvxpy_solver_options=deepcopy(cfg["cvxpy_solver_options"]),
            cvxpy_accept_statuses=deepcopy(cfg["cvxpy_accept_statuses"]),
            classification_margin=float(cfg["classification_margin"]),
            adaptive_eps=bool(cfg["adaptive_eps"]),
            adaptive_eps_shrink_factor=float(cfg["adaptive_eps_shrink_factor"]),
            adaptive_eps_max_shrinks=int(cfg["adaptive_eps_max_shrinks"]),
            adaptive_eps_min=float(cfg["adaptive_eps_min"]),
            adaptive_eps_center_tol=float(cfg["adaptive_eps_center_tol"]),
            adaptive_eps_binary_search_steps=int(cfg["adaptive_eps_binary_search_steps"]),
            sparsity_penalty=str(cfg["sparsity_penalty"]),
            sparsity_lambda=float(cfg["sparsity_lambda"]),
            sparsity_reweight_iters=int(cfg["sparsity_reweight_iters"]),
            sparsity_eps=float(cfg["sparsity_eps"]),
            sparsity_group_ohe=True,
        )

    def _atlas_file_hashes(self) -> dict[str, str]:
        files = [self.paths.atlas_manifest, self.paths.atlas_support]
        manifest = self._read_json(self.paths.atlas_manifest)
        if manifest:
            files.extend(self.paths.atlas_dir / name for name in manifest.get("files", {}).values())
        return {
            str(path.relative_to(self.paths.root)): sha256_file(path)
            for path in files
            if path.exists()
        }

    def _valid_build(self) -> bool:
        metadata = self._read_json(self.paths.build_metadata)
        if not metadata or metadata.get("status") != "complete":
            return False
        if metadata.get("run_fingerprint") != self._run_fingerprint():
            return False
        stored = metadata.get("atlas_file_sha256", {})
        if not stored or stored != self._atlas_file_hashes():
            return False
        return self.paths.atlas_manifest.exists() and self.paths.atlas_support.exists()

    def build(self, *, force: bool = False) -> dict[str, Any]:
        self.prepare()
        if self._valid_build() and not force:
            _log("[BUILD] Atlas serializzato valido, skip.")
            return self._read_json(self.paths.build_metadata) or {}

        x_pool, _, _, _, _ = self._load_pool_and_queries()
        model = self.source_runner._load_model(self.depth, self.width, self.device)
        predicted_labels = self._predict(model, x_pool, self.device)
        x_support, y_support, selected_indices = self._select_atlas_support(
            x_pool, predicted_labels
        )
        partial_dir = None
        if not force:
            partial_dir = self.paths.root / "bounds_partial" / self._run_fingerprint()
        atlas = self._new_atlas(
            model,
            x_support,
            y_support,
            bounds_checkpoint_dir=partial_dir,
        )
        _log(
            f"[BUILD] {self.architecture_id}: {len(x_pool)} punti di supporto, "
            f"{len(x_support)} anchor campionati casualmente."
        )
        started = time.perf_counter()
        atlas.build(verbose=True)
        build_time = time.perf_counter() - started
        _atomic_npz(
            self.paths.atlas_support,
            x_support=x_support,
            y_support=y_support,
            selected_pool_indices=selected_indices,
        )
        atlas.save_bounds(self.paths.atlas_dir)

        region_counts = {
            str(label): int(len(atlas.bounds[label]["X"])) for label in atlas.class_labels
        }
        eps_initial = np.concatenate(
            [np.asarray(atlas.bounds[label]["eps_initial"]) for label in atlas.class_labels]
        )
        eps_final = np.concatenate(
            [np.asarray(atlas.bounds[label]["eps"]) for label in atlas.class_labels]
        )
        shrinks = np.concatenate(
            [
                np.asarray(atlas.bounds[label]["adaptive_eps_n_shrinks"])
                for label in atlas.class_labels
            ]
        )
        metadata = {
            "status": "complete",
            "architecture_id": self.architecture_id,
            "config_fingerprint": self.fingerprint,
            "run_fingerprint": self._run_fingerprint(),
            **self._source_hashes(),
            "build_wall_time_s": float(build_time),
            "support_pool_size": int(len(x_pool)),
            "sampled_anchor_count": int(len(x_support)),
            "sampled_anchor_counts_by_predicted_class": {
                str(label): int(np.sum(y_support == label)) for label in (0, 1)
            },
            "certified_region_count": int(sum(region_counts.values())),
            "certified_region_counts_by_class": region_counts,
            "eps_initial_median": float(np.median(eps_initial)),
            "eps_final_median": float(np.median(eps_final)),
            "adaptive_shrink_fraction": float(np.mean(shrinks > 0)),
            "atlas_file_sha256": self._atlas_file_hashes(),
        }
        _atomic_json(metadata, self.paths.build_metadata)
        _log(
            f"[BUILD] Completato: {metadata['certified_region_count']} regioni "
            f"in {build_time:.1f}s."
        )
        return metadata

    def _load_atlas_and_data(
        self,
    ) -> tuple[CertCFAtlas, torch.nn.Module, np.ndarray, np.ndarray, np.ndarray]:
        if not self._valid_build():
            raise RuntimeError("Missing or stale atlas. Run the build stage first.")
        model = self.source_runner._load_model(self.depth, self.width, self.device)
        with np.load(self.paths.atlas_support, allow_pickle=False) as saved:
            x_support = saved["x_support"].astype(np.float32)
            y_support = saved["y_support"].astype(np.int64)
        atlas = self._new_atlas(model, x_support, y_support)
        atlas.load_bounds(self.paths.atlas_dir)
        _, _, x_test, y_test, query_indices = self._load_pool_and_queries()
        return atlas, model, x_test, y_test, query_indices

    def _valid_benchmark(self) -> bool:
        metadata = self._read_json(self.paths.benchmark_metadata)
        if not metadata or metadata.get("status") != "complete":
            return False
        if metadata.get("run_fingerprint") != self._run_fingerprint():
            return False
        if not self.paths.benchmark_queries.exists():
            return False
        try:
            frame = pd.read_parquet(
                self.paths.benchmark_queries,
                columns=["query_position", "query_idx", "k", "run_fingerprint"],
            )
        except Exception:
            return False
        expected_queries = int(self.config["experiment"]["n_queries"])
        expected_rows = expected_queries * len(self.k_values)
        return bool(
            len(frame) == expected_rows
            and frame[["query_position", "k"]].drop_duplicates().shape[0] == expected_rows
            and set(frame["k"].astype(int)) == set(self.k_values)
            and frame["run_fingerprint"].eq(self._run_fingerprint()).all()
        )

    def _resume_frame(self, force: bool) -> pd.DataFrame:
        if force or not self.paths.partial_queries.exists():
            return pd.DataFrame()
        metadata = self._read_json(self.paths.benchmark_metadata) or {}
        if metadata.get("run_fingerprint") != self._run_fingerprint():
            return pd.DataFrame()
        try:
            frame = pd.read_parquet(self.paths.partial_queries)
        except Exception:
            return pd.DataFrame()
        return frame.loc[frame["run_fingerprint"].eq(self._run_fingerprint())].copy()

    def _save_partial(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [frame for frame in frames if not frame.empty]
        combined = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
        if not combined.empty:
            combined = (
                combined.sort_values(["query_position", "k"])
                .drop_duplicates(["query_position", "k"], keep="last")
                .reset_index(drop=True)
            )
            _atomic_parquet(combined, self.paths.partial_queries)
        return combined

    def benchmark(self, *, force: bool = False) -> pd.DataFrame:
        self.build(force=False)
        if self._valid_benchmark() and not force:
            _log("[BENCHMARK] Risultato completo e valido, skip.")
            return pd.read_parquet(self.paths.benchmark_queries)

        if force and self.paths.partial_queries.exists():
            self.paths.partial_queries.unlink()

        atlas, model, x_test, y_test, query_indices = self._load_atlas_and_data()
        x_queries = x_test[query_indices]
        source_predictions = self._predict(model, x_queries, self.device)
        targets = 1 - source_predictions
        existing = self._resume_frame(force)
        complete_positions: set[int] = set()
        if not existing.empty:
            counts = existing.groupby("query_position")["k"].nunique()
            complete_positions = set(counts[counts == len(self.k_values)].index.astype(int))
        new_frames: list[pd.DataFrame] = []
        metadata = {
            "status": "running",
            "run_fingerprint": self._run_fingerprint(),
            "config_fingerprint": self.fingerprint,
            "completed_queries": int(len(complete_positions)),
            "expected_queries": int(len(query_indices)),
        }
        _atomic_json(metadata, self.paths.benchmark_metadata)

        save_every = int(self.config["experiment"]["partial_save_every"])
        absolute_tolerance = float(self.config["experiment"]["absolute_tolerance"])
        relative_tolerance = float(self.config["experiment"]["relative_tolerance"])
        iterator = tqdm(
            range(len(query_indices)),
            desc=f"TOP-K {self.architecture_id}",
            unit="query",
            dynamic_ncols=True,
        )
        interrupted = [False]
        previous_sigint_handler = signal.getsignal(signal.SIGINT)

        def request_graceful_stop(signum, frame):
            del signum, frame
            if interrupted[0]:
                raise KeyboardInterrupt
            interrupted[0] = True
            _log("[BENCHMARK] Interruzione richiesta; salvo dopo la query corrente.")

        signal.signal(signal.SIGINT, request_graceful_stop)
        try:
            for position in iterator:
                if position in complete_positions:
                    continue
                x_query = x_queries[position]
                target = int(targets[position])
                prefix_states = strict_prefix_search(atlas, x_query, target, self.k_values)

                exact_started = time.perf_counter()
                exact = atlas.find_counterfactual(
                    x_query,
                    target_class=target,
                    method="sorted",
                )
                exact_runtime_ms = 1.0e3 * (time.perf_counter() - exact_started)
                exact_point = None if exact.x_cf is None else np.asarray(exact.x_cf, dtype=np.float32)
                exact_prediction = -1
                if exact_point is not None:
                    exact_prediction = int(self._predict(model, exact_point[None, :], self.device)[0])
                exact_success = bool(exact.success and exact_prediction == target)
                exact_distance = float(exact.distance) if exact_success else math.inf
                exact_profile = dict(exact.profiling or {})

                query_rows: list[dict[str, Any]] = []
                for state in prefix_states:
                    strict_prediction = -1
                    if state.point is not None:
                        strict_prediction = int(
                            self._predict(model, state.point.astype(np.float32)[None, :], self.device)[0]
                        )
                    strict_target_valid = bool(state.success and strict_prediction == target)
                    strict_distance = float(state.distance) if state.success else math.inf
                    raw_gap = strict_distance - exact_distance
                    comparison_tolerance = absolute_tolerance + relative_tolerance * max(
                        1.0, abs(exact_distance)
                    )
                    shared_success = bool(state.success and exact_success)
                    gap_absolute = max(0.0, raw_gap) if shared_success else math.inf
                    gap_relative = (
                        gap_absolute / max(abs(exact_distance), absolute_tolerance)
                        if shared_success
                        else math.inf
                    )
                    exact_recovery = bool(shared_success and raw_gap <= comparison_tolerance)
                    within_1pct = bool(
                        shared_success
                        and strict_distance <= exact_distance * 1.01 + absolute_tolerance
                    )
                    within_5pct = bool(
                        shared_success
                        and strict_distance <= exact_distance * 1.05 + absolute_tolerance
                    )
                    query_rows.append(
                        {
                            "run_fingerprint": self._run_fingerprint(),
                            "architecture_id": self.architecture_id,
                            "depth": self.depth,
                            "width": self.width,
                            "parameter_count": parameter_count(32, [self.width] * self.depth, 2),
                            "query_position": int(position),
                            "query_idx": int(query_indices[position]),
                            "y_true": int(y_test[query_indices[position]]),
                            "source_class": int(source_predictions[position]),
                            "target_class": target,
                            "k": int(state.k),
                            "strict_success": bool(state.success),
                            "strict_target_valid": strict_target_valid,
                            "strict_prediction": strict_prediction,
                            "strict_distance": strict_distance,
                            "strict_anchor_idx": (
                                math.nan if state.anchor_idx is None else int(state.anchor_idx)
                            ),
                            "strict_anchor_rank": (
                                math.nan if state.anchor_rank is None else int(state.anchor_rank)
                            ),
                            "strict_initialization_rank": (
                                math.nan
                                if state.initialization_rank is None
                                else int(state.initialization_rank)
                            ),
                            "strict_runtime_ms": float(state.runtime_ms),
                            "strict_projection_time_ms": float(state.projection_time_ms),
                            "strict_n_projections": int(state.n_projections),
                            "strict_n_pruned_by_bound": int(state.n_pruned_by_bound),
                            "validity_fallback_needed": bool(not state.success),
                            "exact_success": exact_success,
                            "exact_target_valid": exact_success,
                            "exact_prediction": exact_prediction,
                            "exact_distance": exact_distance,
                            "exact_anchor_idx": (
                                math.nan if exact.anchor_idx is None else int(exact.anchor_idx)
                            ),
                            "exact_runtime_ms": float(exact_runtime_ms),
                            "exact_n_projections": int(exact.n_qp_solved),
                            "exact_recovery": exact_recovery,
                            "exhaustive_fallback_needed": bool(not exact_recovery),
                            "gap_raw": float(raw_gap),
                            "gap_absolute": float(gap_absolute),
                            "gap_relative": float(gap_relative),
                            "within_1pct": within_1pct,
                            "within_5pct": within_5pct,
                            "numerical_negative_gap": bool(
                                shared_success and raw_gap < -comparison_tolerance
                            ),
                            "exact_search_time_ms": float(
                                exact_profile.get("search_time_ms", math.nan)
                            ),
                            "exact_projection_time_ms": float(
                                exact_profile.get("projection_time_ms", math.nan)
                            ),
                        }
                    )
                new_frames.append(pd.DataFrame(query_rows))

                completed_now = len(complete_positions) + len(new_frames)
                iterator.set_postfix(completed=completed_now)
                if len(new_frames) % save_every == 0:
                    existing = self._save_partial([existing, *new_frames])
                    new_frames = []
                    metadata["completed_queries"] = int(
                        existing.groupby("query_position")["k"].nunique().eq(len(self.k_values)).sum()
                    )
                    _atomic_json(metadata, self.paths.benchmark_metadata)
                if interrupted[0]:
                    raise KeyboardInterrupt
        except BaseException:
            partial = self._save_partial([existing, *new_frames])
            metadata["status"] = "partial"
            if not partial.empty:
                metadata["completed_queries"] = int(
                    partial.groupby("query_position")["k"]
                    .nunique()
                    .eq(len(self.k_values))
                    .sum()
                )
            _atomic_json(metadata, self.paths.benchmark_metadata)
            raise
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

        completed = self._save_partial([existing, *new_frames])
        expected_rows = len(query_indices) * len(self.k_values)
        if len(completed) != expected_rows:
            raise RuntimeError(
                f"Incomplete benchmark: expected {expected_rows} rows, found {len(completed)}"
            )
        _atomic_parquet(completed, self.paths.benchmark_queries)
        metadata.update(
            {
                "status": "complete",
                "completed_queries": int(len(query_indices)),
                "rows": int(len(completed)),
                "benchmark_sha256": sha256_file(self.paths.benchmark_queries),
            }
        )
        _atomic_json(metadata, self.paths.benchmark_metadata)
        _log(
            f"[BENCHMARK] Completato: {len(query_indices)} query, "
            f"{len(completed)} coppie query-k."
        )
        return completed

    def _parallel_protocol_hash(
        self,
        k_values: Iterable[int],
        worker_values: Iterable[int],
        n_queries: int,
        backend: str,
    ) -> str:
        return _json_hash(
            {
                "run_fingerprint": self._run_fingerprint(),
                "k_values": [int(value) for value in k_values],
                "worker_values": [int(value) for value in worker_values],
                "n_queries": int(n_queries),
                "query_parallelism": 1,
                "method": "nearest_anchor",
                "candidate_parallel_backend": str(backend),
            }
        )

    @staticmethod
    def _attach_parallel_serial_reference(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.drop(
            columns=[
                "serial_runtime_ms",
                "serial_distance",
                "serial_anchor_idx",
                "speedup_vs_serial",
                "matches_serial_distance",
                "matches_serial_anchor",
            ],
            errors="ignore",
        ).copy()
        serial = result.loc[result["candidate_workers_requested"].eq(1)].copy()
        if serial[["query_position", "k"]].duplicated().any():
            raise RuntimeError("Parallel benchmark contains duplicate serial references")
        serial = serial[
            ["query_position", "k", "runtime_ms", "distance", "anchor_idx"]
        ].rename(
            columns={
                "runtime_ms": "serial_runtime_ms",
                "distance": "serial_distance",
                "anchor_idx": "serial_anchor_idx",
            }
        )
        result = result.merge(
            serial,
            on=["query_position", "k"],
            how="left",
            validate="many_to_one",
        )
        if result["serial_runtime_ms"].isna().any():
            raise RuntimeError("Missing serial reference for at least one query-k pair")
        result["speedup_vs_serial"] = (
            result["serial_runtime_ms"] / result["runtime_ms"].clip(lower=1.0e-12)
        )
        tolerance = 1.0e-7 + 1.0e-6 * np.maximum(
            1.0,
            np.abs(result["serial_distance"].to_numpy(dtype=float)),
        )
        result["matches_serial_distance"] = (
            np.abs(
                result["distance"].to_numpy(dtype=float)
                - result["serial_distance"].to_numpy(dtype=float)
            )
            <= tolerance
        )
        result["matches_serial_anchor"] = (
            result["anchor_idx"].fillna(-1).astype(int)
            == result["serial_anchor_idx"].fillna(-1).astype(int)
        )
        return result

    def benchmark_parallelism(
        self,
        *,
        k_values: Iterable[int] = tuple(range(1, 9)),
        worker_values: Iterable[int] = (1, 2, 4, 8),
        n_queries: int | None = None,
        backend: str = "process",
        force: bool = False,
    ) -> pd.DataFrame:
        """Measure isolated per-k latency under intra-query parallelism."""
        self.build(force=False)
        requested_k = sorted({int(value) for value in k_values})
        requested_workers = sorted({int(value) for value in worker_values})
        if not requested_k or any(value <= 0 for value in requested_k):
            raise ValueError("parallel k_values must be positive")
        if not requested_workers or requested_workers[0] != 1:
            raise ValueError("worker_values must be positive and include the serial value 1")
        if any(value <= 0 for value in requested_workers):
            raise ValueError("worker_values must be positive")
        backend = str(backend).lower()
        if backend not in {"thread", "process"}:
            raise ValueError("backend must be one of {'thread', 'process'}")

        configured_queries = int(self.config["experiment"]["n_queries"])
        selected_queries = configured_queries if n_queries is None else int(n_queries)
        if selected_queries <= 0 or selected_queries > configured_queries:
            raise ValueError(
                f"n_queries must lie in [1, {configured_queries}], got {selected_queries}"
            )
        protocol_hash = self._parallel_protocol_hash(
            requested_k,
            requested_workers,
            selected_queries,
            backend,
        )
        metadata = self._read_json(self.paths.parallel_metadata) or {}
        if (
            not force
            and metadata.get("status") == "complete"
            and metadata.get("protocol_hash") == protocol_hash
            and self.paths.parallel_queries.exists()
        ):
            return pd.read_parquet(self.paths.parallel_queries)

        if force:
            for path in (self.paths.parallel_partial_queries, self.paths.parallel_queries):
                if path.exists():
                    path.unlink()

        atlas, model, x_test, _, query_indices = self._load_atlas_and_data()
        query_indices = query_indices[:selected_queries]
        x_queries = x_test[query_indices]
        targets = 1 - self._predict(model, x_queries, self.device)
        for target in np.unique(targets):
            available = int(len(atlas.bounds[int(target)]["X"]))
            if requested_k[-1] > available:
                raise ValueError(
                    f"Requested k={requested_k[-1]}, but target class {target} has "
                    f"only {available} atlas regions"
                )

        existing = pd.DataFrame()
        if not force and self.paths.parallel_partial_queries.exists():
            try:
                candidate = pd.read_parquet(self.paths.parallel_partial_queries)
                if candidate["protocol_hash"].eq(protocol_hash).all():
                    existing = candidate.copy()
            except Exception:
                existing = pd.DataFrame()
        complete_keys = set()
        if not existing.empty:
            complete_keys = set(
                zip(
                    existing["query_position"].astype(int),
                    existing["k"].astype(int),
                    existing["candidate_workers_requested"].astype(int),
                )
            )

        metadata = {
            "status": "running",
            "protocol_hash": protocol_hash,
            "run_fingerprint": self._run_fingerprint(),
            "k_values": requested_k,
            "worker_values": requested_workers,
            "n_queries": selected_queries,
            "candidate_parallel_backend": backend,
        }
        _atomic_json(metadata, self.paths.parallel_metadata)
        if backend == "process" and requested_workers[-1] > 1:
            _log(
                f"[PARALLEL WARMUP] Avvio {requested_workers[-1]} processi e "
                f"warm-up con k={requested_k[-1]}."
            )
            atlas.find_counterfactual(
                x_queries[0],
                target_class=int(targets[0]),
                method="nearest_anchor",
                query_k_candidates=int(requested_k[-1]),
                candidate_parallelism=int(requested_workers[-1]),
                candidate_parallel_backend=backend,
            )
        new_frames: list[pd.DataFrame] = []
        save_every = int(self.config["experiment"]["partial_save_every"])
        iterator = tqdm(
            range(selected_queries),
            desc=f"TOP-K PARALLEL {self.architecture_id}",
            unit="query",
            dynamic_ncols=True,
        )
        try:
            for position in iterator:
                x_query = x_queries[position]
                target = int(targets[position])
                combinations = [
                    (k, workers)
                    for k in requested_k
                    for workers in requested_workers
                    if (position, k, workers) not in complete_keys
                ]
                rng = np.random.default_rng(
                    int(self.config["experiment"]["seed"]) + int(position)
                )
                rng.shuffle(combinations)
                rows: list[dict[str, Any]] = []
                for k, workers in combinations:
                    started = time.perf_counter()
                    result = atlas.find_counterfactual(
                        x_query,
                        target_class=target,
                        method="nearest_anchor",
                        query_k_candidates=int(k),
                        candidate_parallelism=int(workers),
                        candidate_parallel_backend=backend,
                    )
                    runtime_ms = 1.0e3 * (time.perf_counter() - started)
                    profile = dict(result.profiling or {})
                    point = None if result.x_cf is None else np.asarray(result.x_cf, dtype=np.float32)
                    prediction = -1
                    if point is not None:
                        prediction = int(self._predict(model, point[None, :], self.device)[0])
                    rows.append(
                        {
                            "protocol_hash": protocol_hash,
                            "run_fingerprint": self._run_fingerprint(),
                            "architecture_id": self.architecture_id,
                            "query_position": int(position),
                            "query_idx": int(query_indices[position]),
                            "target_class": target,
                            "k": int(k),
                            "candidate_workers_requested": int(workers),
                            "candidate_parallel_backend": backend,
                            "candidate_workers_used": int(
                                profile.get("candidate_workers_used", 1)
                            ),
                            "success": bool(result.success),
                            "target_valid": bool(result.success and prediction == target),
                            "distance": float(result.distance),
                            "anchor_idx": (
                                math.nan if result.anchor_idx is None else int(result.anchor_idx)
                            ),
                            "runtime_ms": float(runtime_ms),
                            "projection_wall_time_ms": float(
                                profile.get("projection_wall_time_ms", math.nan)
                            ),
                            "projection_work_time_ms": float(
                                profile.get("projection_time_ms", math.nan)
                            ),
                            "n_projections": int(result.n_qp_solved),
                            "n_pruned_by_bound": int(
                                profile.get("n_candidates_pruned_by_bound", 0)
                            ),
                            "fallback_used": bool(
                                profile.get("nearest_anchor_fallback_used", False)
                            ),
                        }
                    )
                if rows:
                    new_frames.append(pd.DataFrame(rows))
                if len(new_frames) >= save_every:
                    existing = pd.concat([existing, *new_frames], ignore_index=True)
                    existing = existing.drop_duplicates(
                        ["query_position", "k", "candidate_workers_requested"],
                        keep="last",
                    )
                    _atomic_parquet(existing, self.paths.parallel_partial_queries)
                    new_frames = []
                iterator.set_postfix(rows=len(existing) + sum(len(frame) for frame in new_frames))
        except BaseException:
            partial = pd.concat([existing, *new_frames], ignore_index=True)
            if not partial.empty:
                partial = partial.drop_duplicates(
                    ["query_position", "k", "candidate_workers_requested"],
                    keep="last",
                )
                _atomic_parquet(partial, self.paths.parallel_partial_queries)
            metadata["status"] = "partial"
            metadata["rows"] = int(len(partial))
            _atomic_json(metadata, self.paths.parallel_metadata)
            atlas.close_candidate_process_pool()
            raise

        atlas.close_candidate_process_pool()
        completed = pd.concat([existing, *new_frames], ignore_index=True)
        completed = completed.drop_duplicates(
            ["query_position", "k", "candidate_workers_requested"],
            keep="last",
        ).sort_values(["query_position", "k", "candidate_workers_requested"])
        expected_rows = selected_queries * len(requested_k) * len(requested_workers)
        if len(completed) != expected_rows:
            raise RuntimeError(
                f"Incomplete parallel benchmark: expected {expected_rows} rows, "
                f"found {len(completed)}"
            )
        completed = self._attach_parallel_serial_reference(completed)
        _atomic_parquet(completed, self.paths.parallel_queries)
        metadata.update({"status": "complete", "rows": int(len(completed))})
        _atomic_json(metadata, self.paths.parallel_metadata)
        _log(
            f"[PARALLEL] {selected_queries} query, k={requested_k}, "
            f"worker={requested_workers}."
        )
        return completed

    def analyze_parallelism(self) -> pd.DataFrame:
        if not self.paths.parallel_queries.exists():
            raise RuntimeError("Run the parallel-benchmark stage first")
        frame = self._attach_parallel_serial_reference(
            pd.read_parquet(self.paths.parallel_queries)
        )
        grouped = frame.groupby(["k", "candidate_workers_requested"], sort=True)
        summary = grouped.agg(
            queries=("query_position", "nunique"),
            success_rate=("success", "mean"),
            target_validity=("target_valid", "mean"),
            mean_runtime_ms=("runtime_ms", "mean"),
            median_runtime_ms=("runtime_ms", "median"),
            p95_runtime_ms=("runtime_ms", lambda values: float(np.percentile(values, 95))),
            mean_projection_wall_time_ms=("projection_wall_time_ms", "mean"),
            mean_projection_work_time_ms=("projection_work_time_ms", "mean"),
            mean_speedup_vs_serial=("speedup_vs_serial", "mean"),
            median_speedup_vs_serial=("speedup_vs_serial", "median"),
            serial_distance_match_rate=("matches_serial_distance", "mean"),
            serial_anchor_match_rate=("matches_serial_anchor", "mean"),
            mean_projections=("n_projections", "mean"),
            fallback_rate=("fallback_used", "mean"),
        ).reset_index()
        summary["mean_parallel_efficiency"] = (
            summary["mean_speedup_vs_serial"]
            / summary["candidate_workers_requested"].clip(lower=1)
        )
        _atomic_parquet(summary, self.paths.parallel_summary)
        _log(f"[PARALLEL ANALYZE] Tabella salvata in {self.paths.parallel_summary}.")
        return summary

    def aggregate(self) -> pd.DataFrame:
        if not self._valid_benchmark():
            raise RuntimeError("The complete benchmark artifact is missing or stale")
        raw = pd.read_parquet(self.paths.benchmark_queries)
        combined, ranks = add_minimal_k_columns(raw, max(self.k_values))
        expected_rows = int(self.config["experiment"]["n_queries"]) * len(self.k_values)
        if len(combined) != expected_rows:
            raise RuntimeError(
                f"Aggregate validation failed: expected {expected_rows} rows, found {len(combined)}"
            )
        _atomic_parquet(combined, self.paths.combined_queries)
        _atomic_parquet(ranks, self.paths.minimal_k)
        _log(
            f"[AGGREGATE] {len(combined)} righe, "
            f"{len(ranks)} query, k={self.k_values}."
        )
        return combined

    def analyze(self) -> pd.DataFrame:
        combined = self.aggregate()
        summary = summarize_results(
            combined,
            confidence=float(self.config["experiment"]["confidence_level"]),
        )
        _atomic_parquet(summary, self.paths.summary)
        _log(f"[ANALYZE] Tabella salvata in {self.paths.summary}.")
        return summary

    def status(self) -> dict[str, Any]:
        prepare = self._read_json(self.paths.prepare_metadata) or {}
        build = self._read_json(self.paths.build_metadata) or {}
        benchmark = self._read_json(self.paths.benchmark_metadata) or {}
        parallel = self._read_json(self.paths.parallel_metadata) or {}
        partial_queries = 0
        if self.paths.partial_queries.exists():
            try:
                partial = pd.read_parquet(self.paths.partial_queries, columns=["query_position", "k"])
                partial_queries = int(
                    partial.groupby("query_position")["k"].nunique().eq(len(self.k_values)).sum()
                )
            except Exception:
                partial_queries = 0
        return {
            "config_fingerprint": self.fingerprint,
            "run_fingerprint": self._run_fingerprint(),
            "architecture_id": self.architecture_id,
            "prepare_complete": prepare.get("run_fingerprint") == self._run_fingerprint(),
            "build_complete": self._valid_build(),
            "benchmark_complete": self._valid_benchmark(),
            "partial_queries_complete": partial_queries,
            "expected_queries": int(self.config["experiment"]["n_queries"]),
            "aggregate_complete": self.paths.combined_queries.exists(),
            "analysis_complete": self.paths.summary.exists(),
            "parallel_benchmark_status": parallel.get("status"),
            "parallel_benchmark_rows": int(parallel.get("rows", 0)),
            "parallel_analysis_complete": self.paths.parallel_summary.exists(),
            "build_status": build.get("status"),
            "benchmark_status": benchmark.get("status"),
        }

    def all(self, *, force: bool = False) -> pd.DataFrame:
        _log("[PIPELINE 1/5] Preparazione e validazione degli artifact sorgente")
        self.prepare()
        _log("[PIPELINE 2/5] Costruzione o caricamento dell'atlas certificato")
        self.build(force=force)
        _log("[PIPELINE 3/5] Sweep top-k e riferimento esaustivo")
        self.benchmark(force=force)
        _log("[PIPELINE 4/5] Aggregazione e validazione")
        self.aggregate()
        _log("[PIPELINE 5/5] Analisi")
        return self.analyze()
