"""Anchor-ball/PGD/CertCF ablation of the LiRPA refinement step.

The runner deliberately owns anchor selection and Eq. (1) clearances.  Every
method therefore consumes the same immutable geometry rather than relying on
method-local sampling.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import TensorDataset

from certcf import CertCFAtlas, NearestOppositeClassClearanceStrategy
from counterfactuals.benchmarks.registry import create_default_registries
from counterfactuals.methods.certcf import CertCF
from counterfactuals.models.torch_model import TorchModelWrapper
from dataset_specs import get_tabular_dataset_spec
from experiments.network_complexity import (
    NetworkComplexityRunner,
    PhaseResourceMonitor,
    architecture_id,
    load_config as load_network_grid_config,
    sha256_file,
)


METHODS = ("anchor_ball", "anchor_pgd", "certcf")

_TABULAR_RAW_FILES = {
    "adult": Path("Adult/raw.parquet"),
    "compas": Path("Compas/raw.parquet"),
    "german_credit": Path("GermanCredit/raw.parquet"),
    "give_me_some_credit": Path("Give Me Some Credit/raw.parquet"),
    "heloc": Path("Heloc/raw.parquet"),
    "lending_club": Path("LendingClub/raw.parquet"),
    "wisconsin_breast_cancer": Path("WisconsinBreastCancer/raw.parquet"),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "require_cuda": True,
        "cases": [
            {"id": "heloc", "kind": "tabular", "dataset": "heloc"},
            {"id": "adult", "kind": "tabular", "dataset": "adult"},
            *[
                {
                    "id": architecture_id(depth, 128),
                    "kind": "synthetic32",
                    "depth": depth,
                    "width": 128,
                }
                for depth in range(1, 6)
            ],
        ],
        "source_grid_config": "configs/experiments/network_complexity_grid.yaml",
    },
    "data": {
        "data_dir": "data",
        "queries_per_case": 500,
        "tabular_support_total": None,
        "tabular_support_per_true_class": 10000,
        "synthetic_support_total": 10000,
        "anchors_per_predicted_class": 500,
        "eps_alpha": 0.20,
        "eps_reference_chunk_size": 128,
    },
    "retrieval": {"tabular_top_k": 5, "synthetic_top_k": 3},
    "constraints": {"immutable_features": {"adult": ["sex"]}},
    "anchor_ball": {
        "cvxpy_solvers": ["CLARABEL"],
        "solver_options": {
            "CLARABEL": {
                "max_iter": 100,
                "tol_gap_abs": 1.0e-3,
                "tol_gap_rel": 1.0e-5,
                "tol_feas": 1.0e-8,
            }
        },
        "accept_statuses": {"CLARABEL": ["optimal"]},
        "beam_width": 8,
        "beam_branch_top_k": 3,
        "beam_max_solver_calls": 32,
    },
    "anchor_pgd": {
        "lambda_schedule": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "steps_per_lambda": 200,
        "restarts": 3,
        "classification_margin": 1.0e-4,
        "projection_iterations": 50,
        "projection_tolerance": 1.0e-6,
        "learning_rate_scale": 0.25,
        "learning_rate_min": 1.0e-3,
        "learning_rate_final_fraction": 0.01,
    },
    "certcf": {
        "device": "auto",
        "lirpa_method": "backward",
        "lirpa_batch_size": 128,
        "classification_margin": 1.0e-4,
        "adaptive_eps": True,
        "adaptive_eps_shrink_factor": 0.5,
        "adaptive_eps_max_shrinks": 8,
        "adaptive_eps_min": 1.0e-6,
        "adaptive_eps_center_tol": 1.0e-6,
        "adaptive_eps_binary_search_steps": 0,
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
    "runtime": {"timeout_seconds_per_query": 120, "rss_sample_interval_seconds": 0.05},
    "analysis": {
        "empirical_sigmas": [0.0, 0.01, 0.03, 0.05, 0.10],
        "empirical_categorical_flip_probabilities": [0.0],
        "empirical_samples_per_sigma": 10,
        "certified_l1_maximum": 5.0,
        "certified_l1_steps": 14,
        "lof_neighbors": 20,
        "isolation_forest_estimators": 200,
    },
    "pilot": {"queries_per_case": 50, "cases": ["heloc", "adult", "depth_01_width_128", "depth_05_width_128"]},
    "artifacts": {"output_dir": "results/lirpa_refinement_ablation"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = _deep_merge(DEFAULT_CONFIG, yaml.safe_load(handle) or {})
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    cases = config["experiment"]["cases"]
    identifiers = [str(case["id"]) for case in cases]
    if not cases or len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment.cases must be non-empty and unique")
    if any(case["kind"] not in {"tabular", "synthetic32"} for case in cases):
        raise ValueError("case kind must be tabular or synthetic32")
    if int(config["data"]["queries_per_case"]) <= 0:
        raise ValueError("queries_per_case must be positive")
    if int(config["data"]["anchors_per_predicted_class"]) <= 0:
        raise ValueError("anchors_per_predicted_class must be positive")
    pgd = config["anchor_pgd"]
    lambdas = [float(value) for value in pgd["lambda_schedule"]]
    if not lambdas or any(value <= 0 for value in lambdas) or lambdas != sorted(lambdas):
        raise ValueError("anchor_pgd.lambda_schedule must be positive and increasing")
    if int(pgd["steps_per_lambda"]) <= 0 or int(pgd["restarts"]) != 3:
        raise ValueError("the official PGD protocol uses three restarts and positive steps")
    if float(config["runtime"]["timeout_seconds_per_query"]) <= 0:
        raise ValueError("timeout_seconds_per_query must be positive")


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    payload.pop("artifacts", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _device(config: dict[str, Any]) -> str:
    requested = str(config["certcf"].get("device", "auto")).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if bool(config["experiment"].get("require_cuda", False)) and requested == "cpu":
        raise RuntimeError("CUDA is required by the official ablation configuration")
    return requested


@contextmanager
def _deadline(seconds: float | None):
    if seconds is None or seconds <= 0 or os.name == "nt":
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError(f"query exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "prepared" / "manifest.json"

    def geometry(self, case_id: str) -> Path:
        return self.root / "prepared" / f"{case_id}.npz"

    def build(self, case_id: str) -> Path:
        return self.root / "cases" / case_id / "certcf" / "build.json"

    def atlas(self, case_id: str) -> Path:
        return self.root / "cases" / case_id / "certcf" / "atlas"

    def query(self, case_id: str, method: str, query_position: int) -> Path:
        return self.root / "cases" / case_id / method / "queries" / f"query_{query_position:04d}.json"

    def combined_case(self, case_id: str) -> Path:
        return self.root / "cases" / case_id / "queries.parquet"

    def certification(self, case_id: str, method: str, query_position: int) -> Path:
        return self.root / "analysis" / "certified_l1" / case_id / method / f"query_{query_position:04d}.json"

    @property
    def combined(self) -> Path:
        return self.root / "lirpa_refinement_queries.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "lirpa_refinement_summary.parquet"

    @property
    def summary_macro(self) -> Path:
        return self.root / "lirpa_refinement_summary_macro.parquet"

    @property
    def empirical(self) -> Path:
        return self.root / "empirical_robustness.parquet"


def project_l1_ball(values: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection onto an L1 ball (Duchi et al.)."""
    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    delta = x - c
    if np.linalg.norm(delta, 1) <= float(radius):
        return x.copy()
    absolute = np.abs(delta)
    ordered = np.sort(absolute)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, len(ordered) + 1)
    active = np.flatnonzero(ordered - (cumulative - radius) / indices > 0)
    theta = (cumulative[active[-1]] - radius) / float(active[-1] + 1)
    return c + np.sign(delta) * np.maximum(absolute - theta, 0.0)


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    vector = np.asarray(values, dtype=np.float64)
    ordered = np.sort(vector)[::-1]
    cssv = np.cumsum(ordered) - 1.0
    active = np.flatnonzero(ordered - cssv / np.arange(1, len(vector) + 1) > 0)
    theta = cssv[active[-1]] / float(active[-1] + 1)
    return np.maximum(vector - theta, 0.0)


def project_feasible_intersection(
    values: np.ndarray,
    *,
    anchor: np.ndarray,
    epsilon: float,
    ohe_slices: Sequence[tuple[int, int]] = (),
    fixed_dims: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
    iterations: int = 50,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Dykstra projection onto L1-ball, OHE simplexes, and fixed coordinates."""
    x = np.asarray(values, dtype=np.float64).copy()
    anchor_array = np.asarray(anchor, dtype=np.float64)
    sets: list[Any] = [lambda z: project_l1_ball(z, anchor_array, float(epsilon))]
    for start, end in ohe_slices:
        def simplex_set(z, start=int(start), end=int(end)):
            out = z.copy()
            out[start:end] = project_simplex(out[start:end])
            return out
        sets.append(simplex_set)
    if fixed_dims is not None and len(fixed_dims):
        dims = np.asarray(fixed_dims, dtype=np.int64)
        vals = np.asarray(fixed_values, dtype=np.float64)[dims]

        def fixed_set(z):
            out = z.copy()
            out[dims] = vals
            return out
        sets.append(fixed_set)
    corrections = [np.zeros_like(x) for _ in sets]
    for _ in range(int(iterations)):
        previous = x.copy()
        for index, projection in enumerate(sets):
            shifted = x + corrections[index]
            x = projection(shifted)
            corrections[index] = shifted - x
        residual = max(float(np.max(np.abs(projection(x) - x))) for projection in sets)
        if (
            np.max(np.abs(x - previous)) <= float(tolerance)
            and residual <= float(tolerance)
        ):
            break
    return x.astype(np.float32)


def _immutable_dims(dataset_name: str, configured: dict[str, list[str]]) -> np.ndarray:
    spec = get_tabular_dataset_spec(dataset_name)
    feature_names = list(spec.feature_names)
    dims: list[int] = []
    for name in configured.get(dataset_name, []):
        index = feature_names.index(name)
        start, end = spec.feature_slices[index]
        dims.extend(range(start, end))
    return np.asarray(sorted(set(dims)), dtype=np.int64)


def _solve_ball_projection(
    query: np.ndarray,
    anchor: np.ndarray,
    epsilon: float,
    *,
    ohe_slices: Sequence[tuple[int, int]],
    fixed_dims: np.ndarray,
    solvers: Sequence[str],
    solver_options: dict[str, dict[str, Any]],
    accept_statuses: dict[str, list[str]],
    fixed_assignments: dict[int, int] | None = None,
) -> tuple[np.ndarray | None, float, str]:
    dimension = len(query)
    z = cp.Variable(dimension)
    constraints: list[Any] = [cp.norm(z - anchor, 1) <= float(epsilon)]
    if len(fixed_dims):
        constraints.append(z[fixed_dims] == query[fixed_dims])
    for block_index, (start, end) in enumerate(ohe_slices):
        assignment = None if fixed_assignments is None else fixed_assignments.get(block_index)
        if assignment is None:
            constraints.extend([cp.sum(z[start:end]) == 1.0, z[start:end] >= 0.0])
        else:
            target = np.zeros(end - start, dtype=np.float64)
            target[int(assignment)] = 1.0
            constraints.append(z[start:end] == target)
    problem = cp.Problem(cp.Minimize(cp.norm(z - query, 1)), constraints)
    for solver in solvers:
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False, **solver_options.get(solver, {}))
        except Exception:
            continue
        if problem.status in accept_statuses.get(solver, ["optimal"]) and z.value is not None:
            candidate = np.asarray(z.value, dtype=np.float32)
            return candidate, float(np.linalg.norm(candidate - query, 1)), str(problem.status)
    return None, float("inf"), str(problem.status or "failed")


def anchor_ball_candidate(
    query: np.ndarray,
    anchor: np.ndarray,
    epsilon: float,
    *,
    ohe_slices: Sequence[tuple[int, int]] = (),
    fixed_dims: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Closest feasible point in one initial anchor ball, without model use."""
    cfg = DEFAULT_CONFIG["anchor_ball"] if config is None else config
    dims = np.asarray([], dtype=np.int64) if fixed_dims is None else np.asarray(fixed_dims, dtype=np.int64)
    relaxed, relaxed_distance, status = _solve_ball_projection(
        query, anchor, epsilon, ohe_slices=ohe_slices, fixed_dims=dims,
        solvers=cfg["cvxpy_solvers"], solver_options=cfg["solver_options"],
        accept_statuses=cfg["accept_statuses"],
    )
    calls = 1
    if relaxed is None or not ohe_slices:
        return relaxed, {"solver_calls": calls, "solver_status": status, "relaxed_distance": relaxed_distance}

    direct = {
        block_index: int(np.argmax(relaxed[start:end]))
        for block_index, (start, end) in enumerate(ohe_slices)
    }
    direct_point, direct_distance, _ = _solve_ball_projection(
        query, anchor, epsilon, ohe_slices=ohe_slices, fixed_dims=dims,
        solvers=cfg["cvxpy_solvers"], solver_options=cfg["solver_options"],
        accept_statuses=cfg["accept_statuses"], fixed_assignments=direct,
    )
    calls += 1
    if direct_point is not None:
        return direct_point, {
            "solver_calls": calls,
            "solver_status": status,
            "relaxed_distance": relaxed_distance,
            "decode_mode": "direct",
        }

    # Build a bounded beam using relaxed simplex scores, then spend the solver
    # budget only on complete exact assignments.  Solving every partial beam
    # node would exhaust a 32-call budget before Adult's eight blocks.
    beam: list[tuple[dict[int, int], float]] = [({}, 0.0)]
    for block_index, (start, end) in enumerate(ohe_slices):
        order = np.argsort(relaxed[start:end])[::-1][: int(cfg["beam_branch_top_k"])]
        expanded: list[tuple[dict[int, int], float]] = []
        for assignments, score in beam:
            for category in order:
                current = {**assignments, block_index: int(category)}
                expanded.append((current, score - float(relaxed[start + int(category)])))
        expanded.sort(key=lambda item: item[1])
        beam = expanded[: int(cfg["beam_width"])]
    best: np.ndarray | None = None
    best_distance = float("inf")
    for assignments, _ in beam:
        if calls >= int(cfg["beam_max_solver_calls"]):
            break
        point, distance, _ = _solve_ball_projection(
            query, anchor, epsilon, ohe_slices=ohe_slices, fixed_dims=dims,
            solvers=cfg["cvxpy_solvers"], solver_options=cfg["solver_options"],
            accept_statuses=cfg["accept_statuses"], fixed_assignments=assignments,
        )
        calls += 1
        if point is not None and distance < best_distance:
            best, best_distance = point, distance
    return best, {
        "solver_calls": calls,
        "solver_status": status,
        "relaxed_distance": relaxed_distance,
        "decode_mode": "beam" if best is not None else "failed",
    }


def _target_margin(logits: torch.Tensor, target: int) -> torch.Tensor:
    target_score = logits[..., int(target)]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    mask[int(target)] = False
    return target_score - logits[..., mask].max(dim=-1).values


def anchor_pgd_candidate(
    query: np.ndarray,
    anchor: np.ndarray,
    epsilon: float,
    *,
    model: torch.nn.Module,
    target: int,
    device: str,
    ball_start: np.ndarray | None = None,
    ohe_slices: Sequence[tuple[int, int]] = (),
    fixed_dims: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Projected targeted optimization in one initial anchor ball."""
    cfg = DEFAULT_CONFIG["anchor_pgd"] if config is None else config
    query_array = np.asarray(query, dtype=np.float32)
    anchor_array = np.asarray(anchor, dtype=np.float32)
    dims = np.asarray([], dtype=np.int64) if fixed_dims is None else np.asarray(fixed_dims, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    random_direction = rng.laplace(size=len(anchor_array))
    random_direction /= max(float(np.linalg.norm(random_direction, 1)), 1.0e-12)
    random_start = anchor_array + random_direction.astype(np.float32) * float(epsilon) * float(rng.random())
    starts = [anchor_array, anchor_array if ball_start is None else np.asarray(ball_start, dtype=np.float32), random_start]
    model = model.eval().to(device)
    mutable_count = max(1, len(query_array) - len(dims))
    initial_lr = max(
        float(cfg["learning_rate_min"]),
        float(cfg["learning_rate_scale"]) * float(epsilon) / mutable_count,
    )
    best: np.ndarray | None = None
    best_distance = float("inf")
    model_evaluations = 0
    valid_iterates = 0

    @torch.no_grad()
    def consider(point: np.ndarray) -> None:
        nonlocal best, best_distance, model_evaluations, valid_iterates
        tensor = torch.as_tensor(point[None, :], dtype=torch.float32, device=device)
        prediction = int(model(tensor).argmax(dim=1).item())
        model_evaluations += 1
        if prediction == int(target):
            valid_iterates += 1
            distance = float(np.linalg.norm(point - query_array, 1))
            if distance < best_distance:
                best, best_distance = point.copy(), distance

    # An operative target-class anchor is a valid fallback whenever it is
    # compatible with query-relative immutable coordinates.
    anchor_compatible = bool(
        not len(dims) or np.allclose(anchor_array[dims], query_array[dims], atol=1.0e-6)
    )
    if anchor_compatible:
        consider(anchor_array)

    total_steps = int(cfg["steps_per_lambda"]) * len(cfg["lambda_schedule"])
    for restart, raw_start in enumerate(starts):
        projected = project_feasible_intersection(
            raw_start,
            anchor=anchor_array,
            epsilon=float(epsilon),
            ohe_slices=ohe_slices,
            fixed_dims=dims,
            fixed_values=query_array,
            iterations=int(cfg["projection_iterations"]),
            tolerance=float(cfg["projection_tolerance"]),
        )
        variable = torch.nn.Parameter(torch.as_tensor(projected, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam([variable], lr=initial_lr)
        step_index = 0
        for penalty in cfg["lambda_schedule"]:
            for _ in range(int(cfg["steps_per_lambda"])):
                fraction = step_index / max(1, total_steps - 1)
                cosine = 0.5 * (1.0 + np.cos(np.pi * fraction))
                fraction_final = float(cfg["learning_rate_final_fraction"])
                optimizer.param_groups[0]["lr"] = initial_lr * (
                    fraction_final + (1.0 - fraction_final) * cosine
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(variable.unsqueeze(0))
                model_evaluations += 1
                margin = _target_margin(logits, int(target))[0]
                hinge = torch.relu(float(cfg["classification_margin"]) - margin)
                proximity = torch.linalg.vector_norm(variable - torch.as_tensor(query_array, device=device), ord=1)
                loss = proximity + float(penalty) * hinge
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    repaired = project_feasible_intersection(
                        variable.detach().cpu().numpy(),
                        anchor=anchor_array,
                        epsilon=float(epsilon),
                        ohe_slices=ohe_slices,
                        fixed_dims=dims,
                        fixed_values=query_array,
                        iterations=int(cfg["projection_iterations"]),
                        tolerance=float(cfg["projection_tolerance"]),
                    )
                    variable.copy_(torch.as_tensor(repaired, device=device))
                if step_index % 10 == 0 or step_index == total_steps - 1:
                    consider(variable.detach().cpu().numpy())
                step_index += 1
        consider(variable.detach().cpu().numpy())

    # Relaxed categorical iterates guide a bounded exact one-hot beam.  Only
    # exact decoded candidates may leave this function.
    if ohe_slices:
        guide = anchor_array if best is None else best.copy()
        beam: list[tuple[dict[int, int], float]] = [({}, 0.0)]
        for block_index, (start, end) in enumerate(ohe_slices):
            categories = np.argsort(guide[start:end])[::-1][:3]
            expanded: list[tuple[dict[int, int], float]] = []
            for assignments, score in beam:
                for category in categories:
                    current = {**assignments, block_index: int(category)}
                    expanded.append((current, score - float(guide[start + int(category)])))
            expanded.sort(key=lambda item: item[1])
            beam = expanded[:8]
        exact_best = anchor_array.copy() if anchor_compatible else None
        exact_best_distance = (
            float(np.linalg.norm(anchor_array - query_array, 1)) if anchor_compatible else float("inf")
        )
        for assignments, _ in beam[:32]:
            exact = guide.copy()
            categorical_dims: list[int] = []
            for block_index, (start, end) in enumerate(ohe_slices):
                exact[start:end] = 0.0
                exact[start + assignments[block_index]] = 1.0
                categorical_dims.extend(range(start, end))
            all_fixed = np.unique(np.concatenate([dims, np.asarray(categorical_dims, dtype=np.int64)]))
            fixed_reference = query_array.copy()
            fixed_reference[np.asarray(categorical_dims, dtype=np.int64)] = exact[np.asarray(categorical_dims, dtype=np.int64)]
            exact = project_feasible_intersection(
                exact,
                anchor=anchor_array,
                epsilon=float(epsilon),
                ohe_slices=(),
                fixed_dims=all_fixed,
                fixed_values=fixed_reference,
                iterations=int(cfg["projection_iterations"]),
                tolerance=float(cfg["projection_tolerance"]),
            )
            if np.linalg.norm(exact - anchor_array, 1) > float(epsilon) + 1.0e-5:
                continue
            prediction = int(_prediction(model, exact, device)[0])
            model_evaluations += 1
            if prediction != int(target):
                continue
            distance = float(np.linalg.norm(exact - query_array, 1))
            if distance < exact_best_distance:
                exact_best, exact_best_distance = exact, distance
        best, best_distance = exact_best, exact_best_distance

    return best, {
        "model_evaluations": int(model_evaluations),
        "valid_iterates": int(valid_iterates),
        "best_distance": float(best_distance),
        "restarts": int(cfg["restarts"]),
        "gradient_steps": int(total_steps * len(starts)),
    }


def _prediction(model: torch.nn.Module, values: np.ndarray, device: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(array), 2048):
            logits = model(torch.from_numpy(array[start : start + 2048]).to(device))
            output.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.int64, copy=False)


def _logits(model: torch.nn.Module, values: np.ndarray, device: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    with torch.no_grad():
        return model(torch.from_numpy(array).to(device)).detach().cpu().numpy()


class LiRPARefinementAblationRunner:
    """Resumable seven-case runner for the three nested ablations."""

    def __init__(self, config: dict[str, Any], *, config_path: str | Path | None = None) -> None:
        validate_config(config)
        self.config = copy.deepcopy(config)
        self.config_path = None if config_path is None else Path(config_path)
        self.fingerprint = config_fingerprint(config)
        self.repository_root = Path(__file__).resolve().parents[2]
        output = Path(config["artifacts"]["output_dir"])
        if not output.is_absolute():
            output = self.repository_root / output
        self.paths = Paths(output.resolve())
        grid_path = Path(config["experiment"]["source_grid_config"])
        if not grid_path.is_absolute():
            grid_path = self.repository_root / grid_path
        grid_config = load_network_grid_config(grid_path)
        grid_output = Path(grid_config["artifacts"]["output_dir"])
        if not grid_output.is_absolute():
            grid_config["artifacts"]["output_dir"] = str((self.repository_root / grid_output).resolve())
        self.grid_runner = NetworkComplexityRunner(grid_config, config_path=grid_path)
        certcf_config = self.config["certcf"]
        self.candidate_parallelism = int(certcf_config.get("candidate_parallelism", 1))
        self.candidate_parallel_backend = str(
            certcf_config.get("candidate_parallel_backend", "process")
        ).lower()
        self.candidate_parallel_warmup = bool(
            certcf_config.get("candidate_parallel_warmup", True)
        )
        self.configure_candidate_parallelism(
            workers=self.candidate_parallelism,
            backend=self.candidate_parallel_backend,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LiRPARefinementAblationRunner":
        return cls(load_config(path), config_path=path)

    @property
    def cases(self) -> list[dict[str, Any]]:
        return list(self.config["experiment"]["cases"])

    def resolve_cases(self, identifiers: Sequence[str] | None) -> list[dict[str, Any]]:
        mapping = {str(case["id"]): case for case in self.cases}
        if identifiers is None:
            return self.cases
        unknown = sorted(set(identifiers) - set(mapping))
        if unknown:
            raise ValueError(f"unknown cases: {unknown}")
        return [mapping[identifier] for identifier in identifiers]

    @staticmethod
    def resolve_methods(methods: Sequence[str] | None) -> list[str]:
        if methods is None:
            return list(METHODS)
        unknown = sorted(set(methods) - set(METHODS))
        if unknown:
            raise ValueError(f"unknown methods: {unknown}")
        return list(dict.fromkeys(methods))

    def configure_candidate_parallelism(
        self,
        *,
        workers: int | None = None,
        backend: str | None = None,
    ) -> None:
        """Set execution-only parallelism without changing experiment geometry."""
        if workers is not None:
            workers = int(workers)
            if workers <= 0:
                raise ValueError("candidate parallelism must be positive")
            self.candidate_parallelism = workers
        if backend is not None:
            backend = str(backend).lower()
            if backend not in {"thread", "process"}:
                raise ValueError("candidate parallel backend must be 'thread' or 'process'")
            self.candidate_parallel_backend = backend

    def _checkpoint(self, case: dict[str, Any]) -> Path:
        if case["kind"] == "synthetic32":
            return self.grid_runner.paths.checkpoint(int(case["depth"]), int(case["width"]))
        return self.repository_root / "checkpoints" / f"{case['dataset']}_classifier" / "best.ckpt"

    def _dataset_source(self, case: dict[str, Any]) -> Path:
        if case["kind"] == "synthetic32":
            return self.grid_runner.paths.dataset
        dataset_name = str(case["dataset"])
        try:
            relative_path = _TABULAR_RAW_FILES[dataset_name]
        except KeyError as exc:
            raise ValueError(f"unsupported tabular dataset: {dataset_name}") from exc
        return self.repository_root / self.config["data"]["data_dir"] / relative_path

    def _load_case(self, case: dict[str, Any], device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module, Any]:
        if case["kind"] == "synthetic32":
            with np.load(self.grid_runner.paths.dataset, allow_pickle=False) as data:
                x_train = data["x_train"].astype(np.float32)
                y_train = data["y_train"].astype(np.int64)
                x_test = data["x_test"].astype(np.float32)
                y_test = data["y_test"].astype(np.int64)
            model = self.grid_runner._load_model(int(case["depth"]), int(case["width"]), device)
            spec = get_tabular_dataset_spec("network_complexity")
            return x_train, y_train, x_test, y_test, model, spec
        registries = create_default_registries()
        dataset = registries["dataset"].create(str(case["dataset"]), data_dir=self.config["data"]["data_dir"], seed=int(self.config["experiment"]["seed"]))
        dataset.load()
        x_train, y_train = dataset.get_train()
        x_test, y_test = dataset.get_test()
        from scripts.benchmark import _build_torch_model_from_checkpoint
        wrapper = _build_torch_model_from_checkpoint(
            str(self._checkpoint(case)), device=device, dataset_module=str(case["dataset"]), dropout=0.2,
        )
        return x_train, y_train, x_test, y_test, wrapper.model, dataset.spec

    def _geometry_hash(self, path: Path) -> str:
        return sha256_file(path)

    def _manifest(self) -> dict[str, Any]:
        if not self.paths.manifest.exists():
            raise RuntimeError("prepare stage has not completed")
        manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != self.fingerprint:
            raise RuntimeError("prepared artifacts use a different configuration")
        for case in self.cases:
            identifier = str(case["id"])
            if manifest["checkpoint_sha256"].get(identifier) != sha256_file(self._checkpoint(case)):
                raise RuntimeError(f"checkpoint changed for {identifier}")
            if manifest["dataset_sha256"].get(identifier) != sha256_file(self._dataset_source(case)):
                raise RuntimeError(f"dataset changed for {identifier}")
            if manifest["geometry_sha256"].get(identifier) != self._geometry_hash(self.paths.geometry(identifier)):
                raise RuntimeError(f"geometry changed for {identifier}")
        return manifest

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        if self.paths.manifest.exists() and not force:
            try:
                manifest = self._manifest()
                _log("[PREPARE] Artifact validi, skip.")
                return manifest
            except RuntimeError:
                pass
        device = _device(self.config)
        seed = int(self.config["experiment"]["seed"])
        query_count = int(self.config["data"]["queries_per_case"])
        anchors_per_class = int(self.config["data"]["anchors_per_predicted_class"])
        checkpoint_hashes: dict[str, str] = {}
        dataset_hashes: dict[str, str] = {}
        geometry_hashes: dict[str, str] = {}
        case_metadata: dict[str, Any] = {}
        for case_position, case in enumerate(self.cases):
            identifier = str(case["id"])
            _log(f"[PREPARE {case_position + 1}/{len(self.cases)}] {identifier}: caricamento dati e modello.")
            x_train, y_train, x_test, y_test, model, spec = self._load_case(case, device)
            rng = np.random.default_rng(np.random.SeedSequence([seed, case_position]))
            if case["kind"] == "tabular":
                support_total = self.config["data"].get("tabular_support_total")
                if support_total is not None:
                    cap = min(int(support_total), len(x_train))
                    support_indices = rng.choice(len(x_train), size=cap, replace=False)
                else:
                    cap = int(self.config["data"]["tabular_support_per_true_class"])
                    selected_parts: list[np.ndarray] = []
                    for label in np.unique(y_train):
                        indices = np.flatnonzero(y_train == label)
                        take = min(cap, len(indices))
                        selected_parts.append(rng.choice(indices, size=take, replace=False))
                    support_indices = np.concatenate(selected_parts)
                    rng.shuffle(support_indices)
            else:
                cap = min(int(self.config["data"]["synthetic_support_total"]), len(x_train))
                support_indices = rng.choice(len(x_train), size=cap, replace=False)
            x_support = np.asarray(x_train[support_indices], dtype=np.float32)
            y_support_true = np.asarray(y_train[support_indices], dtype=np.int64)
            y_support_pred = _prediction(model, x_support, device)
            anchor_parts: list[np.ndarray] = []
            for label in (0, 1):
                candidates = np.flatnonzero(y_support_pred == label)
                take = min(anchors_per_class, len(candidates))
                if take == 0:
                    raise RuntimeError(
                        f"{identifier}: no predicted-class-{label} support points"
                    )
                anchor_parts.append(rng.choice(candidates, size=take, replace=False))
            anchor_support_positions = np.concatenate(anchor_parts)
            rng.shuffle(anchor_support_positions)
            x_anchor = x_support[anchor_support_positions]
            y_anchor_pred = y_support_pred[anchor_support_positions]
            strategy = NearestOppositeClassClearanceStrategy(
                alpha=float(self.config["data"]["eps_alpha"]),
                chunk_size=int(self.config["data"]["eps_reference_chunk_size"]),
            )
            strategy.set_reference(x_support, y_support_pred)
            eps_initial = strategy.compute_eps(x_anchor, y_anchor_pred, norm=1).astype(np.float32)
            case_query_count = min(query_count, len(x_test))
            query_indices = np.sort(
                rng.choice(len(x_test), size=case_query_count, replace=False)
            )
            x_query = np.asarray(x_test[query_indices], dtype=np.float32)
            y_query_true = np.asarray(y_test[query_indices], dtype=np.int64)
            y_query_pred = _prediction(model, x_query, device)
            target = (1 - y_query_pred).astype(np.int64)
            checkpoint = self._checkpoint(case)
            _atomic_npz(
                self.paths.geometry(identifier),
                x_support=x_support,
                y_support_true=y_support_true,
                y_support_pred=y_support_pred,
                support_indices=np.asarray(support_indices, dtype=np.int64),
                x_anchor=x_anchor,
                y_anchor_pred=y_anchor_pred,
                anchor_support_positions=np.asarray(anchor_support_positions, dtype=np.int64),
                eps_initial=eps_initial,
                query_indices=query_indices,
                x_query=x_query,
                y_query_true=y_query_true,
                y_query_pred=y_query_pred,
                target=target,
            )
            checkpoint_hashes[identifier] = sha256_file(checkpoint)
            dataset_hashes[identifier] = sha256_file(self._dataset_source(case))
            geometry_hashes[identifier] = self._geometry_hash(self.paths.geometry(identifier))
            case_metadata[identifier] = {
                "kind": case["kind"],
                "dataset": case.get("dataset", "network_complexity"),
                "depth": case.get("depth"),
                "width": case.get("width"),
                "n_features": int(x_train.shape[1]),
                "support_rows": int(len(x_support)),
                "support_sampling": "random_without_replacement",
                "support_cap_total": (
                    int(self.config["data"]["tabular_support_total"])
                    if case["kind"] == "tabular"
                    and self.config["data"].get("tabular_support_total") is not None
                    else None
                ),
                "anchor_rows": int(len(x_anchor)),
                "anchor_sampling": "random_without_replacement_within_predicted_class",
                "anchor_cap_per_predicted_class": anchors_per_class,
                "query_rows": int(len(x_query)),
                "anchor_class_counts": {str(label): int(np.sum(y_anchor_pred == label)) for label in (0, 1)},
                "eps_initial_min": float(eps_initial.min()),
                "eps_initial_median": float(np.median(eps_initial)),
                "eps_initial_max": float(eps_initial.max()),
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "ohe_slices": [list(value) for value in spec.categorical_slices],
            }
        manifest = {
            "status": "complete",
            "config_fingerprint": self.fingerprint,
            "checkpoint_sha256": checkpoint_hashes,
            "dataset_sha256": dataset_hashes,
            "geometry_sha256": geometry_hashes,
            "cases": case_metadata,
            "methods": list(METHODS),
        }
        _atomic_json(manifest, self.paths.manifest)
        _log(
            f"[PREPARE] Complete: {len(self.cases)} cases, "
            f"up to {query_count} queries/case."
        )
        return manifest

    def _geometry(self, case_id: str) -> dict[str, np.ndarray]:
        self._manifest()
        with np.load(self.paths.geometry(case_id), allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    def _top_k(self, case: dict[str, Any]) -> int:
        return int(self.config["retrieval"]["tabular_top_k" if case["kind"] == "tabular" else "synthetic_top_k"])

    def _fixed_dims(self, case: dict[str, Any]) -> np.ndarray:
        dataset_name = str(case.get("dataset", "network_complexity"))
        return _immutable_dims(dataset_name, self.config["constraints"]["immutable_features"])

    def _ohe_slices(self, case: dict[str, Any]) -> tuple[tuple[int, int], ...]:
        spec = get_tabular_dataset_spec(str(case.get("dataset", "network_complexity")))
        return tuple((int(start), int(end)) for start, end in spec.categorical_slices)

    def _make_certcf(self, case: dict[str, Any], model: torch.nn.Module, geometry: dict[str, np.ndarray], device: str, *, bounds_dir: Path | None = None) -> CertCF:
        cfg = self.config["certcf"]
        strategy = NearestOppositeClassClearanceStrategy(
            alpha=float(self.config["data"]["eps_alpha"]),
            chunk_size=int(self.config["data"]["eps_reference_chunk_size"]),
        )
        strategy.set_reference(geometry["x_support"], geometry["y_support_pred"])
        return CertCF(
            model=TorchModelWrapper(model=model, device=device),
            norm=1,
            distance_norm=1,
            lirpa_method=str(cfg["lirpa_method"]),
            eps_strategy=strategy,
            batch_size=int(cfg["lirpa_batch_size"]),
            epsilon_parallelism=int(cfg.get("epsilon_parallelism", 1)),
            build_parallelism=int(cfg.get("build_parallelism", 1)),
            reuse_lirpa_graph=bool(cfg.get("reuse_lirpa_graph", False)),
            ohe_slices=list(self._ohe_slices(case)) or None,
            bounds_checkpoint_dir=bounds_dir,
            default_query_method="nearest_anchor",
            query_k_candidates=self._top_k(case),
            solver_maxiter=int(cfg["solver_maxiter"]),
            candidate_parallelism=self.candidate_parallelism,
            candidate_parallel_backend=self.candidate_parallel_backend,
            cvxpy_solvers=list(cfg["cvxpy_solvers"]),
            cvxpy_solver_options=copy.deepcopy(cfg["cvxpy_solver_options"]),
            cvxpy_accept_statuses=copy.deepcopy(cfg["cvxpy_accept_statuses"]),
            classification_margin=float(cfg["classification_margin"]),
            adaptive_eps=bool(cfg["adaptive_eps"]),
            adaptive_eps_shrink_factor=float(cfg["adaptive_eps_shrink_factor"]),
            adaptive_eps_max_shrinks=int(cfg["adaptive_eps_max_shrinks"]),
            adaptive_eps_min=float(cfg["adaptive_eps_min"]),
            adaptive_eps_center_tol=float(cfg["adaptive_eps_center_tol"]),
            adaptive_eps_binary_search_steps=int(cfg["adaptive_eps_binary_search_steps"]),
            ohe_decode_mode="beam_then_exact",
            decode_beam_width=int(self.config["anchor_ball"]["beam_width"]),
            decode_beam_branch_top_k=int(self.config["anchor_ball"]["beam_branch_top_k"]),
            decode_beam_max_solver_calls=int(self.config["anchor_ball"]["beam_max_solver_calls"]),
            sparsity_penalty="none",
            fixed_dims=self._fixed_dims(case),
            immutable_features=self.config["constraints"]["immutable_features"].get(str(case.get("dataset", "")), []),
            k_per_class=None,
            subsample_method="random",
            random_seed=int(self.config["experiment"]["seed"]),
        )

    def _valid_build(self, case: dict[str, Any]) -> bool:
        path = self.paths.build(str(case["id"]))
        atlas_manifest = self.paths.atlas(str(case["id"])) / "manifest.json"
        if not path.exists() or not atlas_manifest.exists():
            return False
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value.get("status") == "complete" and value.get("run_fingerprint") == self._run_fingerprint(case)
        except (OSError, ValueError):
            return False

    def _run_fingerprint(self, case: dict[str, Any]) -> str:
        identifier = str(case["id"])
        payload = {
            "config": self.fingerprint,
            "geometry": sha256_file(self.paths.geometry(identifier)),
            "checkpoint": sha256_file(self._checkpoint(case)),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def build(self, case: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
        self.prepare()
        identifier = str(case["id"])
        if self._valid_build(case) and not force:
            _log(f"[BUILD] {identifier}: atlas valido, skip.")
            return json.loads(self.paths.build(identifier).read_text(encoding="utf-8"))
        geometry = self._geometry(identifier)
        device = _device(self.config)
        _, _, _, _, model, _ = self._load_case(case, device)
        bounds_dir = self.paths.root / "cases" / identifier / "certcf" / "partial_bounds"
        partial_state = bounds_dir / "state.json"
        expected_fingerprint = self._run_fingerprint(case)
        if force and bounds_dir.exists():
            shutil.rmtree(bounds_dir)
        if partial_state.exists():
            state = json.loads(partial_state.read_text(encoding="utf-8"))
            if state.get("run_fingerprint") != expected_fingerprint:
                shutil.rmtree(bounds_dir)
        bounds_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json({"run_fingerprint": expected_fingerprint}, partial_state)
        if self.paths.atlas(identifier).exists():
            shutil.rmtree(self.paths.atlas(identifier))
        method = self._make_certcf(case, model, geometry, device, bounds_dir=bounds_dir)
        _log(f"[BUILD] {identifier}: {len(geometry['x_anchor'])} anchor condivisi, device={device}.")
        interval = float(self.config["runtime"]["rss_sample_interval_seconds"])
        try:
            with PhaseResourceMonitor("build", device, interval) as monitor:
                method.fit(geometry["x_anchor"], geometry["y_anchor_pred"])
        except BaseException as exc:
            _atomic_json(
                {
                    "status": "failed",
                    "case_id": identifier,
                    "run_fingerprint": expected_fingerprint,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                self.paths.build(identifier),
            )
            raise
        atlas = method.atlas
        if atlas is None or atlas.bounds is None:
            raise RuntimeError("CertCF did not produce an atlas")
        atlas.save_bounds(self.paths.atlas(identifier))
        bounds = [atlas.bounds[label] for label in atlas.class_labels]
        initial = np.concatenate([np.asarray(value["eps_initial"]) for value in bounds])
        final = np.concatenate([np.asarray(value["eps"]) for value in bounds])
        shrinks = np.concatenate([np.asarray(value["adaptive_eps_n_shrinks"]) for value in bounds])
        metadata = {
            "status": "complete",
            "case_id": identifier,
            "run_fingerprint": self._run_fingerprint(case),
            "attempted_anchor_count": int(sum(int(np.asarray(value["attempted_anchor_count"]).item()) for value in bounds)),
            "atlas_region_count": int(sum(len(value["X"]) for value in bounds)),
            "eps_initial_median": float(np.median(initial)),
            "eps_final_median": float(np.median(final)),
            "eps_retention_median": float(np.median(final / np.maximum(initial, 1.0e-30))),
            "adaptive_shrinks_mean": float(np.mean(shrinks)),
            **monitor.metrics,
        }
        _atomic_json(metadata, self.paths.build(identifier))
        _log(f"[BUILD] {identifier}: {metadata['atlas_region_count']} regions in {metadata['build_wall_time_s']:.1f}s.")
        return metadata

    def build_all(self, cases: Iterable[dict[str, Any]] | None = None, *, force: bool = False) -> list[dict[str, Any]]:
        selected = self.cases if cases is None else list(cases)
        return [self.build(case, force=force) for case in selected]

    def _load_certcf(self, case: dict[str, Any], model: torch.nn.Module, geometry: dict[str, np.ndarray], device: str) -> CertCF:
        if not self._valid_build(case):
            raise RuntimeError(f"missing valid CertCF build for {case['id']}")
        atlas_dir = self.paths.atlas(str(case["id"]))
        manifest = json.loads((atlas_dir / "manifest.json").read_text(encoding="utf-8"))
        bounds: dict[int, dict[str, np.ndarray]] = {}
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for raw_label in manifest["class_labels"]:
            label = int(raw_label)
            with np.load(atlas_dir / manifest["files"][str(label)], allow_pickle=False) as data:
                bounds[label] = {key: data[key] for key in data.files}
            x_parts.append(bounds[label]["X"].astype(np.float32))
            y_parts.append(np.full(len(bounds[label]["X"]), label, dtype=np.int64))
        dataset = TensorDataset(torch.from_numpy(np.concatenate(x_parts)), torch.from_numpy(np.concatenate(y_parts)))
        cfg = self.config["certcf"]
        atlas = CertCFAtlas(
            model, dataset, device,
            norm=1, distance_norm=1, lirpa_method=str(cfg["lirpa_method"]),
            epsilon_parallelism=int(cfg.get("epsilon_parallelism", 1)),
            build_parallelism=int(cfg.get("build_parallelism", 1)),
            reuse_lirpa_graph=bool(cfg.get("reuse_lirpa_graph", False)),
            ohe_slices=list(self._ohe_slices(case)) or None,
            default_query_method="nearest_anchor", solver_maxiter=int(cfg["solver_maxiter"]),
            candidate_parallelism=self.candidate_parallelism,
            candidate_parallel_backend=self.candidate_parallel_backend,
            cvxpy_solvers=list(cfg["cvxpy_solvers"]),
            cvxpy_solver_options=copy.deepcopy(cfg["cvxpy_solver_options"]),
            cvxpy_accept_statuses=copy.deepcopy(cfg["cvxpy_accept_statuses"]),
            classification_margin=float(cfg["classification_margin"]),
            ohe_decode_mode="beam_then_exact",
            decode_beam_width=int(self.config["anchor_ball"]["beam_width"]),
            decode_beam_branch_top_k=int(self.config["anchor_ball"]["beam_branch_top_k"]),
            decode_beam_max_solver_calls=int(self.config["anchor_ball"]["beam_max_solver_calls"]),
            sparsity_penalty="none",
        )
        atlas.bounds = bounds
        method = self._make_certcf(case, model, geometry, device)
        method.atlas = atlas
        method._is_fitted = True
        return method

    @staticmethod
    def _strict_ohe_valid(point: np.ndarray, slices: Sequence[tuple[int, int]], tolerance: float = 1.0e-5) -> bool:
        for start, end in slices:
            block = np.asarray(point[start:end])
            if abs(float(block.sum()) - 1.0) > tolerance:
                return False
            if np.sum(block >= 1.0 - tolerance) != 1 or np.any((block > tolerance) & (block < 1.0 - tolerance)):
                return False
        return True

    def _row(
        self,
        *,
        case: dict[str, Any],
        query_position: int,
        method: str,
        geometry: dict[str, np.ndarray],
        model: torch.nn.Module,
        device: str,
        candidate: np.ndarray | None,
        runtime_seconds: float,
        timed_out: bool,
        error: str | None,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        query = geometry["x_query"][query_position]
        target = int(geometry["target"][query_position])
        fixed_dims = self._fixed_dims(case)
        ohe_slices = self._ohe_slices(case)
        spec = get_tabular_dataset_spec(str(case.get("dataset", "network_complexity")))
        finite = candidate is not None and bool(np.isfinite(candidate).all())
        predicted = None
        target_margin = np.nan
        if finite:
            logits = _logits(model, candidate, device)[0]
            predicted = int(logits.argmax())
            target_margin = float(logits[target] - logits[1 - target])
        ohe_valid = bool(finite and self._strict_ohe_valid(candidate, ohe_slices)) if ohe_slices else bool(finite)
        immutable_valid = bool(
            finite and (not len(fixed_dims) or np.allclose(candidate[fixed_dims], query[fixed_dims], atol=1.0e-5))
        )
        feasible = bool(finite and ohe_valid and immutable_valid)
        success = bool(feasible and predicted == target)
        difference = np.full_like(query, np.nan, dtype=np.float64) if not finite else np.abs(candidate - query)
        mad_l1 = np.nan
        redundancy = np.nan
        if finite:
            weights = np.ones(len(query), dtype=np.float64)
            for index, feature_type in enumerate(spec.ohe_feature_types):
                if feature_type != "numerical":
                    continue
                column = geometry["x_support"][:, index].astype(np.float64)
                mad = float(np.median(np.abs(column - np.median(column))))
                weights[index] = mad if mad > 0.0 else 1.0
            normalized = difference / weights
            normalized[
                np.asarray([value == "categorical" for value in spec.ohe_feature_types])
            ] = (
                difference[
                    np.asarray([value == "categorical" for value in spec.ohe_feature_types])
                ]
                > 1.0e-6
            )
            mad_l1 = float(np.mean(normalized))
            if success:
                changed_groups = [
                    (start, end)
                    for start, end in spec.feature_slices
                    if np.any(difference[start:end] > 1.0e-6)
                ]
                if changed_groups:
                    reverted = np.repeat(candidate[None, :], len(changed_groups), axis=0)
                    for row_index, (start, end) in enumerate(changed_groups):
                        reverted[row_index, start:end] = query[start:end]
                    redundancy = float(np.mean(_prediction(model, reverted, device) == target))
                else:
                    redundancy = 0.0
        row: dict[str, Any] = {
            "case_id": str(case["id"]),
            "kind": str(case["kind"]),
            "dataset": str(case.get("dataset", "network_complexity")),
            "depth": case.get("depth"),
            "width": case.get("width"),
            "query_position": int(query_position),
            "query_index": int(geometry["query_indices"][query_position]),
            "method": method,
            "source_class": int(geometry["y_query_pred"][query_position]),
            "true_class": int(geometry["y_query_true"][query_position]),
            "target_class": target,
            "predicted_class": predicted,
            "candidate_found": bool(candidate is not None),
            "success": success,
            "target_valid": bool(predicted == target) if predicted is not None else False,
            "domain_feasible": feasible,
            "ohe_valid": ohe_valid,
            "immutable_valid": immutable_valid,
            "timed_out": bool(timed_out),
            "error": error,
            "runtime_seconds": float(runtime_seconds),
            "l1_distance": float(np.nansum(difference)) if finite else np.nan,
            "l2_distance": float(np.linalg.norm(difference, 2)) if finite else np.nan,
            "l0_changed": int(np.count_nonzero(difference > 1.0e-6)) if finite else np.nan,
            "mad_l1_distance": mad_l1,
            "redundancy": redundancy,
            "target_margin": target_margin,
            "run_fingerprint": self._run_fingerprint(case),
            **{f"diag__{key}": value for key, value in diagnostics.items()},
        }
        for index, value in enumerate(query):
            row[f"x_orig_{index}"] = float(value)
            row[f"x_cf_{index}"] = float(candidate[index]) if finite else np.nan
        return row

    def _query_valid(self, case: dict[str, Any], method: str, position: int) -> bool:
        path = self.paths.query(str(case["id"]), method, position)
        if not path.exists():
            return False
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            return row.get("run_fingerprint") == self._run_fingerprint(case) and row.get("method") == method
        except (OSError, ValueError):
            return False

    def _compatible_order(self, case: dict[str, Any], geometry: dict[str, np.ndarray], query_position: int) -> np.ndarray:
        query = geometry["x_query"][query_position]
        target = int(geometry["target"][query_position])
        indices = np.flatnonzero(geometry["y_anchor_pred"] == target)
        fixed_dims = self._fixed_dims(case)
        if len(fixed_dims):
            compatible = np.all(np.isclose(geometry["x_anchor"][indices][:, fixed_dims], query[fixed_dims], atol=1.0e-6), axis=1)
            indices = indices[compatible]
        distances = np.linalg.norm(geometry["x_anchor"][indices] - query[None, :], ord=1, axis=1)
        return indices[np.argsort(distances)]

    def _baseline_query(self, case: dict[str, Any], method: str, position: int, geometry: dict[str, np.ndarray], model: torch.nn.Module, device: str) -> tuple[np.ndarray | None, dict[str, Any]]:
        query = geometry["x_query"][position]
        target = int(geometry["target"][position])
        order = self._compatible_order(case, geometry, position)
        top_k = self._top_k(case)
        primary = order[:top_k]
        fallback = order[top_k:]
        best: np.ndarray | None = None
        best_distance = float("inf")
        total_solver_calls = 0
        total_model_evaluations = 0
        considered = 0
        baseline_timed_out = False

        if method == "anchor_pgd" and len(order):
            # Preserve the nearest compatible target-class anchor if the
            # fixed wall-clock budget interrupts gradient optimization.
            best = geometry["x_anchor"][int(order[0])].copy()
            best_distance = float(np.linalg.norm(best - query, 1))

        def evaluate_anchor(anchor_index: int) -> bool:
            nonlocal best, best_distance, total_solver_calls, total_model_evaluations, considered
            considered += 1
            anchor = geometry["x_anchor"][anchor_index]
            epsilon = float(geometry["eps_initial"][anchor_index])
            ball_point, ball_diag = anchor_ball_candidate(
                query, anchor, epsilon, ohe_slices=self._ohe_slices(case),
                fixed_dims=self._fixed_dims(case), config=self.config["anchor_ball"],
            )
            total_solver_calls += int(ball_diag.get("solver_calls", 0))
            if method == "anchor_ball":
                point = ball_point
            else:
                point, pgd_diag = anchor_pgd_candidate(
                    query, anchor, epsilon, model=model, target=target, device=device,
                    ball_start=ball_point, ohe_slices=self._ohe_slices(case),
                    fixed_dims=self._fixed_dims(case), config=self.config["anchor_pgd"],
                    seed=int(self.config["experiment"]["seed"]) + position * 1009 + int(anchor_index),
                )
                total_model_evaluations += int(pgd_diag.get("model_evaluations", 0))
            if point is None:
                return False
            if method == "anchor_pgd" and int(_prediction(model, point, device)[0]) != target:
                return False
            distance = float(np.linalg.norm(point - query, 1))
            if distance < best_distance:
                best, best_distance = point, distance
            return True

        fallback_used = False
        try:
            for anchor_index in primary:
                evaluate_anchor(int(anchor_index))
            if best is None:
                fallback_used = True
                for anchor_index in fallback:
                    if evaluate_anchor(int(anchor_index)):
                        break
        except TimeoutError:
            baseline_timed_out = True
        return best, {
            "query_k_candidates": int(top_k),
            "candidates_considered": int(considered),
            "fallback_used": bool(fallback_used),
            "solver_calls": int(total_solver_calls),
            "model_evaluations": int(total_model_evaluations),
            "timed_out_with_incumbent": bool(baseline_timed_out),
        }

    def benchmark_method(
        self,
        case: dict[str, Any],
        method: str,
        *,
        query_positions: Sequence[int] | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        self.prepare()
        if method == "certcf":
            self.build(case, force=False)
        identifier = str(case["id"])
        geometry = self._geometry(identifier)
        positions = list(range(len(geometry["x_query"]))) if query_positions is None else [int(value) for value in query_positions]
        device = _device(self.config)
        _, _, _, _, model, _ = self._load_case(case, device)
        certcf = self._load_certcf(case, model, geometry, device) if method == "certcf" else None
        timeout = float(self.config["runtime"]["timeout_seconds_per_query"])
        rows: list[dict[str, Any]] = []
        if (
            certcf is not None
            and self.candidate_parallelism > 1
            and self.candidate_parallel_backend == "process"
            and self.candidate_parallel_warmup
            and positions
        ):
            warmup_started = time.perf_counter()
            warmup = certcf.generate(
                geometry["x_query"][positions[0]],
                target_class=int(geometry["target"][positions[0]]),
            )
            warmup_seconds = time.perf_counter() - warmup_started
            _log(
                f"[CERTCF WARMUP] {identifier}: {warmup_seconds:.3f}s, "
                f"workers={self.candidate_parallelism}, success={bool(warmup.success)}."
            )
        for ordinal, position in enumerate(positions, start=1):
            path = self.paths.query(identifier, method, position)
            if self._query_valid(case, method, position) and not force:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            _log(f"[{method.upper()} {ordinal}/{len(positions)}] {identifier} query={position}.")
            candidate: np.ndarray | None = None
            diagnostics: dict[str, Any] = {}
            error: str | None = None
            timed_out = False
            started = time.perf_counter()
            try:
                with _deadline(timeout):
                    if method == "certcf":
                        result = certcf.generate(
                            geometry["x_query"][position],
                            target_class=int(geometry["target"][position]),
                        )
                        candidate = result.x_cf
                        diagnostics = dict(result.metadata)
                    else:
                        candidate, diagnostics = self._baseline_query(case, method, position, geometry, model, device)
                        timed_out = bool(diagnostics.get("timed_out_with_incumbent", False))
            except TimeoutError as exc:
                timed_out = True
                error = f"TimeoutError: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            runtime = time.perf_counter() - started
            row = self._row(
                case=case, query_position=position, method=method, geometry=geometry,
                model=model, device=device, candidate=candidate,
                runtime_seconds=runtime, timed_out=timed_out, error=error,
                diagnostics=diagnostics,
            )
            _atomic_json(row, path)
            rows.append(row)
        if certcf is not None and certcf.atlas is not None:
            certcf.atlas.close_candidate_process_pool()
        return rows

    def benchmark(
        self,
        cases: Iterable[dict[str, Any]] | None = None,
        methods: Sequence[str] | None = None,
        *,
        query_positions: Sequence[int] | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        selected_cases = self.cases if cases is None else list(cases)
        selected_methods = self.resolve_methods(methods)
        rows: list[dict[str, Any]] = []
        for case in selected_cases:
            for method in selected_methods:
                rows.extend(
                    self.benchmark_method(
                        case, method, query_positions=query_positions, force=force
                    )
                )
        return rows

    def pilot(self, *, force: bool = False) -> dict[str, Any]:
        cases = self.resolve_cases(self.config["pilot"]["cases"])
        count = int(self.config["pilot"]["queries_per_case"])
        self.prepare(force=force)
        self.build_all(cases, force=force)
        rows: list[dict[str, Any]] = []
        for case in cases:
            available = len(self._geometry(str(case["id"]))["x_query"])
            rows.extend(
                self.benchmark(
                    [case],
                    query_positions=list(range(min(count, available))),
                    force=force,
                )
            )
        return {"cases": len(cases), "queries_per_case": count, "rows": len(rows)}

    def aggregate(self, *, allow_partial: bool = False) -> pd.DataFrame:
        self._manifest()
        case_frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for case in self.cases:
            identifier = str(case["id"])
            expected = len(self._geometry(identifier)["x_query"])
            rows: list[dict[str, Any]] = []
            for method in METHODS:
                for position in range(expected):
                    path = self.paths.query(identifier, method, position)
                    if not self._query_valid(case, method, position):
                        missing.append(f"{identifier}/{method}/{position}")
                        continue
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
            if rows:
                frame = pd.DataFrame(rows)
                if frame.duplicated(["case_id", "method", "query_position"]).any():
                    raise RuntimeError(f"duplicate rows in {identifier}")
                _atomic_parquet(frame, self.paths.combined_case(identifier))
                case_frames.append(frame)
        if missing and not allow_partial:
            raise RuntimeError(
                f"missing {len(missing)} query artifacts; first missing: {missing[0]}"
            )
        combined = pd.concat(case_frames, ignore_index=True, sort=False) if case_frames else pd.DataFrame()
        if not combined.empty:
            _atomic_parquet(combined, self.paths.combined)
        _log(f"[AGGREGATE] {len(combined)} rows for {combined['case_id'].nunique() if not combined.empty else 0} cases.")
        return combined

    def _attach_manifoldness(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["log10_lof"] = np.nan
        output["isolation_forest_score"] = np.nan
        analysis = self.config["analysis"]
        for case in self.cases:
            identifier = str(case["id"])
            geometry = self._geometry(identifier)
            successful = (output["case_id"] == identifier) & output["success"].astype(bool)
            positions = np.flatnonzero(successful.to_numpy())
            if not len(positions):
                continue
            dimension = geometry["x_support"].shape[1]
            columns = [f"x_cf_{index}" for index in range(dimension)]
            values = output.iloc[positions][columns].to_numpy(dtype=np.float32)
            support = geometry["x_support"].astype(np.float32)
            neighbors = max(2, min(int(analysis["lof_neighbors"]), len(support) - 1))
            lof = LocalOutlierFactor(n_neighbors=neighbors, novelty=True, n_jobs=-1).fit(support)
            iforest = IsolationForest(
                n_estimators=int(analysis["isolation_forest_estimators"]),
                contamination="auto", random_state=0, n_jobs=-1,
            ).fit(support)
            positive_lof = np.maximum(-lof.score_samples(values), 1.0e-12)
            output.loc[output.index[positions], "log10_lof"] = np.log10(positive_lof)
            output.loc[output.index[positions], "isolation_forest_score"] = iforest.score_samples(values)
        return output

    def _empirical_robustness(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        device = _device(self.config)
        analysis = self.config["analysis"]
        base_seed = int(self.config["experiment"]["seed"])
        for case_position, case in enumerate(self.cases):
            identifier = str(case["id"])
            geometry = self._geometry(identifier)
            _, _, _, _, model, spec = self._load_case(case, device)
            dimension = geometry["x_support"].shape[1]
            numerical = np.asarray(
                [index for index, feature_type in enumerate(spec.ohe_feature_types) if feature_type == "numerical"],
                dtype=np.int64,
            )
            columns = [f"x_cf_{index}" for index in range(dimension)]
            subset = frame[(frame["case_id"] == identifier) & frame["success"].astype(bool)]
            for record in subset.to_dict(orient="records"):
                point = np.asarray([record[column] for column in columns], dtype=np.float32)
                for sigma in analysis["empirical_sigmas"]:
                    for flip_probability in analysis.get(
                        "empirical_categorical_flip_probabilities", [0.0]
                    ):
                        rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    base_seed,
                                    case_position,
                                    int(record["query_position"]),
                                    int(round(float(sigma) * 1.0e6)),
                                    int(round(float(flip_probability) * 1.0e6)),
                                ]
                            )
                        )
                        perturbations = np.repeat(
                            point[None, :],
                            int(analysis["empirical_samples_per_sigma"]),
                            axis=0,
                        )
                        perturbations[:, numerical] += rng.normal(
                            0.0,
                            float(sigma),
                            size=(len(perturbations), len(numerical)),
                        ).astype(np.float32)
                        if float(flip_probability) > 0.0:
                            for start, end in spec.categorical_slices:
                                width = int(end - start)
                                if width <= 1:
                                    continue
                                flips = rng.random(len(perturbations)) < float(
                                    flip_probability
                                )
                                if not flips.any():
                                    continue
                                current = np.argmax(
                                    perturbations[flips, start:end], axis=1
                                )
                                alternatives = rng.integers(0, width - 1, size=len(current))
                                alternatives += alternatives >= current
                                perturbations[flips, start:end] = 0.0
                                perturbations[
                                    np.flatnonzero(flips), start + alternatives
                                ] = 1.0
                        predictions = _prediction(model, perturbations, device)
                        preserved = predictions == int(record["target_class"])
                        rows.append(
                            {
                                "case_id": identifier,
                                "method": record["method"],
                                "query_position": int(record["query_position"]),
                                "sigma": float(sigma),
                                "categorical_flip_probability": float(
                                    flip_probability
                                ),
                                "target_rate": float(np.mean(preserved)),
                                "all_preserved": bool(np.all(preserved)),
                            }
                        )
        return pd.DataFrame(rows)

    def _certified_l1_radii(self, frame: pd.DataFrame) -> pd.Series:
        """Common post-hoc LiRPA binary search for every successful output."""
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        from certcf.certification.wrapping import WrappedModel
        from counterfactuals.methods.certcf import _strip_dropout_modules

        device = _device(self.config)
        maximum = float(self.config["analysis"]["certified_l1_maximum"])
        steps = int(self.config["analysis"]["certified_l1_steps"])
        radii = pd.Series(np.nan, index=frame.index, dtype=float)
        pending = pd.Series(frame["success"].astype(bool).to_numpy(), index=frame.index)
        for row_index, record in frame[pending].iterrows():
            path = self.paths.certification(
                str(record["case_id"]), str(record["method"]), int(record["query_position"])
            )
            if not path.exists():
                continue
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            case = next(value for value in self.cases if str(value["id"]) == str(record["case_id"]))
            if cached.get("run_fingerprint") == self._run_fingerprint(case):
                radii.loc[row_index] = float(cached["certified_l1_radius"])
                pending.loc[row_index] = False
        for case in self.cases:
            identifier = str(case["id"])
            geometry = self._geometry(identifier)
            _, _, _, _, raw_model, _ = self._load_case(case, device)
            model = _strip_dropout_modules(raw_model).eval().to(device)
            dimension = geometry["x_support"].shape[1]
            columns = [f"x_cf_{index}" for index in range(dimension)]
            mask = (frame["case_id"] == identifier) & pending
            for target in (0, 1):
                local_indices = frame.index[mask & (frame["target_class"] == target)].to_numpy()
                if not len(local_indices):
                    continue
                centers_all = frame.loc[local_indices, columns].to_numpy(dtype=np.float32)
                for batch_start in range(0, len(local_indices), 128):
                    batch_indices = local_indices[batch_start : batch_start + 128]
                    centers_np = centers_all[batch_start : batch_start + 128].copy()
                    centers = torch.as_tensor(centers_np, device=device)
                    wrapped = WrappedModel(model, int(target), torch.device(device), n_labels=2).to(device).eval()
                    low = np.zeros(len(centers_np), dtype=np.float32)
                    high = np.full(len(centers_np), maximum, dtype=np.float32)

                    def certified(radius: np.ndarray) -> np.ndarray:
                        perturbation = PerturbationLpNorm(
                            norm=1.0,
                            eps=torch.as_tensor(radius.reshape(-1, 1), dtype=torch.float32, device=device),
                        )
                        bounded_x = BoundedTensor(centers, perturbation)
                        bounded_model = BoundedModule(wrapped, bounded_x, device=device)
                        lower, _ = bounded_model.compute_bounds(x=(bounded_x,), method=str(self.config["certcf"]["lirpa_method"]))
                        return torch.all(lower >= 0.0, dim=1).detach().cpu().numpy()

                    maximum_ok = certified(high)
                    low[maximum_ok] = maximum
                    active = ~maximum_ok
                    for _ in range(steps):
                        if not active.any():
                            break
                        middle = 0.5 * (low + high)
                        okay = certified(middle)
                        low[okay] = middle[okay]
                        high[~okay] = middle[~okay]
                    radii.loc[batch_indices] = low
                    for output_index, radius in zip(batch_indices, low):
                        record = frame.loc[output_index]
                        _atomic_json(
                            {
                                "run_fingerprint": self._run_fingerprint(case),
                                "case_id": identifier,
                                "method": str(record["method"]),
                                "query_position": int(record["query_position"]),
                                "certified_l1_radius": float(radius),
                            },
                            self.paths.certification(
                                identifier,
                                str(record["method"]),
                                int(record["query_position"]),
                            ),
                        )
        return radii

    def analyze(self, *, allow_partial: bool = False, skip_certification: bool = False) -> pd.DataFrame:
        frame = self.aggregate(allow_partial=allow_partial)
        if frame.empty:
            raise RuntimeError("no query results available")
        frame = self._attach_manifoldness(frame)
        keys = ["case_id", "query_position"]
        success_counts = frame.groupby(keys)["success"].transform("sum")
        frame["shared_success_all"] = success_counts == len(METHODS)
        pgd_certcf = frame[frame["method"].isin(["anchor_pgd", "certcf"])].groupby(keys)["success"].transform("sum")
        frame.loc[frame["method"].isin(["anchor_pgd", "certcf"]), "shared_success_pgd_certcf"] = pgd_certcf == 2
        frame["certified_l1_radius"] = np.nan if skip_certification else self._certified_l1_radii(frame)
        empirical = self._empirical_robustness(frame)
        _atomic_parquet(empirical, self.paths.empirical)
        query_robustness = (
            empirical.groupby(
                ["case_id", "method", "query_position"], observed=True
            )
            .agg(
                target_rate=("target_rate", "mean"),
                all_preserved=("all_preserved", "all"),
            )
            .reset_index()
        )
        robustness_summary = (
            query_robustness.groupby(["case_id", "method"], observed=True)
            .agg(empirical_target_rate=("target_rate", "mean"), empirical_all_preserved=("all_preserved", "mean"))
            .reset_index()
        )
        shared_keys = frame.loc[frame["shared_success_all"], keys].drop_duplicates()
        shared_empirical = query_robustness.merge(shared_keys, on=keys, how="inner")
        shared_robustness_summary = (
            shared_empirical.groupby(["case_id", "method"], observed=True)
            .agg(
                shared_empirical_target_rate=("target_rate", "mean"),
                shared_empirical_all_preserved=("all_preserved", "mean"),
            )
            .reset_index()
        )
        summary = (
            frame.groupby(["case_id", "kind", "dataset", "depth", "width", "method"], dropna=False, observed=True)
            .agg(
                queries=("query_position", "size"),
                success_rate=("success", "mean"),
                timeout_rate=("timed_out", "mean"),
                feasibility_rate=("domain_feasible", "mean"),
                mean_l1=("l1_distance", "mean"),
                mean_l2=("l2_distance", "mean"),
                mean_l0=("l0_changed", "mean"),
                mean_mad_l1=("mad_l1_distance", "mean"),
                mean_redundancy=("redundancy", "mean"),
                mean_runtime_seconds=("runtime_seconds", "mean"),
                median_runtime_seconds=("runtime_seconds", "median"),
                mean_log10_lof=("log10_lof", "mean"),
                mean_isolation_forest_score=("isolation_forest_score", "mean"),
                mean_certified_l1_radius=("certified_l1_radius", "mean"),
            )
            .reset_index()
            .merge(robustness_summary, on=["case_id", "method"], how="left")
        )
        shared_summary = (
            frame.loc[frame["shared_success_all"]]
            .groupby(
                ["case_id", "method"],
                dropna=False,
                observed=True,
            )
            .agg(
                shared_success_queries=("query_position", "size"),
                shared_mean_l1=("l1_distance", "mean"),
                shared_mean_l2=("l2_distance", "mean"),
                shared_mean_l0=("l0_changed", "mean"),
                shared_mean_mad_l1=("mad_l1_distance", "mean"),
                shared_mean_redundancy=("redundancy", "mean"),
                shared_mean_log10_lof=("log10_lof", "mean"),
                shared_mean_isolation_forest_score=("isolation_forest_score", "mean"),
                shared_mean_certified_l1_radius=("certified_l1_radius", "mean"),
            )
            .reset_index()
        )
        summary = (
            summary.merge(shared_summary, on=["case_id", "method"], how="left")
            .merge(
                shared_robustness_summary,
                on=["case_id", "method"],
                how="left",
            )
        )
        # Paired PGD/CertCF L1 ratio, deliberately distinct from the paper's
        # nearest-neighbour relative proximity ratio.
        paired = frame[frame["method"].isin(["anchor_pgd", "certcf"]) & frame["success"].astype(bool)].pivot_table(
            index=keys, columns="method", values="l1_distance", aggfunc="first"
        ).dropna()
        if not paired.empty:
            ratios = (paired["certcf"] / paired["anchor_pgd"].replace(0.0, np.nan)).groupby(level=0).mean()
            summary["mean_certcf_to_pgd_l1_ratio"] = summary["case_id"].map(ratios)
        macro_columns = [
            column
            for column in summary.columns
            if column
            not in {
                "case_id",
                "kind",
                "dataset",
                "depth",
                "width",
                "method",
                "queries",
                "shared_success_queries",
            }
            and pd.api.types.is_numeric_dtype(summary[column])
        ]
        macro = summary.groupby("method", as_index=False)[macro_columns].mean()
        macro.insert(1, "n_datasets", summary["dataset"].nunique())
        _atomic_parquet(frame, self.paths.combined)
        _atomic_parquet(summary, self.paths.summary)
        _atomic_parquet(macro, self.paths.summary_macro)
        _log(f"[ANALYZE] Tabella salvata in {self.paths.summary}.")
        return summary

    def status(self) -> dict[str, Any]:
        output: dict[str, Any] = {"prepared": self.paths.manifest.exists(), "cases": {}}
        for case in self.cases:
            identifier = str(case["id"])
            expected = (
                len(self._geometry(identifier)["x_query"])
                if self.paths.manifest.exists()
                else 0
            )
            output["cases"][identifier] = {
                "build_complete": self._valid_build(case) if self.paths.manifest.exists() else False,
                **{
                    method: sum(self._query_valid(case, method, position) for position in range(expected))
                    if self.paths.manifest.exists() else 0
                    for method in METHODS
                },
            }
        return output

    def all(self, *, force: bool = False, skip_certification: bool = False) -> pd.DataFrame:
        self.prepare(force=force)
        self.build_all(force=force)
        self.benchmark(force=force)
        return self.analyze(skip_certification=skip_certification)


__all__ = [
    "DEFAULT_CONFIG",
    "LiRPARefinementAblationRunner",
    "METHODS",
    "anchor_ball_candidate",
    "anchor_pgd_candidate",
    "config_fingerprint",
    "load_config",
    "project_feasible_intersection",
    "project_l1_ball",
    "project_simplex",
    "validate_config",
]
