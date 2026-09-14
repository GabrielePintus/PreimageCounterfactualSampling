"""Shrinkage and LiRPA-backend ablations on all tabular benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from certcf import NearestOppositeClassClearanceStrategy
from counterfactuals.methods.certcf import CertCF
from counterfactuals.models.torch_model import TorchModelWrapper
from experiments.lirpa_refinement_ablation import (
    LiRPARefinementAblationRunner,
    _device,
    _prediction,
)
from experiments.network_complexity import PhaseResourceMonitor


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "source_config": "configs/experiments/lirpa_refinement_ablation_tabular.yaml",
        "cases": [
            "adult",
            "compas",
            "german_credit",
            "give_me_some_credit",
            "heloc",
            "lending_club",
            "wisconsin_breast_cancer",
        ],
    },
    "common": {
        "norm": 1,
        "distance_norm": 1,
        "classification_margin": 1.0e-4,
        "adaptive_eps_shrink_factor": 0.5,
        "adaptive_eps_max_shrinks": 8,
        "adaptive_eps_min": 1.0e-6,
        "adaptive_eps_center_tol": 1.0e-6,
        "adaptive_eps_binary_search_steps": 0,
        "epsilon_parallelism": 8,
        "build_parallelism": 2,
        "candidate_parallelism": 8,
        "candidate_parallel_backend": "process",
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
    "shrinkage": {
        "alphas": [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
        "variants": {"without_shrinkage": False, "with_shrinkage": True},
        "anchors_per_predicted_class": 500,
        "lirpa_method": "backward",
        "lirpa_batch_size": 512,
    },
    "backend": {
        "alpha": 0.20,
        "methods": {
            "crown": "backward",
            "optimized_crown": "crown-optimized",
            "alpha_crown": "alpha-crown",
        },
        "repetitions": 3,
        "anchors_per_predicted_class": 150,
        "queries_per_case": 200,
        "lirpa_batch_size": 512,
        "volume_samples_per_region": 1024,
        "volume_tolerance": 1.0e-8,
    },
    "pilot": {
        "cases": ["heloc"],
        "shrinkage_alphas": [0.20],
        "backend_methods": ["crown"],
        "backend_repetitions": [1],
        "backend_queries": 1,
        "volume_samples_per_region": 32,
    },
    "artifacts": {"output_dir": "results/appendix_d_tabular/atlas"},
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
    if not config["shrinkage"]["alphas"]:
        raise ValueError("shrinkage.alphas cannot be empty")
    if int(config["backend"]["repetitions"]) <= 0:
        raise ValueError("backend.repetitions must be positive")
    return config


def _atomic_json(payload: Any, path: Path) -> None:
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
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def shrinkage(self, case_id: str, variant: str, alpha: float) -> Path:
        token = f"alpha_{alpha:.2f}".replace(".", "p")
        return self.root / "shrinkage" / case_id / variant / f"{token}.parquet"

    def shrinkage_run(self, case_id: str, variant: str, alpha: float) -> Path:
        return self.shrinkage(case_id, variant, alpha).with_suffix(".json")

    def backend_prefix(self, case_id: str, method: str, repetition: int) -> Path:
        return self.root / "backend" / case_id / f"{method}_rep{repetition}"

    @property
    def shrinkage_regions(self) -> Path:
        return self.root / "shrinkage_regions.parquet"

    @property
    def shrinkage_summary(self) -> Path:
        return self.root / "shrinkage_summary_by_dataset.parquet"

    @property
    def shrinkage_macro(self) -> Path:
        return self.root / "shrinkage_summary_macro.parquet"

    @property
    def backend_runs(self) -> Path:
        return self.root / "backend_runs.parquet"

    @property
    def backend_regions(self) -> Path:
        return self.root / "backend_regions.parquet"

    @property
    def backend_volumes(self) -> Path:
        return self.root / "backend_volumes.parquet"

    @property
    def backend_queries(self) -> Path:
        return self.root / "backend_queries.parquet"

    @property
    def backend_summary(self) -> Path:
        return self.root / "backend_summary_by_dataset.parquet"

    @property
    def backend_macro(self) -> Path:
        return self.root / "backend_summary_macro.parquet"


def _sample_uniform_l1_ball(
    center: np.ndarray,
    radius: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = len(center)
    exponential = rng.exponential(scale=1.0, size=(n_samples, dimension + 1))
    magnitudes = exponential[:, :dimension] / exponential.sum(axis=1, keepdims=True)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_samples, dimension))
    return np.asarray(center)[None, :] + float(radius) * signs * magnitudes


def _log_l1_ball_volume(radius: float, dimension: int) -> float:
    if radius <= 0.0:
        return -math.inf
    return dimension * math.log(2.0) + dimension * math.log(radius) - math.lgamma(
        dimension + 1
    )


class TabularAtlasAblationRunner:
    """Run radius-shrinkage and LiRPA-backend experiments per dataset."""

    def __init__(self, config: dict[str, Any], config_path: str | Path | None = None):
        self.config = copy.deepcopy(config)
        self.config_path = None if config_path is None else Path(config_path)
        self.repository_root = Path(__file__).resolve().parents[2]
        source_path = Path(config["experiment"]["source_config"])
        if not source_path.is_absolute():
            source_path = self.repository_root / source_path
        self.source = LiRPARefinementAblationRunner.from_yaml(source_path)
        output = Path(config["artifacts"]["output_dir"])
        if not output.is_absolute():
            output = self.repository_root / output
        self.paths = Paths(output.resolve())
        payload = copy.deepcopy(config)
        payload.pop("artifacts", None)
        payload["source_config_fingerprint"] = self.source.fingerprint
        self.fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _metadata_matches(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return metadata.get("config_fingerprint") == self.fingerprint

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TabularAtlasAblationRunner":
        return cls(load_config(path), config_path=path)

    def resolve_cases(self, identifiers: Sequence[str] | None = None) -> list[dict[str, Any]]:
        requested = (
            list(self.config["experiment"]["cases"])
            if identifiers is None
            else [str(value) for value in identifiers]
        )
        cases = self.source.resolve_cases(requested)
        if any(case["kind"] != "tabular" for case in cases):
            raise ValueError("the tabular atlas ablation accepts tabular cases only")
        return cases

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        source = self.source.prepare(force=force)
        manifest = {
            "status": "complete",
            "config_fingerprint": self.fingerprint,
            "source_config_fingerprint": self.source.fingerprint,
            "cases": [str(case["id"]) for case in self.resolve_cases()],
            "source_manifest_status": source.get("status"),
        }
        _atomic_json(manifest, self.paths.manifest)
        return manifest

    @staticmethod
    def _select_anchors(
        geometry: dict[str, np.ndarray],
        per_class: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        parts: list[np.ndarray] = []
        labels = geometry["y_anchor_pred"]
        for label in (0, 1):
            candidates = np.flatnonzero(labels == label)
            take = min(int(per_class), len(candidates))
            parts.append(rng.choice(candidates, size=take, replace=False))
        indices = np.concatenate(parts)
        rng.shuffle(indices)
        return geometry["x_anchor"][indices], labels[indices]

    def _make_method(
        self,
        case: dict[str, Any],
        model,
        geometry: dict[str, np.ndarray],
        *,
        alpha: float,
        adaptive_eps: bool,
        lirpa_method: str,
        lirpa_batch_size: int,
    ) -> CertCF:
        common = self.config["common"]
        strategy = NearestOppositeClassClearanceStrategy(alpha=float(alpha), chunk_size=128)
        strategy.set_reference(geometry["x_support"], geometry["y_support_pred"])
        fixed_dims = self.source._fixed_dims(case)
        dataset_name = str(case["dataset"])
        return CertCF(
            model=TorchModelWrapper(model=model, device=_device(self.source.config)),
            norm=int(common["norm"]),
            distance_norm=int(common["distance_norm"]),
            lirpa_method=str(lirpa_method),
            eps_strategy=strategy,
            batch_size=int(lirpa_batch_size),
            epsilon_parallelism=int(common["epsilon_parallelism"]),
            build_parallelism=int(common["build_parallelism"]),
            ohe_slices=list(self.source._ohe_slices(case)) or None,
            default_query_method="nearest_anchor",
            query_k_candidates=5,
            solver_maxiter=int(common["solver_maxiter"]),
            candidate_parallelism=int(common["candidate_parallelism"]),
            candidate_parallel_backend=str(common["candidate_parallel_backend"]),
            cvxpy_solvers=list(common["cvxpy_solvers"]),
            cvxpy_solver_options=copy.deepcopy(common["cvxpy_solver_options"]),
            cvxpy_accept_statuses=copy.deepcopy(common["cvxpy_accept_statuses"]),
            classification_margin=float(common["classification_margin"]),
            adaptive_eps=bool(adaptive_eps),
            adaptive_eps_shrink_factor=float(common["adaptive_eps_shrink_factor"]),
            adaptive_eps_max_shrinks=int(common["adaptive_eps_max_shrinks"]),
            adaptive_eps_min=float(common["adaptive_eps_min"]),
            adaptive_eps_center_tol=float(common["adaptive_eps_center_tol"]),
            adaptive_eps_binary_search_steps=int(common["adaptive_eps_binary_search_steps"]),
            ohe_decode_mode="beam_then_exact",
            decode_beam_width=8,
            decode_beam_branch_top_k=3,
            decode_beam_max_solver_calls=32,
            sparsity_penalty="none",
            fixed_dims=fixed_dims,
            immutable_features=self.source.config["constraints"]["immutable_features"].get(
                dataset_name, []
            ),
            k_per_class=None,
            subsample_method="random",
            random_seed=int(self.config["experiment"]["seed"]),
        )

    def _region_frame(
        self,
        method: CertCF,
        *,
        dataset: str,
        family: str,
        variant: str,
        alpha: float,
        repetition: int | None = None,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        margin = float(self.config["common"]["classification_margin"])
        for label in method.atlas.class_labels:
            bounds = method.atlas.bounds[int(label)]
            centers = np.asarray(bounds["X"], dtype=np.float64)
            initial = np.asarray(bounds.get("eps_initial", bounds["eps"]), dtype=float)
            final = np.asarray(bounds["eps"], dtype=float)
            stored_certified = bounds.get("adaptive_eps_center_certified")
            stored_slack = bounds.get("adaptive_eps_center_slack")
            if stored_slack is None:
                slack_values = []
                for index, center in enumerate(centers):
                    matrix = np.asarray(bounds["lA"][index], dtype=float).reshape(
                        -1, len(center)
                    )
                    bias = np.asarray(bounds["lbias"][index], dtype=float).reshape(-1)
                    slack_values.append(
                        math.inf
                        if not len(bias)
                        else float(np.min(matrix @ center + bias - margin))
                    )
                slack = np.asarray(slack_values)
            else:
                slack = np.asarray(stored_slack, dtype=float)
            certified = (
                np.asarray(stored_certified, dtype=bool)
                if stored_certified is not None
                else slack >= -float(self.config["common"]["adaptive_eps_center_tol"])
            )
            shrinks = np.asarray(
                bounds.get("adaptive_eps_n_shrinks", np.zeros(len(final))), dtype=float
            )
            # A failed atlas build may stop while another class still contains
            # uncertified raw bounds. Keep only the regions that may actually
            # enter a certified atlas.
            retained_indices = np.flatnonzero(certified)
            centers = centers[retained_indices]
            initial = initial[retained_indices]
            final = final[retained_indices]
            slack = slack[retained_indices]
            shrinks = shrinks[retained_indices]
            certified = certified[retained_indices]
            rows.append(
                pd.DataFrame(
                    {
                        "config_fingerprint": self.fingerprint,
                        "family": family,
                        "dataset": dataset,
                        "variant": variant,
                        "backend": variant if family == "backend" else None,
                        "alpha": float(alpha),
                        "repetition": repetition,
                        "class_label": int(label),
                        "polytope_index": retained_indices,
                        "eps_initial": initial,
                        "eps_final": final,
                        "eps_ratio": np.divide(
                            final,
                            initial,
                            out=np.ones_like(final),
                            where=initial > 0,
                        ),
                        "center_certified": certified,
                        "n_shrinks": shrinks,
                        "center_slack": slack,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def run_shrinkage(
        self,
        cases: Iterable[dict[str, Any]] | None = None,
        *,
        alphas: Sequence[float] | None = None,
        variants: Sequence[str] | None = None,
        force: bool = False,
    ) -> list[Path]:
        self.prepare(force=False)
        selected_cases = self.resolve_cases() if cases is None else list(cases)
        cfg = self.config["shrinkage"]
        selected_alphas = (
            [float(value) for value in cfg["alphas"]]
            if alphas is None
            else [float(value) for value in alphas]
        )
        selected_variants = list(cfg["variants"]) if variants is None else list(variants)
        unknown = sorted(set(selected_variants) - set(cfg["variants"]))
        if unknown:
            raise ValueError(f"unknown shrinkage variants: {unknown}")
        device = _device(self.source.config)
        outputs: list[Path] = []
        for case in selected_cases:
            identifier = str(case["id"])
            geometry = self.source._geometry(identifier)
            _, _, _, _, model, _ = self.source._load_case(case, device)
            anchors, labels = self._select_anchors(
                geometry,
                int(cfg["anchors_per_predicted_class"]),
                int(self.config["experiment"]["seed"]),
            )
            for variant in selected_variants:
                adaptive = bool(cfg["variants"][variant])
                for alpha in selected_alphas:
                    output = self.paths.shrinkage(identifier, variant, alpha)
                    metadata_path = self.paths.shrinkage_run(identifier, variant, alpha)
                    if output.exists() and self._metadata_matches(metadata_path) and not force:
                        outputs.append(output)
                        continue
                    method = self._make_method(
                        case,
                        model,
                        geometry,
                        alpha=alpha,
                        adaptive_eps=adaptive,
                        lirpa_method=str(cfg["lirpa_method"]),
                        lirpa_batch_size=int(cfg["lirpa_batch_size"]),
                    )
                    requested_initial = np.asarray(
                        method.eps_strategy.compute_eps(anchors, labels, norm=1),
                        dtype=float,
                    )
                    started = time.perf_counter()
                    build_error = None
                    try:
                        method.fit(anchors, labels)
                    except RuntimeError as exc:
                        if not str(exc).startswith(
                            "No certified atlas anchors remain for class"
                        ):
                            raise
                        build_error = str(exc)
                    elapsed = time.perf_counter() - started
                    frame = self._region_frame(
                        method,
                        dataset=identifier,
                        family="shrinkage",
                        variant=variant,
                        alpha=alpha,
                    )
                    _atomic_parquet(frame, output)
                    certified_by_class = {
                        str(label): int(np.sum(frame["class_label"] == label))
                        for label in (0, 1)
                    }
                    _atomic_json(
                        {
                            "status": "complete",
                            "config_fingerprint": self.fingerprint,
                            "dataset": identifier,
                            "variant": variant,
                            "alpha": alpha,
                            "attempted_anchors": len(anchors),
                            "certified_regions": len(frame),
                            "certified_regions_by_class": certified_by_class,
                            "certification_yield": len(frame) / len(anchors),
                            "complete_binary_atlas": all(
                                count > 0 for count in certified_by_class.values()
                            ),
                            "build_error": build_error,
                            "requested_eps_initial_median": float(
                                np.median(requested_initial)
                            ),
                            "requested_eps_initial_q25": float(
                                np.quantile(requested_initial, 0.25)
                            ),
                            "requested_eps_initial_q75": float(
                                np.quantile(requested_initial, 0.75)
                            ),
                            "build_time_s": elapsed,
                        },
                        metadata_path,
                    )
                    if build_error is not None:
                        print(
                            f"[{identifier} {variant} alpha={alpha:.2f}] "
                            f"recorded incomplete atlas: {build_error}"
                        )
                    outputs.append(output)
        return outputs

    def _volume_frame(
        self,
        method: CertCF,
        *,
        dataset: str,
        variant: str,
        repetition: int,
        samples_per_region: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(self.config["experiment"]["seed"]), repetition, 911]
            )
        )
        tolerance = float(self.config["backend"]["volume_tolerance"])
        rows: list[dict[str, Any]] = []
        for label in method.atlas.class_labels:
            bounds = method.atlas.bounds[int(label)]
            centers = np.asarray(bounds["X"], dtype=float)
            radii = np.asarray(bounds["eps"], dtype=float)
            matrices = np.asarray(bounds["lA"], dtype=float)
            biases = np.asarray(bounds["lbias"], dtype=float)
            dimension = centers.shape[1]
            for index, center in enumerate(centers):
                samples = _sample_uniform_l1_ball(
                    center, radii[index], samples_per_region, rng
                )
                matrix = matrices[index].reshape(-1, dimension)
                bias = biases[index].reshape(-1)
                inside = (
                    np.ones(samples_per_region, dtype=bool)
                    if not len(bias)
                    else np.all(
                        samples @ matrix.T
                        + bias[None, :]
                        - float(self.config["common"]["classification_margin"])
                        >= -tolerance,
                        axis=1,
                    )
                )
                retained = float(inside.mean())
                log_ball = _log_l1_ball_volume(float(radii[index]), dimension)
                rows.append(
                    {
                        "config_fingerprint": self.fingerprint,
                        "dataset": dataset,
                        "backend": variant,
                        "repetition": repetition,
                        "class_label": int(label),
                        "polytope_index": index,
                        "retained_fraction": retained,
                        "zero_hit": retained == 0.0,
                        "log_l1_ball_volume": log_ball,
                        "log_estimated_polytope_volume": log_ball
                        + math.log(max(retained, 0.5 / samples_per_region)),
                    }
                )
        return pd.DataFrame(rows)

    def run_backend(
        self,
        cases: Iterable[dict[str, Any]] | None = None,
        *,
        methods: Sequence[str] | None = None,
        repetitions: Sequence[int] | None = None,
        query_limit: int | None = None,
        volume_samples: int | None = None,
        force: bool = False,
    ) -> list[Path]:
        self.prepare(force=False)
        selected_cases = self.resolve_cases() if cases is None else list(cases)
        cfg = self.config["backend"]
        selected_methods = list(cfg["methods"]) if methods is None else list(methods)
        unknown = sorted(set(selected_methods) - set(cfg["methods"]))
        if unknown:
            raise ValueError(f"unknown backend methods: {unknown}")
        selected_repetitions = (
            list(range(1, int(cfg["repetitions"]) + 1))
            if repetitions is None
            else [int(value) for value in repetitions]
        )
        configured_queries = int(cfg["queries_per_case"])
        configured_volume = int(cfg["volume_samples_per_region"])
        device = _device(self.source.config)
        outputs: list[Path] = []
        for case in selected_cases:
            identifier = str(case["id"])
            geometry = self.source._geometry(identifier)
            _, _, _, _, model, _ = self.source._load_case(case, device)
            anchors, labels = self._select_anchors(
                geometry,
                int(cfg["anchors_per_predicted_class"]),
                int(self.config["experiment"]["seed"]),
            )
            n_queries = min(
                configured_queries if query_limit is None else int(query_limit),
                len(geometry["x_query"]),
            )
            n_volume = configured_volume if volume_samples is None else int(volume_samples)
            for method_name in selected_methods:
                lirpa_method = str(cfg["methods"][method_name])
                for repetition in selected_repetitions:
                    prefix = self.paths.backend_prefix(identifier, method_name, repetition)
                    run_path = prefix.with_suffix(".json")
                    region_path = prefix.with_name(prefix.name + "_regions.parquet")
                    volume_path = prefix.with_name(prefix.name + "_volumes.parquet")
                    query_path = prefix.with_name(prefix.name + "_queries.parquet")
                    if (
                        all(path.exists() for path in (run_path, region_path, volume_path, query_path))
                        and self._metadata_matches(run_path)
                        and not force
                    ):
                        outputs.append(run_path)
                        continue
                    method = self._make_method(
                        case,
                        model,
                        geometry,
                        alpha=float(cfg["alpha"]),
                        adaptive_eps=True,
                        lirpa_method=lirpa_method,
                        lirpa_batch_size=int(cfg["lirpa_batch_size"]),
                    )
                    with PhaseResourceMonitor("build", device, 0.05) as monitor:
                        method.fit(anchors, labels)
                    regions = self._region_frame(
                        method,
                        dataset=identifier,
                        family="backend",
                        variant=method_name,
                        alpha=float(cfg["alpha"]),
                        repetition=repetition,
                    )
                    volumes = self._volume_frame(
                        method,
                        dataset=identifier,
                        variant=method_name,
                        repetition=repetition,
                        samples_per_region=n_volume,
                    )
                    query_rows: list[dict[str, Any]] = []
                    for position in range(n_queries):
                        started = time.perf_counter()
                        result = method.generate(
                            geometry["x_query"][position],
                            target_class=int(geometry["target"][position]),
                        )
                        elapsed = time.perf_counter() - started
                        point = None if result.x_cf is None else np.asarray(result.x_cf)
                        predicted = (
                            -1
                            if point is None
                            else int(_prediction(model, point[None, :], device)[0])
                        )
                        success = bool(result.success and predicted == geometry["target"][position])
                        difference = (
                            np.full(geometry["x_query"].shape[1], np.nan)
                            if point is None
                            else np.abs(point - geometry["x_query"][position])
                        )
                        query_rows.append(
                            {
                                "config_fingerprint": self.fingerprint,
                                "dataset": identifier,
                                "backend": method_name,
                                "repetition": repetition,
                                "query_position": position,
                                "query_index": int(geometry["query_indices"][position]),
                                "success": success,
                                "predicted_class": predicted,
                                "target_class": int(geometry["target"][position]),
                                "l1": float(np.nansum(difference)),
                                "l2": float(np.linalg.norm(difference, 2)),
                                "l0": int(np.count_nonzero(difference > 1.0e-6)),
                                "runtime_s": elapsed,
                            }
                        )
                    queries = pd.DataFrame(query_rows)
                    run = {
                        "status": "complete",
                        "config_fingerprint": self.fingerprint,
                        "dataset": identifier,
                        "backend": method_name,
                        "lirpa_method": lirpa_method,
                        "repetition": repetition,
                        "anchors": len(regions),
                        "queries": n_queries,
                        "volume_samples_per_region": n_volume,
                        **monitor.metrics,
                    }
                    _atomic_parquet(regions, region_path)
                    _atomic_parquet(volumes, volume_path)
                    _atomic_parquet(queries, query_path)
                    _atomic_json(run, run_path)
                    if method.atlas is not None:
                        method.atlas.close_candidate_process_pool()
                    outputs.append(run_path)
        return outputs

    def aggregate_shrinkage(self, *, allow_partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        run_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        cfg = self.config["shrinkage"]
        for case in self.resolve_cases():
            identifier = str(case["id"])
            for variant in cfg["variants"]:
                for alpha in cfg["alphas"]:
                    path = self.paths.shrinkage(identifier, variant, float(alpha))
                    if not path.exists():
                        missing.append(str(path))
                    else:
                        frames.append(pd.read_parquet(path))
                        metadata_path = self.paths.shrinkage_run(
                            identifier, variant, float(alpha)
                        )
                        if metadata_path.exists():
                            run_rows.append(
                                json.loads(metadata_path.read_text(encoding="utf-8"))
                            )
        if missing and not allow_partial:
            raise RuntimeError(f"missing {len(missing)} shrinkage runs; first: {missing[0]}")
        regions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if regions.empty:
            raise RuntimeError("no shrinkage results available")
        region_summary = (
            regions.groupby(["dataset", "variant", "alpha"], as_index=False)
            .agg(
                regions=("polytope_index", "size"),
                certified_classes=("class_label", "nunique"),
                center_certification_rate=("center_certified", "mean"),
                eps_initial_median=("eps_initial", "median"),
                eps_final_median=("eps_final", "median"),
                eps_retention_median=("eps_ratio", "median"),
                shrink_fraction=("n_shrinks", lambda values: float(np.mean(values > 0))),
                mean_shrinks=("n_shrinks", "mean"),
            )
        )
        runs = pd.DataFrame(run_rows)
        required_run_columns = {
            "dataset",
            "variant",
            "alpha",
            "attempted_anchors",
            "certified_regions",
            "certification_yield",
            "requested_eps_initial_median",
            "requested_eps_initial_q25",
            "requested_eps_initial_q75",
            "build_time_s",
        }
        if not required_run_columns.issubset(runs.columns):
            missing_columns = sorted(required_run_columns - set(runs.columns))
            raise RuntimeError(
                "shrinkage metadata predates the tabular protocol; rerun the affected "
                f"runs with --force (missing columns: {missing_columns})"
            )
        summary = runs[list(required_run_columns)].merge(
            region_summary,
            on=["dataset", "variant", "alpha"],
            how="left",
            validate="one_to_one",
        )
        summary["complete_binary_atlas"] = summary["certified_classes"].eq(2)
        metrics = [
            "certification_yield",
            "complete_binary_atlas",
            "center_certification_rate",
            "requested_eps_initial_median",
            "eps_initial_median",
            "eps_final_median",
            "eps_retention_median",
            "shrink_fraction",
            "mean_shrinks",
        ]
        macro = summary.groupby(["variant", "alpha"], as_index=False)[metrics].mean()
        macro.insert(2, "n_datasets", summary["dataset"].nunique())
        _atomic_parquet(regions, self.paths.shrinkage_regions)
        _atomic_parquet(summary, self.paths.shrinkage_summary)
        _atomic_parquet(macro, self.paths.shrinkage_macro)
        return regions, summary, macro

    def aggregate_backend(self, *, allow_partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        run_rows: list[dict[str, Any]] = []
        region_frames: list[pd.DataFrame] = []
        volume_frames: list[pd.DataFrame] = []
        query_frames: list[pd.DataFrame] = []
        missing: list[str] = []
        cfg = self.config["backend"]
        for case in self.resolve_cases():
            identifier = str(case["id"])
            for method in cfg["methods"]:
                for repetition in range(1, int(cfg["repetitions"]) + 1):
                    prefix = self.paths.backend_prefix(identifier, method, repetition)
                    paths = {
                        "run": prefix.with_suffix(".json"),
                        "regions": prefix.with_name(prefix.name + "_regions.parquet"),
                        "volumes": prefix.with_name(prefix.name + "_volumes.parquet"),
                        "queries": prefix.with_name(prefix.name + "_queries.parquet"),
                    }
                    if not all(path.exists() for path in paths.values()):
                        missing.append(str(prefix))
                        continue
                    run_rows.append(json.loads(paths["run"].read_text(encoding="utf-8")))
                    region_frames.append(pd.read_parquet(paths["regions"]))
                    volume_frames.append(pd.read_parquet(paths["volumes"]))
                    query_frames.append(pd.read_parquet(paths["queries"]))
        if missing and not allow_partial:
            raise RuntimeError(f"missing {len(missing)} backend runs; first: {missing[0]}")
        runs = pd.DataFrame(run_rows)
        regions = pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame()
        volumes = pd.concat(volume_frames, ignore_index=True) if volume_frames else pd.DataFrame()
        queries = pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame()
        if runs.empty:
            raise RuntimeError("no backend results available")
        if "backend" not in regions.columns and "variant" in regions.columns:
            regions["backend"] = regions["variant"]
        run_metrics = runs[["dataset", "backend", "repetition", "build_wall_time_s"]]
        region_summary = regions.groupby(["dataset", "backend", "repetition"], as_index=False).agg(
            center_certification_rate=("center_certified", "mean"),
            eps_final_median=("eps_final", "median"),
            eps_retention_median=("eps_ratio", "median"),
            mean_shrinks=("n_shrinks", "mean"),
        )
        volume_summary = volumes.groupby(["dataset", "backend", "repetition"], as_index=False).agg(
            retained_fraction_mean=("retained_fraction", "mean"),
            retained_fraction_median=("retained_fraction", "median"),
            median_log_volume=("log_estimated_polytope_volume", "median"),
        )
        query_summary = queries.groupby(["dataset", "backend", "repetition"], as_index=False).agg(
            validity=("success", "mean"),
            mean_l1=("l1", "mean"),
            mean_l2=("l2", "mean"),
            mean_l0=("l0", "mean"),
            mean_query_s=("runtime_s", "mean"),
        )
        repetitions = run_metrics.merge(region_summary).merge(volume_summary).merge(query_summary)
        summary = repetitions.groupby(["dataset", "backend"], as_index=False).agg(
            build_mean_s=("build_wall_time_s", "mean"),
            build_std_s=("build_wall_time_s", "std"),
            center_certification_rate=("center_certification_rate", "mean"),
            eps_final_median=("eps_final_median", "mean"),
            eps_retention_median=("eps_retention_median", "mean"),
            mean_shrinks=("mean_shrinks", "mean"),
            retained_fraction_mean=("retained_fraction_mean", "mean"),
            retained_fraction_median=("retained_fraction_median", "mean"),
            median_log_volume=("median_log_volume", "mean"),
            validity=("validity", "mean"),
            mean_l1=("mean_l1", "mean"),
            mean_l2=("mean_l2", "mean"),
            mean_l0=("mean_l0", "mean"),
            mean_query_s=("mean_query_s", "mean"),
        )
        macro_metrics = [
            column
            for column in summary.columns
            if column not in {"dataset", "backend", "median_log_volume"}
        ]
        macro = summary.groupby("backend", as_index=False)[macro_metrics].mean()
        macro.insert(1, "n_datasets", summary["dataset"].nunique())
        _atomic_parquet(runs, self.paths.backend_runs)
        _atomic_parquet(regions, self.paths.backend_regions)
        _atomic_parquet(volumes, self.paths.backend_volumes)
        _atomic_parquet(queries, self.paths.backend_queries)
        _atomic_parquet(summary, self.paths.backend_summary)
        _atomic_parquet(macro, self.paths.backend_macro)
        return summary, macro

    def pilot(self, *, force: bool = False) -> dict[str, Any]:
        pilot_config = copy.deepcopy(self.config)
        pilot_config["artifacts"]["output_dir"] = str(self.paths.root / "pilot")
        pilot_runner = TabularAtlasAblationRunner(pilot_config)
        cfg = pilot_runner.config["pilot"]
        cases = pilot_runner.resolve_cases(cfg["cases"])
        pilot_runner.prepare(force=False)
        shrinkage = pilot_runner.run_shrinkage(
            cases,
            alphas=cfg["shrinkage_alphas"],
            force=force,
        )
        backend = pilot_runner.run_backend(
            cases,
            methods=cfg["backend_methods"],
            repetitions=cfg["backend_repetitions"],
            query_limit=int(cfg["backend_queries"]),
            volume_samples=int(cfg["volume_samples_per_region"]),
            force=force,
        )
        return {"shrinkage_runs": len(shrinkage), "backend_runs": len(backend)}

    def status(self) -> dict[str, Any]:
        shrinkage_cfg = self.config["shrinkage"]
        backend_cfg = self.config["backend"]
        return {
            "shrinkage": {
                str(case["id"]): sum(
                    self.paths.shrinkage(str(case["id"]), variant, float(alpha)).exists()
                    for variant in shrinkage_cfg["variants"]
                    for alpha in shrinkage_cfg["alphas"]
                )
                for case in self.resolve_cases()
            },
            "shrinkage_expected_per_dataset": len(shrinkage_cfg["variants"])
            * len(shrinkage_cfg["alphas"]),
            "backend": {
                str(case["id"]): sum(
                    self.paths.backend_prefix(str(case["id"]), method, repetition)
                    .with_suffix(".json")
                    .exists()
                    for method in backend_cfg["methods"]
                    for repetition in range(1, int(backend_cfg["repetitions"]) + 1)
                )
                for case in self.resolve_cases()
            },
            "backend_expected_per_dataset": len(backend_cfg["methods"])
            * int(backend_cfg["repetitions"]),
        }


__all__ = ["DEFAULT_CONFIG", "TabularAtlasAblationRunner", "load_config"]
