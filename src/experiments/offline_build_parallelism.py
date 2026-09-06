"""Controlled serial/parallel benchmarks for CertCF atlas construction."""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.cifar_resnet_scaling import (
    CifarResNetScalingRunner,
    _device as cifar_device,
)
from experiments.lirpa_refinement_ablation import (
    LiRPARefinementAblationRunner,
    _device as ablation_device,
)
from experiments.network_complexity import (
    NetworkComplexityRunner,
    PhaseResourceMonitor,
    _normalize_device,
    architecture_id,
)
from certcf.eps_strategies import NearestOppositeClassClearanceStrategy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOUND_KEYS = (
    "X",
    "eps",
    "eps_initial",
    "adaptive_eps_n_shrinks",
    "adaptive_eps_n_binary_steps",
    "adaptive_eps_center_slack",
    "adaptive_eps_center_certified",
    "lA",
    "lbias",
    "uA",
    "ubias",
)
ATLAS_DECISION_KEYS = (
    "X",
    "eps",
    "eps_initial",
    "adaptive_eps_n_shrinks",
    "adaptive_eps_n_binary_steps",
    "adaptive_eps_center_certified",
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if not config.get("cases"):
        raise ValueError("cases must not be empty")
    if not config.get("variants"):
        raise ValueError("variants must not be empty")
    allowed_kinds = {"lirpa_ablation", "network_complexity", "cifar_scaling"}
    for case_id, case in config["cases"].items():
        if case.get("kind") not in allowed_kinds:
            raise ValueError(f"unknown kind for case {case_id!r}: {case.get('kind')!r}")
    for variant_id, variant in config["variants"].items():
        for key in ("epsilon_parallelism", "build_parallelism"):
            if int(variant.get(key, 0)) <= 0:
                raise ValueError(f"{variant_id}.{key} must be positive")
        if (
            variant.get("lirpa_batch_size") is not None
            and int(variant["lirpa_batch_size"]) <= 0
        ):
            raise ValueError(f"{variant_id}.lirpa_batch_size must be positive")
        if int(variant.get("cnn_radius_batch_size", 1)) <= 0:
            raise ValueError(f"{variant_id}.cnn_radius_batch_size must be positive")
        if float(variant.get("cnn_radius_batch_max_relative_inflation", 0.0)) < 0.0:
            raise ValueError(
                f"{variant_id}.cnn_radius_batch_max_relative_inflation must be non-negative"
            )
        unknown_cases = set(variant.get("cases", ())).difference(config["cases"])
        if unknown_cases:
            raise ValueError(f"{variant_id}.cases contains unknown cases: {unknown_cases}")
    comparison = config.get("comparison", {})
    if float(comparison.get("absolute_tolerance", -1.0)) < 0.0:
        raise ValueError("comparison.absolute_tolerance must be non-negative")
    if float(comparison.get("relative_tolerance", -1.0)) < 0.0:
        raise ValueError("comparison.relative_tolerance must be non-negative")
    if int(comparison.get("chunk_elements", 0)) <= 0:
        raise ValueError("comparison.chunk_elements must be positive")


def config_fingerprint(config: Mapping[str, Any]) -> str:
    protocol = deepcopy(dict(config))
    protocol.pop("artifacts", None)
    # The FCNN grid is an independent follow-up protocol and must not
    # invalidate the earlier per-case serial/parallel artifacts.
    protocol.pop("fcnn_grid", None)
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mapping_fingerprint(protocol: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        deepcopy(dict(protocol)), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


@dataclass(frozen=True)
class OfflineBuildPaths:
    root: Path

    def variant_dir(self, case_id: str, variant_id: str) -> Path:
        return self.root / case_id / variant_id

    def metadata(self, case_id: str, variant_id: str) -> Path:
        return self.variant_dir(case_id, variant_id) / "build.json"

    def generated_reference(self, case_id: str) -> Path:
        return self.root / case_id / "serial" / "atlas"

    @property
    def epsilon_sweep(self) -> Path:
        return self.root / "epsilon_parallelism_sweep.parquet"

    @property
    def batch_size_sweep(self) -> Path:
        return self.root / "lirpa_batch_size_sweep.parquet"

    @property
    def batch_size_summary(self) -> Path:
        return self.root / "lirpa_batch_size_summary.parquet"

    @property
    def fcnn_grid_sweep(self) -> Path:
        return self.root / "fcnn_grid_optimized_builds.parquet"

    @property
    def fcnn_grid_comparison(self) -> Path:
        return self.root / "fcnn_grid_serial_optimized.parquet"

    @property
    def fcnn_grid_models(self) -> Path:
        return self.root / "fcnn_grid_build_models.json"

    @property
    def summary(self) -> Path:
        return self.root / "offline_build_summary.parquet"


def _atlas_decision_fingerprint(method) -> str:
    """Hash the anchor/radius decisions that batching must preserve."""
    atlas = method.atlas
    if atlas is None or atlas.bounds is None:
        raise RuntimeError("CertCF did not produce an atlas")
    digest = hashlib.sha256()
    for label in sorted(atlas.bounds):
        digest.update(f"class:{int(label)}".encode())
        values = atlas.bounds[label]
        for key in ATLAS_DECISION_KEYS:
            if key not in values:
                continue
            array = np.ascontiguousarray(np.asarray(values[key]))
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _fit_interaction_model(frame: pd.DataFrame, response: str) -> dict[str, Any]:
    """Fit and leave-one-architecture-out evaluate 1 + D + (D-1)W."""
    depth = frame["depth"].to_numpy(dtype=np.float64)
    width = frame["width"].to_numpy(dtype=np.float64)
    target = frame[response].to_numpy(dtype=np.float64)
    design = np.column_stack(
        [np.ones(len(frame), dtype=np.float64), depth, (depth - 1.0) * width]
    )
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    loo_prediction = np.empty(len(frame), dtype=np.float64)
    for held_out in range(len(frame)):
        keep = np.arange(len(frame)) != held_out
        fold_coefficients = np.linalg.lstsq(
            design[keep], target[keep], rcond=None
        )[0]
        loo_prediction[held_out] = float(design[held_out] @ fold_coefficients)
    residual = target - loo_prediction
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    predictive_r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    return {
        "coefficients": {
            "intercept": float(coefficients[0]),
            "depth": float(coefficients[1]),
            "depth_minus_one_times_width": float(coefficients[2]),
        },
        "loo_prediction": loo_prediction,
        "loo_mae_s": float(np.mean(np.abs(residual))),
        "loo_rmse_s": float(np.sqrt(np.mean(residual**2))),
        "loo_predictive_r2": predictive_r2,
    }


def _compare_arrays(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
    chunk_elements: int,
) -> dict[str, Any]:
    current = np.asarray(current)
    reference = np.asarray(reference)
    if current.shape != reference.shape:
        return {
            "shape_equal": False,
            "dtype_equal": current.dtype == reference.dtype,
            "exact": False,
            "allclose": False,
            "max_absolute_difference": float("inf"),
        }

    current_flat = current.reshape(-1)
    reference_flat = reference.reshape(-1)
    exact = True
    allclose = True
    maximum = 0.0
    for start in range(0, current_flat.size, chunk_elements):
        stop = min(start + chunk_elements, current_flat.size)
        left = current_flat[start:stop]
        right = reference_flat[start:stop]
        exact = exact and bool(np.array_equal(left, right, equal_nan=True))
        if np.issubdtype(current.dtype, np.number) and np.issubdtype(reference.dtype, np.number):
            allclose = allclose and bool(
                np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
            )
            if left.size:
                difference = np.abs(
                    left.astype(np.float64, copy=False)
                    - right.astype(np.float64, copy=False)
                )
                finite = difference[np.isfinite(difference)]
                if finite.size:
                    maximum = max(maximum, float(np.max(finite)))
                if np.any(np.isinf(difference)):
                    maximum = float("inf")
        else:
            allclose = allclose and bool(np.array_equal(left, right, equal_nan=True))
            maximum = 0.0 if allclose else float("inf")
    return {
        "shape_equal": True,
        "dtype_equal": current.dtype == reference.dtype,
        "exact": exact,
        "allclose": allclose,
        "max_absolute_difference": maximum,
    }


def compare_bounds_to_saved(
    bounds: Mapping[int, Mapping[str, np.ndarray]],
    reference_dir: str | Path,
    *,
    atol: float,
    rtol: float,
    chunk_elements: int,
) -> dict[str, Any]:
    reference_dir = Path(reference_dir)
    manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    expected_labels = {int(label) for label in manifest["class_labels"]}
    current_labels = {int(label) for label in bounds}
    per_array: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for label in sorted(expected_labels | current_labels):
        if label not in expected_labels or label not in current_labels:
            missing.append(f"class_{label}")
            continue
        reference_file = reference_dir / manifest["files"][str(label)]
        with np.load(reference_file, allow_pickle=False) as reference:
            current_class = bounds[label]
            for key in BOUND_KEYS:
                identity = f"class_{label}.{key}"
                current_has_key = key in current_class
                reference_has_key = key in reference.files
                if not current_has_key and not reference_has_key:
                    continue
                if current_has_key != reference_has_key:
                    missing.append(identity)
                    continue
                per_array[identity] = _compare_arrays(
                    current_class[key],
                    reference[key],
                    atol=atol,
                    rtol=rtol,
                    chunk_elements=chunk_elements,
                )
    compared = list(per_array.values())
    return {
        "reference_atlas": str(reference_dir.resolve()),
        "labels_equal": expected_labels == current_labels,
        "missing": missing,
        "arrays_compared": len(compared),
        "all_exact": bool(compared) and not missing and all(v["exact"] for v in compared),
        "all_close": bool(compared) and not missing and all(v["allclose"] for v in compared),
        "maximum_absolute_difference": max(
            (float(value["max_absolute_difference"]) for value in compared),
            default=float("inf"),
        ),
        "per_array": per_array,
    }


class OfflineBuildParallelismRunner:
    """Run one isolated offline build per CLI process and aggregate results."""

    def __init__(self, config: Mapping[str, Any], *, config_path: str | Path):
        validate_config(config)
        self.config = deepcopy(dict(config))
        self.config_path = Path(config_path).resolve()
        self.fingerprint = config_fingerprint(self.config)
        self.paths = OfflineBuildPaths(
            _resolve_path(self.config["artifacts"]["output_dir"]).resolve()
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OfflineBuildParallelismRunner":
        return cls(load_config(path), config_path=path)

    @property
    def cases(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.config["cases"])

    @property
    def variants(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self.config["variants"])

    def _source_config(self, name: str) -> Path:
        return _resolve_path(self.config["source_configs"][name]).resolve()

    def _reference_path(self, case_id: str, case: Mapping[str, Any]) -> Path:
        configured = case.get("reference_atlas")
        if configured is not None:
            return _resolve_path(configured).resolve()
        return self.paths.generated_reference(case_id).resolve()

    def _valid_result(self, case_id: str, variant_id: str) -> bool:
        path = self.paths.metadata(case_id, variant_id)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            payload.get("status") == "complete"
            and payload.get("config_fingerprint") == self.fingerprint
            and payload.get("implementation_commit") == _git_commit()
        )

    def _variant_applies(self, case_id: str, variant_id: str) -> bool:
        selected = self.config["variants"][variant_id].get("cases")
        return selected is None or case_id in selected

    @staticmethod
    def _predict_support(model: torch.nn.Module, values: np.ndarray, device: str) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for start in range(0, len(values), 2048):
                logits = model(torch.from_numpy(values[start:start + 2048]).to(device))
                parts.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(parts).astype(np.int64, copy=False)

    def _build_heloc(
        self,
        case: Mapping[str, Any],
        eps_workers: int,
        build_workers: int,
        lirpa_batch_size: int | None = None,
    ):
        source = LiRPARefinementAblationRunner.from_yaml(
            self._source_config("lirpa_ablation")
        )
        source.prepare()
        selected = source.resolve_cases([str(case["source_case"])])[0]
        geometry = source._geometry(str(case["source_case"]))
        device = ablation_device(source.config)
        _, _, _, _, model, _ = source._load_case(selected, device)
        method = source._make_certcf(selected, model, geometry, device)
        method.epsilon_parallelism = eps_workers
        method.build_parallelism = build_workers
        if lirpa_batch_size is not None:
            method.batch_size = int(lirpa_batch_size)
        interval = float(source.config["runtime"]["rss_sample_interval_seconds"])
        with PhaseResourceMonitor("build", device, interval) as monitor:
            method.fit(geometry["x_anchor"], geometry["y_anchor_pred"])
        return method, monitor.metrics, device

    def _build_fcnn(
        self,
        case: Mapping[str, Any],
        eps_workers: int,
        build_workers: int,
        lirpa_batch_size: int | None = None,
    ):
        source = NetworkComplexityRunner.from_yaml(
            self._source_config("network_complexity")
        )
        source.configure_epsilon_parallelism(eps_workers)
        source.configure_build_parallelism(build_workers)
        source.prepare(announce=False)
        depth, width = int(case["depth"]), int(case["width"])
        device = _normalize_device(source.config["certcf"]["device"])
        model = source._load_model(depth, width, device)
        x_train, _, _, _, _ = source._shared_benchmark_data()
        y_support = self._predict_support(model, x_train, device)
        method, metrics = source._make_certcf(
            model,
            x_train,
            y_support,
            device=device,
            batch_size=(
                source._effective_lirpa_batch_size()
                if lirpa_batch_size is None
                else int(lirpa_batch_size)
            ),
        )
        return method, metrics, device

    def _build_cifar(
        self,
        case: Mapping[str, Any],
        eps_workers: int,
        build_workers: int,
        cnn_radius_batch_size: int,
        cnn_radius_batch_max_relative_inflation: float,
    ):
        source = CifarResNetScalingRunner.from_yaml(
            self._source_config("cifar_scaling")
        )
        source.configure_epsilon_parallelism(eps_workers)
        source.configure_build_parallelism(build_workers)
        source.prepare()
        prepared = source._prepared()
        device = cifar_device(source.config)
        network = str(case["network"])
        model = source._model(network, device)
        values = prepared["x_support"].astype(np.float32)
        y_support = source._predict(model, values.reshape(-1, 3, 32, 32), device)
        method = source._method(model, None)
        method.cnn_radius_batch_size = int(cnn_radius_batch_size)
        method.cnn_radius_batch_max_relative_inflation = float(
            cnn_radius_batch_max_relative_inflation
        )
        interval = float(source.config["resources"]["rss_sample_interval_seconds"])
        with PhaseResourceMonitor("build", device, interval) as monitor:
            method.fit(values, y_support)
        return method, monitor.metrics, device

    @staticmethod
    def _load_reference_centers(reference_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        manifest = json.loads(
            (reference_dir / "manifest.json").read_text(encoding="utf-8")
        )
        centers = []
        labels = []
        radii = []
        for label in manifest["class_labels"]:
            with np.load(
                reference_dir / manifest["files"][str(label)],
                allow_pickle=False,
            ) as data:
                centers.append(data["X"].astype(np.float32))
                labels.append(np.full(len(data["X"]), int(label), dtype=np.int64))
                radii.append(data["eps_initial"].astype(np.float64))
        return np.concatenate(centers), np.concatenate(labels), np.concatenate(radii)

    def run_epsilon_sweep(
        self,
        *,
        case_id: str,
        worker_counts: list[int],
        repetitions: int,
        force: bool = False,
    ) -> pd.DataFrame:
        """Measure Equation (1) independently of model loading and LiRPA."""
        if case_id not in self.config["cases"]:
            raise ValueError(f"unknown case: {case_id}")
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        worker_counts = list(dict.fromkeys(int(value) for value in worker_counts))
        if not worker_counts or any(value <= 0 for value in worker_counts):
            raise ValueError("worker counts must be positive")

        reference = self._reference_path(case_id, self.config["cases"][case_id])
        if not (reference / "manifest.json").exists():
            raise RuntimeError(f"missing serial reference atlas for {case_id}")
        X, labels, expected_radii = self._load_reference_centers(reference)

        source = CifarResNetScalingRunner.from_yaml(
            self._source_config("cifar_scaling")
        )
        certcf_config = source.config["certcf"]
        strategy = NearestOppositeClassClearanceStrategy(
            alpha=float(certcf_config["eps_alpha"]),
            chunk_size=int(certcf_config["eps_reference_chunk_size"]),
        )

        columns = [
            "case_id",
            "implementation_commit",
            "repetition",
            "workers",
            "epsilon_time_s",
            "radii_exact",
            "radii_max_absolute_difference",
        ]
        if self.paths.epsilon_sweep.exists() and not force:
            frame = pd.read_parquet(self.paths.epsilon_sweep)
        else:
            frame = pd.DataFrame(columns=columns)

        completed = {
            (int(row.repetition), int(row.workers))
            for row in frame.itertuples()
            if row.case_id == case_id
        }
        rng = np.random.default_rng(20260905)
        for repetition in range(repetitions):
            ordered_workers = worker_counts.copy()
            rng.shuffle(ordered_workers)
            for workers in ordered_workers:
                if (repetition, workers) in completed:
                    continue
                gc.collect()
                started = time.perf_counter()
                observed = strategy.compute_eps(
                    X,
                    labels,
                    norm=1,
                    parallelism=workers,
                )
                elapsed = time.perf_counter() - started
                difference = np.abs(observed - expected_radii)
                row = pd.DataFrame(
                    [
                        {
                            "case_id": case_id,
                            "implementation_commit": _git_commit(),
                            "repetition": repetition,
                            "workers": workers,
                            "epsilon_time_s": elapsed,
                            "radii_exact": bool(
                                np.array_equal(observed, expected_radii)
                            ),
                            "radii_max_absolute_difference": float(
                                np.max(difference)
                            ),
                        }
                    ]
                )
                frame = pd.concat([frame, row], ignore_index=True)
                self.paths.epsilon_sweep.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(self.paths.epsilon_sweep, index=False)
                print(
                    f"[EPSILON] repetition={repetition + 1}/{repetitions}, "
                    f"workers={workers}: {elapsed:.3f}s, "
                    f"exact={bool(np.array_equal(observed, expected_radii))}",
                    flush=True,
                )
        return frame.sort_values(["workers", "repetition"]).reset_index(drop=True)

    @staticmethod
    def _diagnostics(method) -> dict[str, Any]:
        atlas = method.atlas
        if atlas is None or atlas.bounds is None:
            raise RuntimeError("CertCF did not produce an atlas")
        bounds = [atlas.bounds[label] for label in atlas.class_labels]
        initial = np.concatenate([np.asarray(value["eps_initial"]) for value in bounds])
        final = np.concatenate([np.asarray(value["eps"]) for value in bounds])
        shrinks = np.concatenate(
            [np.asarray(value["adaptive_eps_n_shrinks"]) for value in bounds]
        )
        return {
            "atlas_region_count": int(sum(len(value["X"]) for value in bounds)),
            "atlas_region_counts_by_class": {
                str(label): int(len(atlas.bounds[label]["X"]))
                for label in atlas.class_labels
            },
            "eps_initial_median": float(np.median(initial)),
            "eps_final_median": float(np.median(final)),
            "eps_retention_median": float(
                np.median(final / np.maximum(initial, 1.0e-30))
            ),
            "adaptive_shrinks_mean": float(np.mean(shrinks)),
            "adaptive_shrinks_max": int(np.max(shrinks)),
            **dict(atlas.build_profiling),
        }

    def run(self, case_id: str, variant_id: str, *, force: bool = False) -> dict[str, Any]:
        if case_id not in self.config["cases"]:
            raise ValueError(f"unknown case: {case_id}")
        if variant_id not in self.config["variants"]:
            raise ValueError(f"unknown variant: {variant_id}")
        if not self._variant_applies(case_id, variant_id):
            raise ValueError(f"variant {variant_id} does not apply to case {case_id}")
        if self._valid_result(case_id, variant_id) and not force:
            return json.loads(
                self.paths.metadata(case_id, variant_id).read_text(encoding="utf-8")
            )

        case = self.config["cases"][case_id]
        variant = self.config["variants"][variant_id]
        eps_workers = int(variant["epsilon_parallelism"])
        build_workers = int(variant["build_parallelism"])
        lirpa_batch_size = variant.get("lirpa_batch_size")
        if lirpa_batch_size is not None:
            lirpa_batch_size = int(lirpa_batch_size)
        cnn_radius_batch_size = int(variant.get("cnn_radius_batch_size", 1))
        cnn_radius_batch_max_relative_inflation = float(
            variant.get("cnn_radius_batch_max_relative_inflation", 0.0)
        )
        print(
            f"[OFFLINE] {case_id}/{variant_id}: epsilon workers={eps_workers}, "
            f"LiRPA workers={build_workers}, LiRPA batch={lirpa_batch_size or 'source'}, "
            f"CNN radius batch={cnn_radius_batch_size}",
            flush=True,
        )
        kind = str(case["kind"])
        if kind == "lirpa_ablation":
            method, resources, device = self._build_heloc(
                case, eps_workers, build_workers, lirpa_batch_size
            )
        elif kind == "network_complexity":
            method, resources, device = self._build_fcnn(
                case, eps_workers, build_workers, lirpa_batch_size
            )
        else:
            method, resources, device = self._build_cifar(
                case,
                eps_workers,
                build_workers,
                cnn_radius_batch_size,
                cnn_radius_batch_max_relative_inflation,
            )

        atlas = method.atlas
        diagnostics = self._diagnostics(method)
        reference = self._reference_path(case_id, case)
        reference_created = False
        serialization_time_s = 0.0
        if not (reference / "manifest.json").exists():
            if variant_id != "serial":
                raise RuntimeError(
                    f"missing serial reference atlas for {case_id}: run serial first"
                )
            started = time.perf_counter()
            atlas.save_bounds(reference)
            serialization_time_s = time.perf_counter() - started
            reference_created = True

        comparison_cfg = self.config["comparison"]
        comparison = compare_bounds_to_saved(
            atlas.bounds,
            reference,
            atol=float(comparison_cfg["absolute_tolerance"]),
            rtol=float(comparison_cfg["relative_tolerance"]),
            chunk_elements=int(comparison_cfg["chunk_elements"]),
        )
        payload = {
            "status": "complete",
            "case_id": case_id,
            "case_kind": kind,
            "variant_id": variant_id,
            "config_fingerprint": self.fingerprint,
            "implementation_commit": _git_commit(),
            "device": str(device),
            "epsilon_parallelism": eps_workers,
            "build_parallelism": build_workers,
            "lirpa_batch_size": lirpa_batch_size,
            "cnn_radius_batch_size": cnn_radius_batch_size,
            "cnn_radius_batch_max_relative_inflation": (
                cnn_radius_batch_max_relative_inflation
            ),
            "reference_created": reference_created,
            "reference_serialization_time_s": serialization_time_s,
            **resources,
            **diagnostics,
            "equivalence": comparison,
        }
        _atomic_json(payload, self.paths.metadata(case_id, variant_id))
        print(
            f"[OFFLINE] {case_id}/{variant_id}: "
            f"{payload['build_wall_time_s']:.3f}s, "
            f"equivalent={comparison['all_close']}",
            flush=True,
        )
        if atlas is not None:
            atlas.close_candidate_process_pool()
        del method
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return payload

    def run_batch_size_sweep(
        self,
        *,
        case_id: str,
        batch_sizes: list[int],
        repetitions: int,
        force: bool = False,
    ) -> pd.DataFrame:
        """Sweep FC LiRPA batches or CNN radius buckets with checkpointed rows."""
        if case_id not in self.config["cases"]:
            raise ValueError(f"unknown case: {case_id}")
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        batch_sizes = list(dict.fromkeys(int(value) for value in batch_sizes))
        if not batch_sizes or any(value <= 0 for value in batch_sizes):
            raise ValueError("batch sizes must be positive")

        case = self.config["cases"][case_id]
        kind = str(case["kind"])
        reference = self._reference_path(case_id, case)
        if not (reference / "manifest.json").exists():
            raise RuntimeError(
                f"missing reference atlas for {case_id}: run its serial variant first"
            )

        path = self.paths.batch_size_sweep
        if path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = pd.DataFrame()
        if force and not frame.empty:
            frame = frame[frame["case_id"] != case_id].copy()

        completed = set()
        if not frame.empty:
            completed = {
                (int(row.repetition), int(row.batch_size))
                for row in frame.itertuples()
                if row.case_id == case_id and row.status == "complete"
            }

        rng = np.random.default_rng(20260906)
        for repetition in range(repetitions):
            ordered = batch_sizes.copy()
            rng.shuffle(ordered)
            for batch_size in ordered:
                identity = (repetition, batch_size)
                if identity in completed:
                    continue
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                print(
                    f"[BATCH SWEEP] {case_id}: repetition={repetition + 1}/"
                    f"{repetitions}, batch={batch_size}",
                    flush=True,
                )
                method = None
                try:
                    if kind == "lirpa_ablation":
                        method, resources, device = self._build_heloc(
                            case, 1, 1, batch_size
                        )
                        batch_kind = "variable_epsilon"
                    elif kind == "network_complexity":
                        method, resources, device = self._build_fcnn(
                            case, 1, 1, batch_size
                        )
                        batch_kind = "variable_epsilon"
                    elif kind == "cifar_scaling":
                        method, resources, device = self._build_cifar(
                            case,
                            8,
                            1,
                            batch_size,
                            0.05,
                        )
                        batch_kind = "cnn_radius_bucket"
                    else:  # pragma: no cover - guarded by config validation
                        raise ValueError(f"unsupported case kind: {kind}")

                    diagnostics = self._diagnostics(method)
                    comparison_cfg = self.config["comparison"]
                    comparison = compare_bounds_to_saved(
                        method.atlas.bounds,
                        reference,
                        atol=float(comparison_cfg["absolute_tolerance"]),
                        rtol=float(comparison_cfg["relative_tolerance"]),
                        chunk_elements=int(comparison_cfg["chunk_elements"]),
                    )
                    decision_arrays = [
                        value
                        for identity, value in comparison["per_array"].items()
                        if identity.rsplit(".", maxsplit=1)[-1]
                        in ATLAS_DECISION_KEYS
                    ]
                    row = {
                        "status": "complete",
                        "error": None,
                        "case_id": case_id,
                        "case_kind": kind,
                        "batch_kind": batch_kind,
                        "repetition": repetition,
                        "batch_size": batch_size,
                        "device": str(device),
                        "implementation_commit": _git_commit(),
                        **resources,
                        **diagnostics,
                        "atlas_all_exact": bool(comparison["all_exact"]),
                        "atlas_all_close": bool(comparison["all_close"]),
                        "atlas_decisions_exact": bool(decision_arrays)
                        and all(value["exact"] for value in decision_arrays),
                        "atlas_decisions_close": bool(decision_arrays)
                        and all(value["allclose"] for value in decision_arrays),
                        "atlas_max_absolute_difference": float(
                            comparison["maximum_absolute_difference"]
                        ),
                    }
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "case_id": case_id,
                        "case_kind": kind,
                        "batch_kind": (
                            "cnn_radius_bucket"
                            if kind == "cifar_scaling"
                            else "variable_epsilon"
                        ),
                        "repetition": repetition,
                        "batch_size": batch_size,
                        "implementation_commit": _git_commit(),
                    }
                    print(
                        f"[BATCH SWEEP] {case_id} batch={batch_size} failed: {exc}",
                        flush=True,
                    )
                finally:
                    if method is not None and method.atlas is not None:
                        method.atlas.close_candidate_process_pool()
                    del method
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
                if row["status"] == "complete":
                    print(
                        f"[BATCH SWEEP] {case_id} batch={batch_size}: "
                        f"{row['build_wall_time_s']:.3f}s, "
                        f"equivalent={row['atlas_all_close']}",
                        flush=True,
                    )

        return frame.sort_values(
            ["case_id", "batch_size", "repetition"]
        ).reset_index(drop=True)

    def analyze_batch_size_sweep(self) -> pd.DataFrame:
        """Aggregate successful batch-size measurements across repetitions."""
        path = self.paths.batch_size_sweep
        if not path.exists():
            raise RuntimeError("batch-size sweep has not been run")
        frame = pd.read_parquet(path)
        complete = frame[frame["status"] == "complete"].copy()
        if complete.empty:
            raise RuntimeError("batch-size sweep contains no successful measurements")
        summary = (
            complete.groupby(
                ["case_id", "case_kind", "batch_kind", "batch_size"],
                as_index=False,
            )
            .agg(
                repetitions=("build_wall_time_s", "size"),
                build_mean_s=("build_wall_time_s", "mean"),
                build_std_s=("build_wall_time_s", "std"),
                build_median_s=("build_wall_time_s", "median"),
                lirpa_mean_s=("lirpa_time_s", "mean"),
                lirpa_std_s=("lirpa_time_s", "std"),
                epsilon_mean_s=("epsilon_time_s", "mean"),
                rss_peak_mean_bytes=("build_rss_peak_delta_bytes", "mean"),
                cuda_peak_mean_bytes=("build_cuda_peak_allocated_bytes", "mean"),
                all_equivalent=("atlas_all_close", "all"),
                all_decisions_exact=("atlas_decisions_exact", "all"),
                all_decisions_close=("atlas_decisions_close", "all"),
                maximum_difference=("atlas_max_absolute_difference", "max"),
                fallback_mean=("cnn_radius_bucket_fallback_count", "mean"),
            )
            .sort_values(["case_id", "batch_size"])
            .reset_index(drop=True)
        )
        baselines = summary[summary["batch_size"] == 1].set_index("case_id")[
            "build_mean_s"
        ]
        summary["speedup_vs_batch1"] = (
            summary["case_id"].map(baselines) / summary["build_mean_s"]
        )
        self.paths.batch_size_summary.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(self.paths.batch_size_summary, index=False)
        return summary

    def run_fcnn_grid_builds(
        self,
        *,
        repetitions: int,
        force: bool = False,
    ) -> pd.DataFrame:
        """Benchmark the selected optimized offline build on all FCNN cells."""
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        protocol = self.config.get("fcnn_grid")
        if not protocol:
            raise ValueError("fcnn_grid configuration is missing")
        optimized = protocol["optimized"]
        grid_fingerprint = _mapping_fingerprint(protocol)
        eps_workers = int(optimized["epsilon_parallelism"])
        build_workers = int(optimized["build_parallelism"])
        batch_size = int(optimized["lirpa_batch_size"])

        source = NetworkComplexityRunner.from_yaml(
            self._source_config("network_complexity")
        )
        source.prepare(announce=False)
        grid = list(source.grid)
        path = self.paths.fcnn_grid_sweep
        if path.exists() and not force:
            frame = pd.read_parquet(path)
        else:
            frame = pd.DataFrame()
        if force:
            frame = pd.DataFrame()
        completed = {
            (int(row.repetition), int(row.depth), int(row.width))
            for row in frame.itertuples()
            if row.status == "complete"
            and row.grid_fingerprint == grid_fingerprint
        }

        tasks = [
            (repetition, depth, width)
            for repetition in range(repetitions)
            for depth, width in grid
        ]
        rng = np.random.default_rng(20260906)
        rng.shuffle(tasks)
        for repetition, depth, width in tasks:
            identity = (repetition, depth, width)
            if identity in completed:
                continue
            method = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            architecture = architecture_id(depth, width)
            print(
                f"[FCNN GRID {repetition + 1}/{repetitions}] {architecture}: "
                f"epsilon workers={eps_workers}, LiRPA workers={build_workers}, "
                f"batch={batch_size}",
                flush=True,
            )
            try:
                method, resources, device = self._build_fcnn(
                    {"depth": depth, "width": width},
                    eps_workers,
                    build_workers,
                    batch_size,
                )
                row = {
                    "status": "complete",
                    "error": None,
                    "grid_fingerprint": grid_fingerprint,
                    "implementation_commit": _git_commit(),
                    "repetition": repetition,
                    "architecture_id": architecture,
                    "depth": depth,
                    "width": width,
                    "epsilon_parallelism": eps_workers,
                    "build_parallelism": build_workers,
                    "lirpa_batch_size": batch_size,
                    "device": str(device),
                    **resources,
                    **self._diagnostics(method),
                    "atlas_decision_fingerprint": _atlas_decision_fingerprint(method),
                }
            except Exception as exc:
                row = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "grid_fingerprint": grid_fingerprint,
                    "implementation_commit": _git_commit(),
                    "repetition": repetition,
                    "architecture_id": architecture,
                    "depth": depth,
                    "width": width,
                    "epsilon_parallelism": eps_workers,
                    "build_parallelism": build_workers,
                    "lirpa_batch_size": batch_size,
                }
                print(f"[FCNN GRID] {architecture} failed: {exc}", flush=True)
            finally:
                if method is not None and method.atlas is not None:
                    method.atlas.close_candidate_process_pool()
                del method
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(path)
            if row["status"] == "complete":
                print(
                    f"[FCNN GRID] {architecture}: "
                    f"build={row['build_wall_time_s']:.3f}s",
                    flush=True,
                )

        return frame.sort_values(
            ["depth", "width", "repetition"]
        ).reset_index(drop=True)

    def analyze_fcnn_grid(self) -> pd.DataFrame:
        """Combine historical serial and optimized FCNN build/query timings."""
        protocol = self.config.get("fcnn_grid")
        if not protocol:
            raise ValueError("fcnn_grid configuration is missing")
        if not self.paths.fcnn_grid_sweep.exists():
            raise RuntimeError("optimized FCNN grid builds have not been run")
        raw = pd.read_parquet(self.paths.fcnn_grid_sweep)
        grid_fingerprint = _mapping_fingerprint(protocol)
        complete = raw[
            (raw["status"] == "complete")
            & (raw["grid_fingerprint"] == grid_fingerprint)
        ].copy()
        optimized = (
            complete.groupby(
                ["architecture_id", "depth", "width"], as_index=False
            )
            .agg(
                optimized_repetitions=("build_wall_time_s", "size"),
                optimized_build_mean_s=("build_wall_time_s", "mean"),
                optimized_build_std_s=("build_wall_time_s", "std"),
                optimized_lirpa_mean_s=("lirpa_time_s", "mean"),
                optimized_epsilon_mean_s=("epsilon_time_s", "mean"),
                optimized_cuda_peak_mean_bytes=(
                    "build_cuda_peak_allocated_bytes",
                    "mean",
                ),
                optimized_region_count=("atlas_region_count", "min"),
                optimized_decision_fingerprints=(
                    "atlas_decision_fingerprint",
                    "nunique",
                ),
            )
        )

        serial = pd.read_parquet(_resolve_path(protocol["serial_summary"]))
        parallel = pd.read_parquet(_resolve_path(protocol["parallel_summary"]))
        serial = serial[
            [
                "architecture_id",
                "parameter_count",
                "build_wall_time_s",
                "query_wall_time_s",
                "query_time_median_s",
                "query_time_p95_s",
                "n_queries",
            ]
        ].rename(
            columns={
                "build_wall_time_s": "serial_build_s",
                "query_time_median_s": "serial_query_median_s",
                "query_time_p95_s": "serial_query_p95_s",
            }
        )
        serial["serial_query_mean_s"] = (
            serial["query_wall_time_s"] / serial["n_queries"]
        )
        serial = serial.drop(columns=["query_wall_time_s", "n_queries"])
        parallel = parallel[
            [
                "architecture_id",
                "query_wall_time_s",
                "query_time_median_s",
                "query_time_p95_s",
                "n_queries",
            ]
        ].rename(
            columns={
                "query_time_median_s": "parallel_query_median_s",
                "query_time_p95_s": "parallel_query_p95_s",
            }
        )
        parallel["parallel_query_mean_s"] = (
            parallel["query_wall_time_s"] / parallel["n_queries"]
        )
        parallel = parallel.drop(columns=["query_wall_time_s", "n_queries"])

        comparison = (
            optimized.merge(serial, on="architecture_id", validate="one_to_one")
            .merge(parallel, on="architecture_id", validate="one_to_one")
            .sort_values(["depth", "width"])
            .reset_index(drop=True)
        )
        expected = len(NetworkComplexityRunner.from_yaml(
            self._source_config("network_complexity")
        ).grid)
        if len(comparison) != expected:
            raise RuntimeError(
                f"FCNN grid is incomplete: found {len(comparison)} of {expected} cells"
            )
        comparison["build_speedup"] = (
            comparison["serial_build_s"]
            / comparison["optimized_build_mean_s"]
        )
        comparison["query_median_speedup"] = (
            comparison["serial_query_median_s"]
            / comparison["parallel_query_median_s"]
        )

        models: dict[str, Any] = {}
        for label, response in (
            ("serial", "serial_build_s"),
            ("optimized", "optimized_build_mean_s"),
        ):
            fit = _fit_interaction_model(comparison, response)
            comparison[f"{label}_build_loo_prediction_s"] = fit.pop(
                "loo_prediction"
            )
            models[label] = fit
        models["formula"] = "intercept + depth + (depth - 1) * width"
        models["architectures"] = len(comparison)

        self.paths.fcnn_grid_comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_parquet(self.paths.fcnn_grid_comparison, index=False)
        _atomic_json(models, self.paths.fcnn_grid_models)
        return comparison

    def analyze(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for case_id in self.config["cases"]:
            for variant_id in self.config["variants"]:
                if not self._variant_applies(case_id, variant_id):
                    continue
                path = self.paths.metadata(case_id, variant_id)
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "complete":
                    continue
                equivalence = payload.get("equivalence", {})
                per_array = equivalence.get("per_array", {})
                decision_arrays = [
                    comparison
                    for identity, comparison in per_array.items()
                    if identity.rsplit(".", maxsplit=1)[-1] in ATLAS_DECISION_KEYS
                ]
                rows.append(
                    {
                        "case_id": case_id,
                        "variant_id": variant_id,
                        "epsilon_parallelism": int(payload["epsilon_parallelism"]),
                        "build_parallelism": int(payload["build_parallelism"]),
                        "cnn_radius_batch_size": int(
                            payload.get("cnn_radius_batch_size", 1)
                        ),
                        "cnn_radius_batch_max_relative_inflation": float(
                            payload.get(
                                "cnn_radius_batch_max_relative_inflation",
                                0.0,
                            )
                        ),
                        "lirpa_workers_used": int(payload["lirpa_workers_used"]),
                        "build_wall_time_s": float(payload["build_wall_time_s"]),
                        "epsilon_time_s": float(payload["epsilon_time_s"]),
                        "lirpa_time_s": float(payload["lirpa_time_s"]),
                        "build_rss_peak_delta_bytes": int(
                            payload["build_rss_peak_delta_bytes"]
                        ),
                        "build_cuda_peak_allocated_bytes": int(
                            payload["build_cuda_peak_allocated_bytes"]
                        ),
                        "build_cuda_peak_reserved_bytes": int(
                            payload["build_cuda_peak_reserved_bytes"]
                        ),
                        "atlas_region_count": int(payload["atlas_region_count"]),
                        "atlas_decisions_exact": bool(decision_arrays)
                        and all(item["exact"] for item in decision_arrays),
                        "atlas_all_close": bool(equivalence.get("all_close", False)),
                        "atlas_all_exact": bool(equivalence.get("all_exact", False)),
                        "atlas_max_absolute_difference": float(
                            equivalence.get("maximum_absolute_difference", float("inf"))
                        ),
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("no completed offline build results")
        serial = (
            frame[frame["variant_id"] == "serial"]
            .set_index("case_id")["build_wall_time_s"]
        )
        frame["serial_build_wall_time_s"] = frame["case_id"].map(serial)
        frame["speedup_vs_serial"] = (
            frame["serial_build_wall_time_s"] / frame["build_wall_time_s"]
        )
        self.paths.summary.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values(["case_id", "variant_id"]).to_parquet(
            self.paths.summary, index=False
        )
        return frame.sort_values(["case_id", "variant_id"]).reset_index(drop=True)

    def status(self) -> dict[str, Any]:
        complete = []
        missing = []
        for case_id in self.config["cases"]:
            for variant_id in self.config["variants"]:
                if not self._variant_applies(case_id, variant_id):
                    continue
                pair = f"{case_id}/{variant_id}"
                if self._valid_result(case_id, variant_id):
                    complete.append(pair)
                else:
                    missing.append(pair)
        return {"complete": complete, "missing": missing}
