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
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
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
    def summary(self) -> Path:
        return self.root / "offline_build_summary.parquet"


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

    @staticmethod
    def _predict_support(model: torch.nn.Module, values: np.ndarray, device: str) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for start in range(0, len(values), 2048):
                logits = model(torch.from_numpy(values[start:start + 2048]).to(device))
                parts.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(parts).astype(np.int64, copy=False)

    def _build_heloc(self, case: Mapping[str, Any], eps_workers: int, build_workers: int):
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
        interval = float(source.config["runtime"]["rss_sample_interval_seconds"])
        with PhaseResourceMonitor("build", device, interval) as monitor:
            method.fit(geometry["x_anchor"], geometry["y_anchor_pred"])
        return method, monitor.metrics, device

    def _build_fcnn(self, case: Mapping[str, Any], eps_workers: int, build_workers: int):
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
            batch_size=source._effective_lirpa_batch_size(),
        )
        return method, metrics, device

    def _build_cifar(self, case: Mapping[str, Any], eps_workers: int, build_workers: int):
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
        interval = float(source.config["resources"]["rss_sample_interval_seconds"])
        with PhaseResourceMonitor("build", device, interval) as monitor:
            method.fit(values, y_support)
        return method, monitor.metrics, device

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
        if self._valid_result(case_id, variant_id) and not force:
            return json.loads(
                self.paths.metadata(case_id, variant_id).read_text(encoding="utf-8")
            )

        case = self.config["cases"][case_id]
        variant = self.config["variants"][variant_id]
        eps_workers = int(variant["epsilon_parallelism"])
        build_workers = int(variant["build_parallelism"])
        print(
            f"[OFFLINE] {case_id}/{variant_id}: epsilon workers={eps_workers}, "
            f"LiRPA workers={build_workers}",
            flush=True,
        )
        kind = str(case["kind"])
        if kind == "lirpa_ablation":
            method, resources, device = self._build_heloc(
                case, eps_workers, build_workers
            )
        elif kind == "network_complexity":
            method, resources, device = self._build_fcnn(
                case, eps_workers, build_workers
            )
        else:
            method, resources, device = self._build_cifar(
                case, eps_workers, build_workers
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

    def analyze(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for case_id in self.config["cases"]:
            for variant_id in self.config["variants"]:
                path = self.paths.metadata(case_id, variant_id)
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "complete":
                    continue
                equivalence = payload.get("equivalence", {})
                rows.append(
                    {
                        "case_id": case_id,
                        "variant_id": variant_id,
                        "epsilon_parallelism": int(payload["epsilon_parallelism"]),
                        "build_parallelism": int(payload["build_parallelism"]),
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
                        "atlas_region_count": int(payload["atlas_region_count"]),
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
                pair = f"{case_id}/{variant_id}"
                if self._valid_result(case_id, variant_id):
                    complete.append(pair)
                else:
                    missing.append(pair)
        return {"complete": complete, "missing": missing}
