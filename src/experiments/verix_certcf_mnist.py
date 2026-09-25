"""Resumable paired VERIX/CertCF benchmark on the repository's MNIST LeNet-5.

VERIX is run first.  Its nearest canonically valid native counterexample
determines the target class used by CertCF for the same query.  This preserves
the semantics of VERIX, which explains the source prediction rather than
accepting a user-selected counterfactual target.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from verix.lenet5_export import (
    MarabouCompatibleLeNet5,
    export_lenet5_onnx,
    load_lenet5_checkpoint,
    onnx_operator_types,
    sha256_file,
    validate_lenet5_equivalence,
    validate_marabou_parse,
)
from verix.mnist import (
    MNIST_PIXELS,
    ONNXMNISTScorer,
    run_mnist_verix,
    select_nearest_valid_witness,
    validate_mnist_witnesses,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "checkpoint": "checkpoints/mnist_lenet5_classifier/best.ckpt",
        "data_dir": "data/",
    },
    "queries": {
        "pilot_indices": [10],
        "benchmark_policy": "first_n",
        "benchmark_size": 100,
    },
    "export": {
        "opset_version": 13,
        "validation_batch_size": 256,
        "validate_full_test_set": True,
        "atol": 1.0e-5,
        "rtol": 1.0e-5,
    },
    "verix": {
        "epsilon": 0.05,
        "traversal": "reversal",
        "feature_granularity": "pixel",
        "input_bounds": [0.0, 1.0],
        "discrepancy": 0.0,
        "classification_margin": 1.0e-6,
        "num_workers": 16,
        "timeout_seconds": 300,
        "solve_with_milp": False,
        "witness_selection_distance": 1,
    },
    "certcf": {
        "device": "auto",
        "train_per_true_class": 1000,
        "anchors_per_predicted_class": 1000,
        "atlas_subsample_method": "random",
        "norm": "inf",
        "distance_norm": 1,
        "eps_alpha": 0.20,
        "lirpa_method": "backward",
        "lirpa_batch_size": 8,
        "classification_margin": 0.0,
        "adaptive_eps": True,
        "adaptive_eps_shrink_factor": 0.5,
        "adaptive_eps_max_shrinks": 8,
        "adaptive_eps_min": 1.0e-6,
        "adaptive_eps_center_tol": 1.0e-6,
        "adaptive_eps_binary_search_steps": 0,
        "query_method": "nearest_anchor",
        "query_k_candidates": 3,
        "query_parallelism": 1,
        "timeout_seconds_per_query": 120,
        "solver_maxiter": 500,
        "cvxpy_solvers": ["CLARABEL"],
        "input_bounds": [0.0, 1.0],
        "sparsity_penalty": "none",
    },
    "robustness": {
        "empirical": {
            "enabled": True,
            "radii": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
            "samples_per_radius": 100,
        },
        "certified": {
            "enabled": True,
            "max_radius": 0.05,
            "binary_search_steps": 10,
            "lirpa_method": "backward",
        },
    },
    "resources": {"rss_sample_interval_seconds": 0.05},
    "artifacts": {"output_dir": "results/verix_certcf_mnist"},
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


def _as_norm(value: Any) -> int | float:
    return np.inf if str(value).strip().lower() in {"inf", "infinity"} else int(value)


def validate_config(config: dict[str, Any]) -> None:
    queries = config["queries"]
    if queries["benchmark_policy"] != "first_n":
        raise ValueError("queries.benchmark_policy must be 'first_n'")
    if int(queries["benchmark_size"]) <= 0 or int(queries["benchmark_size"]) > 10_000:
        raise ValueError("queries.benchmark_size must lie in [1, 10000]")
    pilots = [int(value) for value in queries["pilot_indices"]]
    if any(value < 0 or value >= 10_000 for value in pilots):
        raise ValueError("pilot indices must lie in [0, 9999]")

    verix = config["verix"]
    if verix["traversal"] != "reversal":
        raise ValueError("the paper-faithful MNIST protocol requires reversal traversal")
    if verix["feature_granularity"] != "pixel":
        raise ValueError("the paper-faithful MNIST protocol treats each pixel as a feature")
    if _as_norm("inf") != np.inf or float(verix["epsilon"]) < 0:
        raise ValueError("verix.epsilon must be non-negative")
    if list(map(float, verix["input_bounds"])) != [0.0, 1.0]:
        raise ValueError("MNIST VERIX input bounds must be [0, 1]")
    if float(verix["discrepancy"]) != 0.0:
        raise ValueError("classification VERIX requires discrepancy=0")
    if int(verix["witness_selection_distance"]) != 1:
        raise ValueError("paired target selection is fixed to L1 distance")

    certcf = config["certcf"]
    if int(certcf["train_per_true_class"]) <= 0:
        raise ValueError("certcf.train_per_true_class must be positive")
    if int(certcf["anchors_per_predicted_class"]) <= 0:
        raise ValueError("certcf.anchors_per_predicted_class must be positive")
    if certcf["atlas_subsample_method"] == "kmedoids":
        raise ValueError("this experiment must not use k-medoids")
    if int(certcf["query_k_candidates"]) <= 0:
        raise ValueError("certcf.query_k_candidates must be positive")


def config_fingerprint(config: dict[str, Any]) -> str:
    protocol = deepcopy(config)
    protocol.pop("artifacts", None)
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def query_indices(config: dict[str, Any], *, pilot: bool = False) -> np.ndarray:
    if pilot:
        return np.asarray(config["queries"]["pilot_indices"], dtype=np.int64)
    return np.arange(int(config["queries"]["benchmark_size"]), dtype=np.int64)


def balanced_class_indices(
    labels: np.ndarray,
    per_class: int,
    *,
    seed: int,
) -> np.ndarray:
    """Deterministically sample exactly ``per_class`` rows for every class."""

    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for label in sorted(np.unique(labels).tolist()):
        available = np.flatnonzero(labels == label)
        if len(available) < int(per_class):
            raise ValueError(
                f"class {label} has {len(available)} rows, fewer than {per_class}"
            )
        selected.extend(
            rng.choice(available, size=int(per_class), replace=False).tolist()
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def counterfactual_metrics(
    query: np.ndarray,
    counterfactual: np.ndarray | None,
    *,
    target: int | None,
    logits_function: Callable[[np.ndarray], np.ndarray],
    l0_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    if counterfactual is None or target is None:
        return {
            "success": False,
            "predicted_class": None,
            "l1_distance": np.nan,
            "l2_distance": np.nan,
            "l0_changed": np.nan,
            "target_logit_margin": np.nan,
            "pixel_feasible": False,
        }
    x = np.asarray(query, dtype=np.float64).reshape(-1)
    x_cf = np.asarray(counterfactual, dtype=np.float64).reshape(-1)
    logits = np.asarray(logits_function(x_cf[None, :]), dtype=np.float64)[0]
    predicted = int(logits.argmax())
    other = logits.copy()
    other[int(target)] = -np.inf
    diff = np.abs(x_cf - x)
    feasible = bool(
        np.isfinite(x_cf).all()
        and np.all(x_cf >= -1.0e-6)
        and np.all(x_cf <= 1.0 + 1.0e-6)
    )
    return {
        "success": bool(predicted == int(target) and feasible),
        "predicted_class": predicted,
        "l1_distance": float(np.linalg.norm(diff, ord=1)),
        "l2_distance": float(np.linalg.norm(diff, ord=2)),
        "l0_changed": int(np.count_nonzero(diff > l0_tolerance)),
        "target_logit_margin": float(logits[int(target)] - np.max(other)),
        "pixel_feasible": feasible,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_value(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _read_npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _normalize_device(value: Any) -> str:
    raw = str(value or "cpu").strip().lower()
    if raw in {"auto", "gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return raw


class ResourceMonitor(AbstractContextManager):
    """Small phase monitor for wall time, RSS and CUDA peaks."""

    def __init__(self, phase: str, device: str, interval_seconds: float) -> None:
        self.phase = str(phase)
        self.device = torch.device(device)
        self.interval_seconds = max(0.005, float(interval_seconds))
        self.process = psutil.Process(os.getpid())
        self.samples: list[int] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.samples.append(int(self.process.memory_info().rss))

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def __enter__(self):
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA resource monitoring requested, but CUDA is unavailable")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sample()
        self.baseline = self.samples[0]
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(0.1, 2 * self.interval_seconds))
        self._sample()
        peak = max(self.samples)
        self.metrics = {
            f"{self.phase}_wall_time_seconds": time.perf_counter() - self.started,
            f"{self.phase}_rss_baseline_bytes": self.baseline,
            f"{self.phase}_rss_peak_bytes": peak,
            f"{self.phase}_rss_peak_delta_bytes": max(0, peak - self.baseline),
            f"{self.phase}_cuda_peak_allocated_bytes": 0,
            f"{self.phase}_cuda_peak_reserved_bytes": 0,
        }
        if self.device.type == "cuda":
            self.metrics[f"{self.phase}_cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            self.metrics[f"{self.phase}_cuda_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device)
            )
        return False


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def onnx(self) -> Path:
        return self.root / "model" / "lenet5_marabou.onnx"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def export_validation(self) -> Path:
        return self.root / "model" / "equivalence.json"

    @property
    def queries(self) -> Path:
        return self.root / "data" / "queries.npz"

    @property
    def train_pool(self) -> Path:
        return self.root / "data" / "certcf_train_pool.npz"

    @property
    def certcf_build(self) -> Path:
        return self.root / "certcf" / "build.json"

    @property
    def combined(self) -> Path:
        return self.root / "verix_certcf_queries.parquet"

    @property
    def robustness(self) -> Path:
        return self.root / "robustness.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def verix_query(self, index: int) -> Path:
        return self.root / "verix" / f"query_{int(index):05d}.npz"

    def certcf_query(self, index: int) -> Path:
        return self.root / "certcf" / f"query_{int(index):05d}.npz"


class PostHocLiRPACertifier:
    """Target-class L-infinity certification with clipped pixel bounds."""

    def __init__(self, model: torch.nn.Module, *, device: str, method: str) -> None:
        self.model = deepcopy(model).eval().to(device)
        self.device = torch.device(device)
        self.method = str(method)

    def certified(self, x: np.ndarray, target: int, radius: float) -> bool:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        from certcf.certification.wrapping import WrappedModel

        center = torch.as_tensor(
            np.asarray(x, dtype=np.float32).reshape(1, 1, 28, 28),
            device=self.device,
        )
        lower = torch.clamp(center - float(radius), min=0.0, max=1.0)
        upper = torch.clamp(center + float(radius), min=0.0, max=1.0)
        perturbation = PerturbationLpNorm(
            norm=np.inf,
            eps=float(radius),
            x_L=lower,
            x_U=upper,
        )
        bounded_x = BoundedTensor(center, perturbation)
        wrapped = WrappedModel(self.model, int(target), self.device, n_labels=10)
        bounded_model = BoundedModule(wrapped, bounded_x, device=self.device)
        lower_bounds, _ = bounded_model.compute_bounds(
            x=(bounded_x,),
            method=self.method,
        )
        return bool(torch.all(lower_bounds >= 0.0).item())

    def radius(
        self,
        x: np.ndarray,
        target: int,
        *,
        maximum: float,
        steps: int,
    ) -> float:
        if not self.certified(x, target, 0.0):
            return 0.0
        low, high = 0.0, float(maximum)
        if self.certified(x, target, high):
            return high
        for _ in range(int(steps)):
            middle = 0.5 * (low + high)
            if self.certified(x, target, middle):
                low = middle
            else:
                high = middle
        return low


class VeriXCertCFMNISTRunner:
    """Orchestrate preparation, paired methods, robustness and analysis."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
    ) -> None:
        validate_config(config)
        self.config = deepcopy(config)
        self.config_path = None if config_path is None else Path(config_path)
        self.fingerprint = config_fingerprint(self.config)
        self.paths = ArtifactPaths(Path(config["artifacts"]["output_dir"]).resolve())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VeriXCertCFMNISTRunner":
        return cls(load_config(path), config_path=path)

    def _load_mnist(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from counterfactuals.datasets.loaders import MNISTDataset

        dataset = MNISTDataset(
            data_dir=str(self.config["experiment"]["data_dir"]),
            seed=int(self.config["experiment"]["seed"]),
        )
        dataset.load()
        x_train, y_train = dataset.get_train()
        x_test, y_test = dataset.get_test()
        return x_train, y_train, x_test, y_test

    def _model(self, device: str = "cpu") -> torch.nn.Module:
        return load_lenet5_checkpoint(
            self.config["experiment"]["checkpoint"],
            device=device,
        )

    @staticmethod
    def _torch_logits_function(
        model: torch.nn.Module,
        *,
        device: str,
        batch_size: int = 256,
    ) -> Callable[[np.ndarray], np.ndarray]:
        device_object = torch.device(device)

        @torch.no_grad()
        def logits(x: np.ndarray) -> np.ndarray:
            values = np.asarray(x, dtype=np.float32).reshape(-1, 1, 28, 28)
            parts: list[np.ndarray] = []
            for start in range(0, len(values), int(batch_size)):
                tensor = torch.from_numpy(values[start : start + batch_size]).to(
                    device_object
                )
                parts.append(model(tensor).detach().cpu().numpy())
            return np.concatenate(parts, axis=0)

        return logits

    def _manifest(self) -> dict[str, Any]:
        if not self.paths.manifest.exists():
            raise RuntimeError("prepare stage has not completed")
        manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        checkpoint = Path(self.config["experiment"]["checkpoint"])
        if manifest.get("config_fingerprint") != self.fingerprint:
            raise RuntimeError(
                "prepared artifacts belong to a different configuration; "
                "rerun prepare --force"
            )
        if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError("the classifier checkpoint changed; rerun prepare --force")
        if not self.paths.onnx.exists():
            raise RuntimeError("prepared ONNX model is missing")
        if manifest.get("onnx_sha256") != sha256_file(self.paths.onnx):
            raise RuntimeError("prepared ONNX model hash mismatch; rerun prepare --force")
        return manifest

    def _artifact_valid(self, path: Path, *, stage: str) -> bool:
        if not path.exists():
            return False
        try:
            metadata = _read_npz_metadata(path)
        except Exception:
            return False
        return bool(
            metadata.get("stage") == stage
            and metadata.get("run_fingerprint") == self._manifest()["run_fingerprint"]
            and metadata.get("status") == "complete"
        )

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        checkpoint = Path(self.config["experiment"]["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        if self.paths.manifest.exists() and not force:
            try:
                manifest = self._manifest()
                _log("[PREPARE] Artifact validi già presenti; nessuna rigenerazione.")
                return manifest
            except RuntimeError:
                pass

        _log("[PREPARE] Caricamento MNIST e checkpoint LeNet5.")
        x_train, y_train, x_test, y_test = self._load_mnist()
        original = self._model("cpu")
        rewritten = MarabouCompatibleLeNet5(original)
        export_config = self.config["export"]

        temporary_onnx = self.paths.onnx.with_name(
            f".{self.paths.onnx.name}.tmp-{os.getpid()}"
        )
        temporary_onnx.parent.mkdir(parents=True, exist_ok=True)
        export_lenet5_onnx(
            rewritten,
            temporary_onnx,
            opset_version=int(export_config["opset_version"]),
        )
        os.replace(temporary_onnx, self.paths.onnx)
        operators = onnx_operator_types(self.paths.onnx)
        forbidden = {"AveragePool", "MaxPool", "BatchNormalization"}
        if forbidden.intersection(operators):
            raise RuntimeError(f"unsupported ONNX operators: {forbidden.intersection(operators)}")
        parse_report = validate_marabou_parse(self.paths.onnx)
        if parse_report != {"input_variables": 784, "output_variables": 10}:
            raise RuntimeError(f"unexpected Marabou parse report: {parse_report}")

        from torch.utils.data import TensorDataset

        validation_size = len(x_test) if export_config["validate_full_test_set"] else min(
            int(export_config["validation_batch_size"]), len(x_test)
        )
        validation_dataset = TensorDataset(
            torch.from_numpy(x_test[:validation_size].reshape(-1, 1, 28, 28)),
            torch.from_numpy(y_test[:validation_size]),
        )
        report = validate_lenet5_equivalence(
            original,
            rewritten,
            self.paths.onnx,
            DataLoader(
                validation_dataset,
                batch_size=int(export_config["validation_batch_size"]),
                shuffle=False,
            ),
            atol=float(export_config["atol"]),
            rtol=float(export_config["rtol"]),
        )
        if not report["rewritten_allclose"] or not report["onnx_allclose"]:
            raise RuntimeError(f"LeNet5 export is not equivalent: {report}")
        _atomic_json(
            {**report, "operators": operators, "marabou": parse_report},
            self.paths.export_validation,
        )

        benchmark_indices = query_indices(self.config)
        pilot_indices = query_indices(self.config, pilot=True)
        selected_indices = np.unique(np.concatenate([benchmark_indices, pilot_indices]))
        _atomic_npz(
            self.paths.queries,
            indices=selected_indices,
            x=x_test[selected_indices].astype(np.float32),
            y_true=y_test[selected_indices].astype(np.int64),
            benchmark_indices=benchmark_indices,
            pilot_indices=pilot_indices,
        )

        train_indices = balanced_class_indices(
            y_train,
            int(self.config["certcf"]["train_per_true_class"]),
            seed=int(self.config["experiment"]["seed"]),
        )
        _atomic_npz(
            self.paths.train_pool,
            indices=train_indices,
            x=x_train[train_indices].astype(np.float32),
            y_true=y_train[train_indices].astype(np.int64),
        )

        checkpoint_hash = sha256_file(checkpoint)
        onnx_hash = sha256_file(self.paths.onnx)
        prepared_at_ns = time.time_ns()
        run_payload = (
            f"{self.fingerprint}:{checkpoint_hash}:{onnx_hash}:"
            f"{sha256_file(self.paths.queries)}:{sha256_file(self.paths.train_pool)}:"
            f"{prepared_at_ns}"
        ).encode()
        manifest = {
            "status": "complete",
            "config_fingerprint": self.fingerprint,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "onnx_sha256": onnx_hash,
            "queries_sha256": sha256_file(self.paths.queries),
            "train_pool_sha256": sha256_file(self.paths.train_pool),
            "run_fingerprint": hashlib.sha256(run_payload).hexdigest(),
            "prepared_at_ns": prepared_at_ns,
            "benchmark_query_count": int(len(benchmark_indices)),
            "pilot_indices": pilot_indices.tolist(),
            "certcf_train_pool_rows": int(len(train_indices)),
            "certcf_train_pool_true_class_counts": {
                str(label): int(np.sum(y_train[train_indices] == label))
                for label in sorted(np.unique(y_train).tolist())
            },
            "export_validation": report,
        }
        _atomic_json(manifest, self.paths.manifest)
        _log(
            "[PREPARE] Complete: equivalent export, "
            f"{len(benchmark_indices)} query e {len(train_indices)} punti atlas."
        )
        return manifest

    def _query_data(self, index: int) -> tuple[np.ndarray, int]:
        with np.load(self.paths.queries, allow_pickle=False) as data:
            positions = np.flatnonzero(data["indices"] == int(index))
            if len(positions) != 1:
                raise KeyError(f"query index {index} is not prepared")
            position = int(positions[0])
            return data["x"][position].copy(), int(data["y_true"][position])

    def run_verix(
        self,
        indices: Sequence[int] | None = None,
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        from verix.backends.marabou import (
            MarabouClassificationChecker,
            MarabouOptions,
        )

        manifest = self._manifest()
        selected = query_indices(self.config) if indices is None else np.asarray(indices)
        config = self.config["verix"]
        scorer = ONNXMNISTScorer(str(self.paths.onnx))
        checker = MarabouClassificationChecker(
            self.paths.onnx,
            input_lower_bounds=float(config["input_bounds"][0]),
            input_upper_bounds=float(config["input_bounds"][1]),
            options=MarabouOptions(
                num_workers=int(config["num_workers"]),
                timeout_seconds=int(config["timeout_seconds"]),
                solve_with_milp=bool(config["solve_with_milp"]),
                classification_margin=float(config["classification_margin"]),
            ),
        )
        canonical_model = self._model("cpu")
        canonical_logits = self._torch_logits_function(canonical_model, device="cpu")
        summaries: list[dict[str, Any]] = []

        for position, index_value in enumerate(selected, start=1):
            index = int(index_value)
            path = self.paths.verix_query(index)
            if not force and self._artifact_valid(path, stage="verix"):
                metadata = _read_npz_metadata(path)
                summaries.append(metadata)
                _log(
                    f"[VERIX {position}/{len(selected)}] query={index}: "
                    "artifact valido, salto."
                )
                continue
            x_query, y_true = self._query_data(index)
            _log(
                f"[VERIX {position}/{len(selected)}] query={index}: "
                f"inizio traversata di {MNIST_PIXELS} pixel."
            )
            progress = tqdm(
                total=MNIST_PIXELS,
                desc=f"  VERIX q={index}",
                unit="pixel",
                leave=False,
            )
            interval = float(self.config["resources"]["rss_sample_interval_seconds"])
            try:
                with ResourceMonitor("query", "cpu", interval) as monitor:
                    result, sensitivity = run_mnist_verix(
                        checker,
                        scorer,
                        x_query,
                        epsilon=float(config["epsilon"]),
                        step_callback=lambda step: progress.update(1),
                    )
                    validations = validate_mnist_witnesses(
                        result,
                        x_query,
                        logits_function=canonical_logits,
                        epsilon=float(config["epsilon"]),
                        bounds=tuple(map(float, config["input_bounds"])),
                    )
                    selected_witness = select_nearest_valid_witness(
                        result.counterfactuals,
                        validations,
                    )
                witnesses = (
                    np.stack([item.x for item in result.counterfactuals]).astype(np.float32)
                    if result.counterfactuals
                    else np.empty((0, MNIST_PIXELS), dtype=np.float32)
                )
                metadata = {
                    "status": "complete",
                    "stage": "verix",
                    "run_fingerprint": manifest["run_fingerprint"],
                    "query_index": index,
                    "true_class": y_true,
                    "source_class": int(result.reference_output),
                    "paired_eligible": selected_witness is not None,
                    "target_class": (
                        None
                        if selected_witness is None
                        else int(selected_witness.target_class)
                    ),
                    "selected_witness_index": (
                        None
                        if selected_witness is None
                        else int(selected_witness.witness_index)
                    ),
                    "selected_feature": (
                        None
                        if selected_witness is None
                        else int(selected_witness.feature)
                    ),
                    "runtime_seconds": monitor.metrics["query_wall_time_seconds"],
                    "explanation_size": len(result.explanation),
                    "irrelevant_size": len(result.irrelevant),
                    "unknown_size": len(result.unknown),
                    "native_witness_count": len(result.counterfactuals),
                    "canonically_valid_witness_count": int(
                        sum(item.canonical_valid for item in validations)
                    ),
                    "locally_minimal": result.locally_minimal,
                    "checker_is_complete": result.checker_is_complete,
                    "validations": [asdict(item) for item in validations],
                    "steps": [
                        {
                            "feature": int(step.feature),
                            "status": step.status.value,
                            "elapsed_seconds": float(step.elapsed_seconds),
                            "metadata": _json_value(step.metadata),
                        }
                        for step in result.steps
                    ],
                    **monitor.metrics,
                }
                _atomic_npz(
                    path,
                    metadata_json=np.asarray(json.dumps(_json_value(metadata))),
                    query=x_query.astype(np.float32),
                    sensitivity=np.asarray(sensitivity, dtype=np.float64),
                    traversal_order=np.asarray(result.traversal_order, dtype=np.int64),
                    explanation=np.asarray(result.explanation, dtype=np.int64),
                    irrelevant=np.asarray(result.irrelevant, dtype=np.int64),
                    unknown=np.asarray(result.unknown, dtype=np.int64),
                    witnesses=witnesses,
                    witness_features=np.asarray(
                        [item.feature for item in result.counterfactuals],
                        dtype=np.int64,
                    ),
                    witness_outputs=np.asarray(
                        [
                            -1 if item.output is None else int(item.output)
                            for item in result.counterfactuals
                        ],
                        dtype=np.int64,
                    ),
                    selected_counterfactual=(
                        np.empty((0,), dtype=np.float32)
                        if selected_witness is None
                        else selected_witness.x.astype(np.float32)
                    ),
                )
                summaries.append(metadata)
                _log(
                    f"[VERIX {position}/{len(selected)}] query={index}: "
                    f"target={metadata['target_class']}, "
                    f"witness validi={metadata['canonically_valid_witness_count']}."
                )
            finally:
                progress.close()
        return summaries

    def _build_certcf(self) -> tuple[CertCF, torch.nn.Module, dict[str, Any]]:
        from certcf import NearestOppositeClassClearanceStrategy
        from counterfactuals.methods.certcf import CertCF
        from counterfactuals.models.torch_model import TorchModelWrapper

        manifest = self._manifest()
        config = self.config["certcf"]
        device = _normalize_device(config["device"])
        model = self._model(device)
        with np.load(self.paths.train_pool, allow_pickle=False) as data:
            x_train = data["x"].astype(np.float32)
            y_true = data["y_true"].astype(np.int64)
        logits = self._torch_logits_function(model, device=device)
        y_pred = logits(x_train).argmax(axis=1).astype(np.int64)
        agreement = float(np.mean(y_pred == y_true))
        wrapper = TorchModelWrapper(model=model, device=device)
        method = CertCF(
            model=wrapper,
            norm=_as_norm(config["norm"]),
            distance_norm=_as_norm(config["distance_norm"]),
            lirpa_method=str(config["lirpa_method"]),
            eps_strategy=NearestOppositeClassClearanceStrategy(
                alpha=float(config["eps_alpha"])
            ),
            batch_size=int(config["lirpa_batch_size"]),
            cnn=True,
            default_query_method=str(config["query_method"]),
            query_k_candidates=int(config["query_k_candidates"]),
            solver_maxiter=int(config["solver_maxiter"]),
            query_parallelism=int(config["query_parallelism"]),
            cvxpy_solvers=list(config["cvxpy_solvers"]),
            classification_margin=float(config["classification_margin"]),
            adaptive_eps=bool(config["adaptive_eps"]),
            adaptive_eps_shrink_factor=float(config["adaptive_eps_shrink_factor"]),
            adaptive_eps_max_shrinks=int(config["adaptive_eps_max_shrinks"]),
            adaptive_eps_min=float(config["adaptive_eps_min"]),
            adaptive_eps_center_tol=float(config["adaptive_eps_center_tol"]),
            adaptive_eps_binary_search_steps=int(
                config["adaptive_eps_binary_search_steps"]
            ),
            input_bounds=list(config["input_bounds"]),
            sparsity_penalty=str(config["sparsity_penalty"]),
            k_per_class=int(config["anchors_per_predicted_class"]),
            subsample_method=str(config["atlas_subsample_method"]),
            random_seed=int(self.config["experiment"]["seed"]),
        )
        interval = float(self.config["resources"]["rss_sample_interval_seconds"])
        _log(
            "[CERTCF BUILD] Costruzione atlas: "
            f"{len(x_train)} supporti, device={device}."
        )
        with ResourceMonitor("build", device, interval) as monitor:
            method.fit(x_train=x_train, y_train=y_pred)
        atlas = method.atlas
        if atlas is None or atlas.bounds is None:
            raise RuntimeError("CertCF did not build an atlas")
        eps_initial = np.concatenate(
            [atlas.bounds[label]["eps_initial"] for label in atlas.class_labels]
        )
        eps_final = np.concatenate(
            [atlas.bounds[label]["eps"] for label in atlas.class_labels]
        )
        shrinks = np.concatenate(
            [
                atlas.bounds[label]["adaptive_eps_n_shrinks"]
                for label in atlas.class_labels
            ]
        )
        certified = np.concatenate(
            [
                atlas.bounds[label]["adaptive_eps_center_certified"]
                for label in atlas.class_labels
            ]
        )
        build_metadata = {
            "status": "complete",
            "stage": "certcf_build",
            "run_fingerprint": manifest["run_fingerprint"],
            "device": device,
            "train_rows": len(x_train),
            "train_prediction_agreement": agreement,
            "predicted_class_counts": {
                str(label): int(np.sum(y_pred == label))
                for label in sorted(np.unique(y_pred).tolist())
            },
            "atlas_region_count": int(sum(len(atlas.bounds[label]["X"]) for label in atlas.class_labels)),
            "eps_initial_median": float(np.median(eps_initial)),
            "eps_final_median": float(np.median(eps_final)),
            "eps_ratio_median": float(
                np.median(
                    np.divide(
                        eps_final,
                        eps_initial,
                        out=np.zeros_like(eps_final),
                        where=eps_initial > 0,
                    )
                )
            ),
            "adaptive_shrinks_mean": float(np.mean(shrinks)),
            "center_certified_fraction": float(np.mean(certified)),
            **monitor.metrics,
        }
        _atomic_json(build_metadata, self.paths.certcf_build)
        _log(
            "[CERTCF BUILD] Complete: "
            f"{build_metadata['atlas_region_count']} regions in "
            f"{build_metadata['build_wall_time_seconds']:.1f}s."
        )
        return method, model, build_metadata

    def run_certcf(
        self,
        indices: Sequence[int] | None = None,
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        manifest = self._manifest()
        selected = query_indices(self.config) if indices is None else np.asarray(indices)
        missing_verix = [
            int(index)
            for index in selected
            if not self._artifact_valid(self.paths.verix_query(int(index)), stage="verix")
        ]
        if missing_verix:
            raise RuntimeError(
                f"VERIX must complete first; missing query artifacts: {missing_verix[:10]}"
            )
        pending = [
            int(index)
            for index in selected
            if force
            or not self._artifact_valid(self.paths.certcf_query(int(index)), stage="certcf")
        ]
        if not pending:
            _log("[CERTCF] Tutte le query richieste hanno artifact validi.")
            return [_read_npz_metadata(self.paths.certcf_query(int(i))) for i in selected]

        method, model, _ = self._build_certcf()
        device = _normalize_device(self.config["certcf"]["device"])
        logits = self._torch_logits_function(model, device=device)
        summaries: list[dict[str, Any]] = []
        timeout = float(self.config["certcf"]["timeout_seconds_per_query"])
        interval = float(self.config["resources"]["rss_sample_interval_seconds"])

        for position, index in enumerate(selected, start=1):
            path = self.paths.certcf_query(int(index))
            if not force and self._artifact_valid(path, stage="certcf"):
                summaries.append(_read_npz_metadata(path))
                _log(
                    f"[CERTCF {position}/{len(selected)}] query={index}: "
                    "artifact valido, salto."
                )
                continue
            x_query, y_true = self._query_data(int(index))
            verix_metadata = _read_npz_metadata(self.paths.verix_query(int(index)))
            target = verix_metadata.get("target_class")
            if target is None:
                metadata = {
                    "status": "complete",
                    "stage": "certcf",
                    "run_fingerprint": manifest["run_fingerprint"],
                    "query_index": int(index),
                    "true_class": y_true,
                    "source_class": verix_metadata["source_class"],
                    "target_class": None,
                    "paired_eligible": False,
                    "method_success": False,
                    "failure_reason": "verix_has_no_canonically_valid_witness",
                    "runtime_seconds": 0.0,
                }
                _atomic_npz(
                    path,
                    metadata_json=np.asarray(json.dumps(metadata)),
                    query=x_query.astype(np.float32),
                    counterfactual=np.empty((0,), dtype=np.float32),
                )
                summaries.append(metadata)
                continue
            _log(
                f"[CERTCF {position}/{len(selected)}] query={index}: "
                f"target VERIX={target}."
            )
            result = None
            generation_error: str | None = None
            try:
                with ResourceMonitor("query", device, interval) as monitor:
                    result = method.generate_batch(
                        x=x_query[None, :],
                        target_class=np.asarray([int(target)]),
                        timeout_s_per_query=timeout,
                    )[0]
            except Exception as exc:
                generation_error = f"{type(exc).__name__}: {exc}"
            x_cf = (
                None
                if result is None or result.x_cf is None
                else np.asarray(result.x_cf, dtype=np.float32)
            )
            metrics = counterfactual_metrics(
                x_query,
                x_cf,
                target=int(target),
                logits_function=logits,
            )
            metadata = {
                "status": "complete",
                "stage": "certcf",
                "run_fingerprint": manifest["run_fingerprint"],
                "query_index": int(index),
                "true_class": y_true,
                "source_class": verix_metadata["source_class"],
                "target_class": int(target),
                "paired_eligible": True,
                "method_success": bool(result is not None and result.success),
                "benchmark_success": bool(
                    result is not None and result.success and metrics["success"]
                ),
                "failure_reason": (
                    None
                    if result is not None and result.success
                    else (
                        generation_error
                        if generation_error is not None
                        else str(result.metadata.get("reason", "no_counterfactual"))
                    )
                ),
                "runtime_seconds": monitor.metrics["query_wall_time_seconds"],
                "metrics": metrics,
                "method_metadata": (
                    {} if result is None else _json_value(result.metadata)
                ),
                **monitor.metrics,
            }
            _atomic_npz(
                path,
                metadata_json=np.asarray(json.dumps(_json_value(metadata))),
                query=x_query.astype(np.float32),
                counterfactual=(
                    np.empty((0,), dtype=np.float32) if x_cf is None else x_cf
                ),
            )
            summaries.append(metadata)
        return summaries

    def pilot(self, *, force: bool = False) -> dict[str, Any]:
        indices = query_indices(self.config, pilot=True)
        self.run_verix(indices, force=force)
        self.run_certcf(indices, force=force)
        return {
            "pilot_indices": indices.tolist(),
            "complete": True,
        }

    def _method_rows(self, indices: Iterable[int]) -> list[dict[str, Any]]:
        model = self._model("cpu")
        logits = self._torch_logits_function(model, device="cpu")
        rows: list[dict[str, Any]] = []
        for index_value in indices:
            index = int(index_value)
            x_query, y_true = self._query_data(index)
            verix_path = self.paths.verix_query(index)
            certcf_path = self.paths.certcf_query(index)
            if not self._artifact_valid(verix_path, stage="verix"):
                raise RuntimeError(f"missing valid VERIX artifact for query {index}")
            if not self._artifact_valid(certcf_path, stage="certcf"):
                raise RuntimeError(f"missing valid CertCF artifact for query {index}")
            with np.load(verix_path, allow_pickle=False) as data:
                verix_metadata = json.loads(str(data["metadata_json"].item()))
                selected = data["selected_counterfactual"]
                verix_cf = None if selected.size == 0 else selected
            target = verix_metadata.get("target_class")
            verix_metrics = counterfactual_metrics(
                x_query,
                verix_cf,
                target=target,
                logits_function=logits,
            )
            rows.append(
                {
                    "query_index": index,
                    "method": "verix",
                    "true_class": y_true,
                    "source_class": verix_metadata["source_class"],
                    "target_class": target,
                    "paired_eligible": verix_metadata["paired_eligible"],
                    "runtime_seconds": verix_metadata["runtime_seconds"],
                    "success": verix_metrics["success"],
                    **{
                        key: value
                        for key, value in verix_metrics.items()
                        if key != "success"
                    },
                    "counterfactual": verix_cf,
                }
            )
            with np.load(certcf_path, allow_pickle=False) as data:
                certcf_metadata = json.loads(str(data["metadata_json"].item()))
                candidate = data["counterfactual"]
                certcf_cf = None if candidate.size == 0 else candidate
            certcf_metrics = counterfactual_metrics(
                x_query,
                certcf_cf,
                target=target,
                logits_function=logits,
            )
            rows.append(
                {
                    "query_index": index,
                    "method": "certcf",
                    "true_class": y_true,
                    "source_class": verix_metadata["source_class"],
                    "target_class": target,
                    "paired_eligible": verix_metadata["paired_eligible"],
                    "runtime_seconds": certcf_metadata["runtime_seconds"],
                    "success": bool(
                        certcf_metadata.get("method_success", False)
                        and certcf_metrics["success"]
                    ),
                    **{
                        key: value
                        for key, value in certcf_metrics.items()
                        if key != "success"
                    },
                    "counterfactual": certcf_cf,
                }
            )
        return rows

    def robustness(self, *, force: bool = False) -> pd.DataFrame:
        self._manifest()
        if self.paths.robustness.exists() and not force:
            _log("[ROBUSTNESS] Risultato già presente; uso la cache.")
            return pd.read_parquet(self.paths.robustness)
        indices = query_indices(self.config)
        method_rows = self._method_rows(indices)
        model = self._model(_normalize_device(self.config["certcf"]["device"]))
        device = _normalize_device(self.config["certcf"]["device"])
        logits = self._torch_logits_function(model, device=device)
        empirical = self.config["robustness"]["empirical"]
        certified_config = self.config["robustness"]["certified"]
        certifier = (
            PostHocLiRPACertifier(
                model,
                device=device,
                method=str(certified_config["lirpa_method"]),
            )
            if certified_config["enabled"]
            else None
        )
        seed = int(self.config["experiment"]["seed"])
        output: list[dict[str, Any]] = []
        by_query: dict[int, list[dict[str, Any]]] = {}
        for row in method_rows:
            by_query.setdefault(int(row["query_index"]), []).append(row)

        for position, index in enumerate(indices, start=1):
            rows = by_query[int(index)]
            _log(f"[ROBUSTNESS {position}/{len(indices)}] query={int(index)}.")
            common_noise: dict[float, np.ndarray] = {}
            for radius in empirical["radii"]:
                radius_value = float(radius)
                rng = np.random.default_rng(
                    np.random.SeedSequence([seed, int(index), int(round(radius_value * 1e9))])
                )
                common_noise[radius_value] = rng.uniform(
                    -radius_value,
                    radius_value,
                    size=(int(empirical["samples_per_radius"]), MNIST_PIXELS),
                ).astype(np.float32)
            for row in rows:
                x_cf = row.pop("counterfactual")
                base = {
                    "query_index": int(index),
                    "method": row["method"],
                    "target_class": row["target_class"],
                    "success": bool(row["success"]),
                }
                certified_radius = np.nan
                if x_cf is not None and row["success"] and certifier is not None:
                    certified_radius = certifier.radius(
                        x_cf,
                        int(row["target_class"]),
                        maximum=float(certified_config["max_radius"]),
                        steps=int(certified_config["binary_search_steps"]),
                    )
                for radius in empirical["radii"]:
                    radius_value = float(radius)
                    empirical_rate = np.nan
                    if x_cf is not None and row["success"] and empirical["enabled"]:
                        perturbed = np.clip(
                            np.asarray(x_cf)[None, :] + common_noise[radius_value],
                            0.0,
                            1.0,
                        )
                        empirical_rate = float(
                            np.mean(logits(perturbed).argmax(axis=1) == int(row["target_class"]))
                        )
                    output.append(
                        {
                            **base,
                            "empirical_radius": radius_value,
                            "empirical_target_rate": empirical_rate,
                            "certified_linf_radius": certified_radius,
                        }
                    )
        frame = pd.DataFrame(output)
        _atomic_parquet(frame, self.paths.robustness)
        return frame

    def analyze(self, *, require_complete: bool = True) -> pd.DataFrame:
        indices = query_indices(self.config)
        if require_complete:
            rows = self._method_rows(indices)
        else:
            complete = [
                int(index)
                for index in indices
                if self._artifact_valid(self.paths.verix_query(int(index)), stage="verix")
                and self._artifact_valid(self.paths.certcf_query(int(index)), stage="certcf")
            ]
            rows = self._method_rows(complete)
        flat_rows: list[dict[str, Any]] = []
        for row in rows:
            copy = dict(row)
            x_cf = copy.pop("counterfactual")
            copy["counterfactual_json"] = (
                None if x_cf is None else json.dumps(np.asarray(x_cf).tolist())
            )
            flat_rows.append(copy)
        frame = pd.DataFrame(flat_rows).sort_values(["query_index", "method"])
        _atomic_parquet(frame, self.paths.combined)
        eligible = frame[frame["paired_eligible"]]
        summary = {
            "status": "complete",
            "query_count": int(frame["query_index"].nunique()) if len(frame) else 0,
            "row_count": int(len(frame)),
            "paired_eligible_queries": int(eligible["query_index"].nunique()) if len(eligible) else 0,
            "methods": {
                method: {
                    "rows": int(len(group)),
                    "success_rate": float(group["success"].mean()),
                    "mean_l1": float(group.loc[group["success"], "l1_distance"].mean()),
                    "median_l1": float(group.loc[group["success"], "l1_distance"].median()),
                    "mean_l0": float(group.loc[group["success"], "l0_changed"].mean()),
                    "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
                }
                for method, group in frame.groupby("method")
            },
        }
        if {"certcf", "verix"}.issubset(set(frame["method"])):
            paired = frame.pivot(index="query_index", columns="method", values="l1_distance")
            ratios = paired["certcf"] / paired["verix"]
            summary["certcf_over_verix_l1_ratio"] = {
                "mean": float(ratios.replace([np.inf, -np.inf], np.nan).mean()),
                "median": float(ratios.replace([np.inf, -np.inf], np.nan).median()),
            }
        if self.paths.robustness.exists():
            robustness = pd.read_parquet(self.paths.robustness)
            summary["robustness"] = {
                method: {
                    "median_certified_linf_radius": float(
                        group.groupby("query_index")["certified_linf_radius"]
                        .first()
                        .median()
                    ),
                    "empirical_target_rate_by_radius": {
                        str(radius): float(radius_group["empirical_target_rate"].mean())
                        for radius, radius_group in group.groupby("empirical_radius")
                    },
                }
                for method, group in robustness.groupby("method")
            }
        _atomic_json(summary, self.paths.summary)
        _log(
            f"[ANALYZE] Saved {len(frame)} rows for "
            f"{summary['query_count']} query."
        )
        return frame

    def status(self) -> dict[str, Any]:
        prepared = False
        try:
            self._manifest()
            prepared = True
        except RuntimeError:
            pass
        benchmark = query_indices(self.config)
        pilots = query_indices(self.config, pilot=True)

        def count(paths: Iterable[Path], stage: str) -> int:
            if not prepared:
                return 0
            return sum(self._artifact_valid(path, stage=stage) for path in paths)

        return {
            "prepared": prepared,
            "run_fingerprint": (
                self._manifest()["run_fingerprint"] if prepared else None
            ),
            "benchmark_queries": int(len(benchmark)),
            "pilot_queries": pilots.tolist(),
            "verix_complete": count(
                (self.paths.verix_query(int(index)) for index in benchmark),
                "verix",
            ),
            "certcf_complete": count(
                (self.paths.certcf_query(int(index)) for index in benchmark),
                "certcf",
            ),
            "combined_exists": self.paths.combined.exists(),
            "robustness_exists": self.paths.robustness.exists(),
        }

    def all(self, *, force: bool = False) -> pd.DataFrame:
        self.prepare(force=force)
        # Establish that the official VERIX query is tractable, then finish
        # VERIX before constructing the expensive CertCF atlas exactly once.
        self.run_verix(query_indices(self.config, pilot=True), force=force)
        self.run_verix(force=False)
        self.run_certcf(force=False)
        self.robustness(force=force)
        return self.analyze()
