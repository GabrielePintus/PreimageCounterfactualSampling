"""Resumable CertCF network-complexity scaling experiment.

The public entry point is :class:`NetworkComplexityRunner`; the thin CLI wrapper
is ``scripts/network_complexity_grid.py``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import threading
import time
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import lightning as L
import numpy as np
import pandas as pd
import psutil
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from certcf import NearestOppositeClassClearanceStrategy
from counterfactuals.methods.certcf import CertCF
from counterfactuals.models.torch_model import TorchModelWrapper
from dataset_specs import get_tabular_dataset_spec
from models.classifiers import TabularClassifier
from training.datamodules.network_complexity import (
    NetworkComplexityDataModule,
    prepare_network_complexity_dataset,
    read_dataset_metadata,
)
from training.lit_classifier import LitClassifier


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "widths": [16, 32, 64, 128, 256],
        "depths": [1, 2, 3, 4, 5],
        "input_dim": 32,
        "num_classes": 2,
        "accuracy_gate": 0.90,
    },
    "dataset": {
        "n_train": 10_000,
        "n_validation": 1_000,
        "n_test": 1_000,
        "class_sep": 1.5,
        "flip_y": 0.01,
        "n_queries": 1_000,
        "queries_per_class": 500,
    },
    "training": {
        "max_epochs": 50,
        "batch_size": 128,
        "num_workers": 0,
        "dropout": 0.1,
        "initial_lr": 5.0e-3,
        "final_lr": 1.0e-6,
        "weight_decay": 1.0e-5,
        "accelerator": "gpu",
        "devices": 1,
        "deterministic": True,
    },
    "certcf": {
        "device": "auto",
        "train_pool_per_true_class": 5_000,
        "anchors_per_predicted_class": 500,
        "atlas_subsample_method": "random",
        "eps_alpha": 0.20,
        "norm": 1,
        "distance_norm": 1,
        "lirpa_method": "backward",
        "lirpa_batch_size": 128,
        "lirpa_batch_size_candidates": [128, 64, 32, 16, 8],
        "classification_margin": 1.0e-4,
        "adaptive_eps": True,
        "adaptive_eps_shrink_factor": 0.5,
        "adaptive_eps_max_shrinks": 8,
        "adaptive_eps_min": 1.0e-6,
        "adaptive_eps_center_tol": 1.0e-6,
        "adaptive_eps_binary_search_steps": 0,
        "sparsity_penalty": "reweighted_l1",
        "sparsity_lambda": 1.0,
        "sparsity_reweight_iters": 2,
        "sparsity_eps": 1.0e-3,
        "query_method": "nearest_anchor",
        "query_k_candidates": 3,
        "query_parallelism": 1,
        "timeout_s_per_query": 120,
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
    "resources": {"rss_sample_interval_s": 0.05},
    "artifacts": {"output_dir": "results/network_complexity"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    exp = cfg["experiment"]
    widths = [int(value) for value in exp["widths"]]
    depths = [int(value) for value in exp["depths"]]
    if not widths or not depths or any(value <= 0 for value in widths + depths):
        raise ValueError("widths and depths must be non-empty positive integer lists")
    if len(set(widths)) != len(widths) or len(set(depths)) != len(depths):
        raise ValueError("widths and depths must not contain duplicates")
    if int(exp["input_dim"]) != 32:
        raise ValueError("The official network-complexity experiment fixes input_dim=32")
    if int(exp["num_classes"]) != 2:
        raise ValueError("The official network-complexity experiment is binary")
    if not 0.0 <= float(exp["accuracy_gate"]) <= 1.0:
        raise ValueError("accuracy_gate must lie in [0, 1]")
    dataset_cfg = cfg["dataset"]
    for split_name in ("n_train", "n_validation", "n_test"):
        split_size = int(dataset_cfg[split_name])
        if split_size <= 0 or split_size % 2:
            raise ValueError(f"dataset.{split_name} must be a positive even integer")
    if int(dataset_cfg["n_queries"]) != 2 * int(dataset_cfg["queries_per_class"]):
        raise ValueError("n_queries must equal 2 * queries_per_class")
    if int(dataset_cfg["queries_per_class"]) > int(dataset_cfg["n_test"]) // 2:
        raise ValueError("queries_per_class exceeds the per-class test size")
    candidates = [int(value) for value in cfg["certcf"]["lirpa_batch_size_candidates"]]
    if int(cfg["certcf"]["lirpa_batch_size"]) not in candidates:
        raise ValueError("lirpa_batch_size must be included in lirpa_batch_size_candidates")
    if any(value <= 0 for value in candidates):
        raise ValueError("LiRPA batch-size candidates must be positive")
    if int(cfg["certcf"].get("epsilon_parallelism", 1)) <= 0:
        raise ValueError("epsilon_parallelism must be positive")
    if int(cfg["certcf"].get("build_parallelism", 1)) <= 0:
        raise ValueError("build_parallelism must be positive")


def config_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint all protocol choices while excluding only the artifact location."""
    protocol = deepcopy(cfg)
    protocol.pop("artifacts", None)
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_config_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint only inputs that can change trained classifier artifacts."""
    experiment = cfg["experiment"]
    payload = {
        "experiment": {
            key: deepcopy(experiment[key])
            for key in ("seed", "widths", "depths", "input_dim", "num_classes")
        },
        "dataset": deepcopy(cfg["dataset"]),
        "training": deepcopy(cfg["training"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_config_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint only inputs that can change CertCF benchmark artifacts."""
    experiment = cfg["experiment"]
    payload = {
        "experiment": {
            key: deepcopy(experiment[key])
            for key in ("seed", "widths", "depths", "input_dim", "num_classes")
        },
        "dataset": deepcopy(cfg["dataset"]),
        "certcf": deepcopy(cfg["certcf"]),
        "resources": deepcopy(cfg["resources"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def architecture_id(depth: int, width: int) -> str:
    return f"depth_{int(depth):02d}_width_{int(width):03d}"


def architecture_grid(cfg: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(depth), int(width))
        for depth in cfg["experiment"]["depths"]
        for width in cfg["experiment"]["widths"]
    ]


def parameter_count(input_dim: int, hidden_dims: Iterable[int], num_classes: int) -> int:
    dims = [int(input_dim), *[int(value) for value in hidden_dims], int(num_classes)]
    return int(sum((left + 1) * right for left, right in zip(dims[:-1], dims[1:])))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _normalize_device(raw_device: Any) -> str:
    raw = str(raw_device or "cpu").strip().lower()
    if raw in {"auto", "gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if raw.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return raw


def _log(message: str) -> None:
    """Print a timestamped pipeline progress message immediately."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError)) or (
        "out of memory" in message
        or "cannot allocate memory" in message
        or "can't allocate memory" in message
    )


class PhaseResourceMonitor(AbstractContextManager):
    """Collect phase-specific wall time, process RSS, and CUDA peak counters."""

    def __init__(self, phase: str, device: str, interval_s: float = 0.05):
        self.phase = str(phase)
        self.device = torch.device(device)
        self.interval_s = max(float(interval_s), 0.005)
        self.process = psutil.Process(os.getpid())
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.samples.append(int(self.process.memory_info().rss))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sample()
        self.rss_baseline = self.samples[0]
        self.started = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_s = time.perf_counter() - self.started
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, 2 * self.interval_s))
        self._sample()
        peak = max(self.samples)
        prefix = self.phase
        self.metrics = {
            f"{prefix}_wall_time_s": float(self.elapsed_s),
            f"{prefix}_rss_baseline_bytes": int(self.rss_baseline),
            f"{prefix}_rss_peak_bytes": int(peak),
            f"{prefix}_rss_peak_delta_bytes": int(max(0, peak - self.rss_baseline)),
            f"{prefix}_rss_sample_count": int(len(self.samples)),
            f"{prefix}_cuda_peak_allocated_bytes": 0,
            f"{prefix}_cuda_peak_reserved_bytes": 0,
        }
        if self.device.type == "cuda":
            self.metrics[f"{prefix}_cuda_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            self.metrics[f"{prefix}_cuda_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(self.device)
            )
        return False


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def dataset(self) -> Path:
        return self.root / "data" / "dataset.npz"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def combined(self) -> Path:
        return self.root / "network_complexity_queries.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "network_complexity_summary.parquet"

    def checkpoint_dir(self, depth: int, width: int) -> Path:
        return self.root / "checkpoints" / architecture_id(depth, width)

    def checkpoint(self, depth: int, width: int) -> Path:
        return self.checkpoint_dir(depth, width) / "best.ckpt"

    def train_metadata(self, depth: int, width: int) -> Path:
        return self.checkpoint_dir(depth, width) / "training.json"

    def benchmark(self, depth: int, width: int) -> Path:
        return self.root / "benchmarks" / f"{architecture_id(depth, width)}.parquet"

    def benchmark_metadata(self, depth: int, width: int) -> Path:
        return self.root / "benchmarks" / f"{architecture_id(depth, width)}.json"


class AccuracyGateError(RuntimeError):
    """Raised when a common training protocol fails the all-architecture gate."""


class NetworkComplexityRunner:
    """Run preparation, training, benchmarking, validation, and analysis stages."""

    def __init__(self, config: dict[str, Any], *, config_path: str | Path | None = None):
        validate_config(config)
        self.config = deepcopy(config)
        self.config_path = None if config_path is None else Path(config_path)
        self.fingerprint = config_fingerprint(self.config)
        self.training_fingerprint = training_config_fingerprint(self.config)
        self.benchmark_config_fingerprint = benchmark_config_fingerprint(self.config)
        self.paths = ArtifactPaths(Path(self.config["artifacts"]["output_dir"]).resolve())
        certcf_config = self.config["certcf"]
        self.candidate_parallelism = int(certcf_config.get("candidate_parallelism", 1))
        self.candidate_parallel_backend = str(
            certcf_config.get("candidate_parallel_backend", "process")
        ).lower()
        self.candidate_parallel_warmup = bool(
            certcf_config.get("candidate_parallel_warmup", True)
        )
        self.epsilon_parallelism = int(certcf_config.get("epsilon_parallelism", 1))
        self.build_parallelism = int(certcf_config.get("build_parallelism", 1))
        self.configure_epsilon_parallelism(self.epsilon_parallelism)
        self.configure_build_parallelism(self.build_parallelism)
        self.configure_candidate_parallelism(
            workers=self.candidate_parallelism,
            backend=self.candidate_parallel_backend,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NetworkComplexityRunner":
        return cls(load_config(path), config_path=path)

    @property
    def grid(self) -> list[tuple[int, int]]:
        return architecture_grid(self.config)

    def configure_candidate_parallelism(
        self,
        *,
        workers: int | None = None,
        backend: str | None = None,
    ) -> None:
        """Set execution-only candidate parallelism for query-time reruns."""
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

    def configure_build_parallelism(self, workers: int | None = None) -> None:
        """Set execution-only LiRPA class-shard parallelism."""
        if workers is None:
            return
        workers = int(workers)
        if workers <= 0:
            raise ValueError("build parallelism must be positive")
        self.build_parallelism = workers

    def configure_epsilon_parallelism(self, workers: int | None = None) -> None:
        """Set execution-only initial-radius parallelism."""
        if workers is None:
            return
        workers = int(workers)
        if workers <= 0:
            raise ValueError("epsilon parallelism must be positive")
        self.epsilon_parallelism = workers

    @property
    def endpoints(self) -> list[tuple[int, int]]:
        grid = self.grid
        return [min(grid, key=lambda cell: (cell[0], cell[1])), max(grid, key=lambda cell: (cell[0], cell[1]))]

    def prepare(self, *, force: bool = False, announce: bool = True) -> dict[str, Any]:
        dataset_cfg = self.config["dataset"]
        exp = self.config["experiment"]
        existed = self.paths.dataset.exists()
        if announce:
            action = "Rigenerazione" if force and existed else "Preparazione"
            _log(f"[DATASET] {action} dataset condiviso: {self.paths.dataset}")
        metadata = prepare_network_complexity_dataset(
            self.paths.dataset,
            n_train=int(dataset_cfg["n_train"]),
            n_validation=int(dataset_cfg["n_validation"]),
            n_test=int(dataset_cfg["n_test"]),
            n_features=int(exp["input_dim"]),
            seed=int(exp["seed"]),
            class_sep=float(dataset_cfg["class_sep"]),
            flip_y=float(dataset_cfg["flip_y"]),
            n_queries=int(dataset_cfg["n_queries"]),
            queries_per_class=int(dataset_cfg["queries_per_class"]),
            force=force,
        )
        if announce:
            status = "cache valida riutilizzata" if existed and not force else "dataset creato"
            counts = metadata["split_class_counts"]
            _log(
                "[DATASET] "
                f"{status}; train={metadata['split_sizes']['train']} {counts['train']}, "
                f"validation={metadata['split_sizes']['validation']} {counts['validation']}, "
                f"test/query={metadata['split_sizes']['test']} {counts['test']}"
            )
        return metadata

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _valid_training_artifact(self, depth: int, width: int) -> bool:
        checkpoint = self.paths.checkpoint(depth, width)
        metadata = self._read_json(self.paths.train_metadata(depth, width))
        if not checkpoint.exists() or not metadata:
            return False
        stored_training_fingerprint = metadata.get("training_config_fingerprint")
        if stored_training_fingerprint is not None:
            fingerprint_matches = stored_training_fingerprint == self.training_fingerprint
        else:
            # Compatibility for checkpoints created before stage-specific
            # fingerprints. The only supported migration is the just-completed
            # top-5 benchmark protocol, with identical data/training settings.
            legacy_top5_config = deepcopy(self.config)
            legacy_top5_config["certcf"]["query_k_candidates"] = 5
            fingerprint_matches = metadata.get("config_fingerprint") in {
                self.fingerprint,
                config_fingerprint(legacy_top5_config),
            }
        return bool(
            fingerprint_matches
            and metadata.get("checkpoint_sha256") == sha256_file(checkpoint)
        )

    def _make_backbone(self, depth: int, width: int) -> TabularClassifier:
        input_dim = int(self.config["experiment"]["input_dim"])
        return TabularClassifier(
            input_types=["numerical"] * input_dim,
            cardinalities=[],
            hidden_dims=[int(width)] * int(depth),
            num_classes=int(self.config["experiment"]["num_classes"]),
            dropout=float(self.config["training"]["dropout"]),
        )

    @staticmethod
    def _accuracy(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, device: str) -> float:
        model = model.eval().to(device)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), 2048):
                logits = model(torch.from_numpy(x[start : start + 2048]).to(device))
                predictions.append(logits.argmax(dim=1).cpu().numpy())
        predicted = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)
        return float(np.mean(predicted == y)) if len(y) else float("nan")

    def train_architecture(self, depth: int, width: int, *, force: bool = False) -> dict[str, Any]:
        self.prepare(announce=False)
        if self._valid_training_artifact(depth, width) and not force:
            return self._read_json(self.paths.train_metadata(depth, width)) or {}

        exp = self.config["experiment"]
        training = self.config["training"]
        if str(training["accelerator"]).lower() in {"gpu", "cuda"} and not torch.cuda.is_available():
            raise RuntimeError(
                "GPU training is required by the experiment configuration, "
                "but PyTorch cannot access CUDA in this session."
            )
        seed = int(exp["seed"])
        L.seed_everything(seed, workers=True)
        datamodule = NetworkComplexityDataModule(
            cache_path=str(self.paths.dataset),
            batch_size=int(training["batch_size"]),
            num_workers=int(training["num_workers"]),
            seed=seed,
            n_train=int(self.config["dataset"]["n_train"]),
            n_validation=int(self.config["dataset"]["n_validation"]),
            n_test=int(self.config["dataset"]["n_test"]),
            n_features=int(exp["input_dim"]),
            class_sep=float(self.config["dataset"]["class_sep"]),
            flip_y=float(self.config["dataset"]["flip_y"]),
            n_queries=int(self.config["dataset"]["n_queries"]),
            queries_per_class=int(self.config["dataset"]["queries_per_class"]),
        )
        datamodule.prepare_data()
        datamodule.setup()

        backbone = self._make_backbone(depth, width)
        module = LitClassifier(
            model=backbone,
            initial_lr=float(training["initial_lr"]),
            final_lr=float(training["final_lr"]),
            weight_decay=float(training["weight_decay"]),
        )
        checkpoint_dir = self.paths.checkpoint_dir(depth, width)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )
        trainer = L.Trainer(
            max_epochs=int(training["max_epochs"]),
            accelerator=training["accelerator"],
            devices=training["devices"],
            deterministic=bool(training["deterministic"]),
            callbacks=[callback],
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=True,
        )
        started = time.perf_counter()
        trainer.fit(module, datamodule=datamodule)
        training_time_s = time.perf_counter() - started
        best_path = Path(callback.best_model_path)
        if not best_path.exists():
            raise RuntimeError(f"Lightning did not produce a best checkpoint for {architecture_id(depth, width)}")
        checkpoint_path = self.paths.checkpoint(depth, width)
        if best_path.resolve() != checkpoint_path.resolve():
            os.replace(best_path, checkpoint_path)

        restored_backbone = self._make_backbone(depth, width)
        restored = LitClassifier.load_from_checkpoint(
            checkpoint_path,
            model=restored_backbone,
            map_location="cpu",
        ).model
        with np.load(self.paths.dataset, allow_pickle=False) as cached:
            x_test = cached["x_test"].astype(np.float32)
            y_test = cached["y_test"].astype(np.int64)
        evaluation_device = _normalize_device(
            "cuda" if str(training["accelerator"]).lower() in {"auto", "gpu", "cuda"} else training["accelerator"]
        )
        test_accuracy = self._accuracy(restored, x_test, y_test, evaluation_device)
        metadata = {
            "architecture_id": architecture_id(depth, width),
            "depth": int(depth),
            "width": int(width),
            "hidden_dims": [int(width)] * int(depth),
            "parameter_count": parameter_count(
                int(exp["input_dim"]),
                [int(width)] * int(depth),
                int(exp["num_classes"]),
            ),
            "test_accuracy": test_accuracy,
            "training_time_s": float(training_time_s),
            "best_validation_loss": float(callback.best_model_score.cpu())
            if callback.best_model_score is not None
            else float("nan"),
            "config_fingerprint": self.fingerprint,
            "training_config_fingerprint": self.training_fingerprint,
            "dataset_fingerprint": read_dataset_metadata(self.paths.dataset)["fingerprint"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
        }
        _atomic_json(metadata, self.paths.train_metadata(depth, width))
        return metadata

    def train(
        self,
        architectures: Iterable[tuple[int, int]] | None = None,
        *,
        force: bool = False,
        enforce_gate: bool = True,
    ) -> list[dict[str, Any]]:
        selected = list(self.grid if architectures is None else architectures)
        metadata = []
        for index, (depth, width) in enumerate(selected, start=1):
            label = architecture_id(depth, width)
            if self._valid_training_artifact(depth, width) and not force:
                _log(f"[TRAIN {index}/{len(selected)}] {label}: checkpoint valido, skip")
            else:
                _log(
                    f"[TRAIN {index}/{len(selected)}] {label}: avvio "
                    f"({self.config['training']['max_epochs']} epoche, "
                    f"accelerator={self.config['training']['accelerator']})"
                )
            result = self.train_architecture(depth, width, force=force)
            metadata.append(result)
            _log(
                f"[TRAIN {index}/{len(selected)}] {label}: completato; "
                f"test_accuracy={float(result['test_accuracy']):.2%}, "
                f"tempo={float(result['training_time_s']):.1f}s"
            )
        if enforce_gate and set(selected) == set(self.grid):
            self.enforce_accuracy_gate()
        return metadata

    def enforce_accuracy_gate(self) -> None:
        missing = [cell for cell in self.grid if not self._valid_training_artifact(*cell)]
        if missing:
            raise AccuracyGateError(f"Cannot enforce accuracy gate; missing checkpoints for {missing}")
        gate = float(self.config["experiment"]["accuracy_gate"])
        failed = []
        for depth, width in self.grid:
            metadata = self._read_json(self.paths.train_metadata(depth, width)) or {}
            if float(metadata.get("test_accuracy", -math.inf)) < gate:
                failed.append((architecture_id(depth, width), metadata.get("test_accuracy")))
        if failed:
            raise AccuracyGateError(
                f"Common protocol failed the {gate:.1%} test-accuracy gate for {failed}. "
                "Revise the shared training configuration and retrain the complete grid."
            )
        _log(f"[ACCURACY GATE] Tutte le {len(self.grid)} architetture superano {gate:.1%}")

    def _load_model(self, depth: int, width: int, device: str) -> TabularClassifier:
        checkpoint = self.paths.checkpoint(depth, width)
        if not self._valid_training_artifact(depth, width):
            raise RuntimeError(f"Missing or stale training artifact for {architecture_id(depth, width)}")
        backbone = self._make_backbone(depth, width)
        lit = LitClassifier.load_from_checkpoint(
            checkpoint,
            model=backbone,
            map_location=device,
        )
        return lit.model.eval().to(device)

    def _state(self) -> dict[str, Any]:
        state = self._read_json(self.paths.state) or {}
        if state.get("benchmark_config_fingerprint") != self.benchmark_config_fingerprint:
            return {
                "config_fingerprint": self.fingerprint,
                "benchmark_config_fingerprint": self.benchmark_config_fingerprint,
            }
        return state

    def _effective_lirpa_batch_size(self) -> int:
        state = self._state()
        return int(state.get("effective_lirpa_batch_size", self.config["certcf"]["lirpa_batch_size"]))

    def _set_effective_lirpa_batch_size(self, batch_size: int) -> None:
        state = self._state()
        state["config_fingerprint"] = self.fingerprint
        state["benchmark_config_fingerprint"] = self.benchmark_config_fingerprint
        state["effective_lirpa_batch_size"] = int(batch_size)
        _atomic_json(state, self.paths.state)

    def _benchmark_fingerprint(self, checkpoint_hash: str, batch_size: int) -> str:
        payload = {
            "benchmark_config_fingerprint": self.benchmark_config_fingerprint,
            "checkpoint_sha256": checkpoint_hash,
            "effective_lirpa_batch_size": int(batch_size),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _valid_benchmark_artifact(self, depth: int, width: int, batch_size: int) -> bool:
        parquet = self.paths.benchmark(depth, width)
        metadata = self._read_json(self.paths.benchmark_metadata(depth, width))
        if not parquet.exists() or not metadata or not self._valid_training_artifact(depth, width):
            return False
        checkpoint_hash = sha256_file(self.paths.checkpoint(depth, width))
        expected = self._benchmark_fingerprint(checkpoint_hash, batch_size)
        if metadata.get("benchmark_fingerprint") != expected:
            return False
        try:
            frame = pd.read_parquet(parquet, columns=["query_idx", "architecture_id"])
        except Exception:
            return False
        expected_queries = int(self.config["dataset"]["n_queries"])
        return len(frame) == expected_queries and frame["query_idx"].nunique() == expected_queries

    def _shared_benchmark_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.paths.dataset, allow_pickle=False) as cached:
            x_train = cached["x_train"].astype(np.float32)
            y_train = cached["y_train"].astype(np.int64)
            x_test = cached["x_test"].astype(np.float32)
            y_test = cached["y_test"].astype(np.int64)
            query_indices = cached["query_indices"].astype(np.int64)
        cap = int(self.config["certcf"]["train_pool_per_true_class"])
        rng = np.random.default_rng(int(self.config["experiment"]["seed"]))
        selected_parts = []
        for label in (0, 1):
            indices = np.flatnonzero(y_train == label)
            if len(indices) > cap:
                indices = rng.choice(indices, size=cap, replace=False)
            selected_parts.append(indices)
        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)
        return x_train[selected], y_train[selected], x_test, y_test, query_indices

    @staticmethod
    def _atlas_diagnostics(method: CertCF) -> dict[str, Any]:
        atlas = method.atlas
        if atlas is None or atlas.bounds is None:
            return {}
        bounds = [atlas.bounds[label] for label in atlas.class_labels]
        eps_final = np.concatenate([np.asarray(item["eps"]) for item in bounds])
        eps_initial = np.concatenate([np.asarray(item["eps_initial"]) for item in bounds])
        shrinks = np.concatenate([np.asarray(item["adaptive_eps_n_shrinks"]) for item in bounds])
        binary = np.concatenate([np.asarray(item["adaptive_eps_n_binary_steps"]) for item in bounds])
        certified = np.concatenate([np.asarray(item["adaptive_eps_center_certified"]) for item in bounds])
        diagnostics: dict[str, Any] = {
            "atlas_region_count": int(sum(len(item["X"]) for item in bounds)),
            "atlas_region_counts_by_class": {
                str(label): int(len(atlas.bounds[label]["X"])) for label in atlas.class_labels
            },
            "eps_initial_min": float(np.min(eps_initial)),
            "eps_initial_median": float(np.median(eps_initial)),
            "eps_initial_max": float(np.max(eps_initial)),
            "eps_final_min": float(np.min(eps_final)),
            "eps_final_median": float(np.median(eps_final)),
            "eps_final_max": float(np.max(eps_final)),
            "adaptive_eps_shrunk_count": int(np.count_nonzero(shrinks)),
            "adaptive_eps_shrink_total": int(np.sum(shrinks)),
            "adaptive_eps_shrink_max": int(np.max(shrinks)),
            "adaptive_eps_binary_step_total": int(np.sum(binary)),
            "adaptive_eps_center_certified_fraction": float(np.mean(certified)),
        }
        return diagnostics

    def _make_certcf(
        self,
        model: TabularClassifier,
        x_train: np.ndarray,
        y_support: np.ndarray,
        *,
        device: str,
        batch_size: int,
    ) -> tuple[CertCF, dict[str, Any]]:
        cfg = self.config["certcf"]
        wrapper = TorchModelWrapper(model=model.net, device=device)
        method = CertCF(
            model=wrapper,
            norm=int(cfg["norm"]),
            distance_norm=int(cfg["distance_norm"]),
            lirpa_method=str(cfg["lirpa_method"]),
            eps_strategy=NearestOppositeClassClearanceStrategy(alpha=float(cfg["eps_alpha"])),
            batch_size=int(batch_size),
            epsilon_parallelism=self.epsilon_parallelism,
            build_parallelism=self.build_parallelism,
            default_query_method=str(cfg["query_method"]),
            query_k_candidates=int(cfg["query_k_candidates"]),
            solver_maxiter=int(cfg["solver_maxiter"]),
            query_parallelism=int(cfg["query_parallelism"]),
            candidate_parallelism=self.candidate_parallelism,
            candidate_parallel_backend=self.candidate_parallel_backend,
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
            k_per_class=int(cfg["anchors_per_predicted_class"]),
            subsample_method=str(cfg["atlas_subsample_method"]),
            random_seed=int(self.config["experiment"]["seed"]),
        )
        interval = float(self.config["resources"]["rss_sample_interval_s"])
        _log(
            f"[ATLAS] Costruzione con {len(x_train)} support point, "
            f"{int(cfg['anchors_per_predicted_class'])} anchor/classe predetta, "
            f"LiRPA batch={batch_size}"
        )
        with PhaseResourceMonitor("build", device, interval) as monitor:
            method.fit(x_train=x_train, y_train=y_support)
        return method, monitor.metrics

    def benchmark_architecture(
        self,
        depth: int,
        width: int,
        *,
        force: bool = False,
        calibrate_batch_size: bool = False,
    ) -> pd.DataFrame:
        self.prepare(announce=False)
        checkpoint_hash = sha256_file(self.paths.checkpoint(depth, width))
        batch_size = self._effective_lirpa_batch_size()
        if self._valid_benchmark_artifact(depth, width, batch_size) and not force:
            return pd.read_parquet(self.paths.benchmark(depth, width))

        device = _normalize_device(self.config["certcf"]["device"])
        model = self._load_model(depth, width, device)
        x_train, y_train_true, x_test, y_test, query_indices = self._shared_benchmark_data()
        with torch.no_grad():
            y_support = np.concatenate(
                [
                    model(torch.from_numpy(x_train[start : start + 2048]).to(device))
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                    for start in range(0, len(x_train), 2048)
                ]
            ).astype(np.int64, copy=False)

        candidates = [batch_size]
        if calibrate_batch_size:
            configured = [int(value) for value in self.config["certcf"]["lirpa_batch_size_candidates"]]
            candidates = [value for value in configured if value <= batch_size]
        method = None
        build_metrics: dict[str, Any] = {}
        last_oom: BaseException | None = None
        for candidate in candidates:
            try:
                _log(
                    f"[BENCHMARK] {architecture_id(depth, width)}: "
                    f"tentativo LiRPA batch={candidate}"
                )
                method, build_metrics = self._make_certcf(
                    model,
                    x_train,
                    y_support,
                    device=device,
                    batch_size=candidate,
                )
                batch_size = candidate
                if calibrate_batch_size:
                    self._set_effective_lirpa_batch_size(candidate)
                break
            except Exception as exc:
                if not calibrate_batch_size or not _is_cuda_oom(exc):
                    raise
                last_oom = exc
                _log(
                    f"[BENCHMARK] memoria insufficiente con LiRPA batch={candidate}; "
                    "provo il candidato successivo"
                )
                method = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if method is None:
            raise RuntimeError("All common LiRPA batch-size candidates exhausted") from last_oom

        diagnostics = self._atlas_diagnostics(method)
        x_queries = x_test[query_indices]
        y_queries = y_test[query_indices]
        with torch.no_grad():
            query_predictions = model(torch.from_numpy(x_queries).to(device)).argmax(dim=1).cpu().numpy()
        targets = 1 - query_predictions
        query_rows: list[dict[str, Any]] = []
        query_times: list[float] = []
        interval = float(self.config["resources"]["rss_sample_interval_s"])
        timeout_s = float(self.config["certcf"]["timeout_s_per_query"])
        n_valid = 0
        warmup_time_s = 0.0
        warmup_success = False
        if (
            self.candidate_parallelism > 1
            and self.candidate_parallel_backend == "process"
            and self.candidate_parallel_warmup
            and len(x_queries) > 0
        ):
            warmup_started = time.perf_counter()
            warmup = method.generate_batch(
                x=x_queries[:1],
                target_class=int(targets[0]),
                timeout_s_per_query=timeout_s,
            )[0]
            warmup_time_s = time.perf_counter() - warmup_started
            warmup_success = bool(warmup.success)
            _log(
                f"[QUERY WARMUP] {architecture_id(depth, width)}: "
                f"{warmup_time_s:.3f}s, workers={self.candidate_parallelism}, "
                f"success={warmup_success}"
            )
        with PhaseResourceMonitor("query", device, interval) as query_monitor:
            query_iterator = tqdm(
                enumerate(zip(query_indices, x_queries, y_queries, query_predictions, targets)),
                total=len(query_indices),
                desc=f"QUERY {architecture_id(depth, width)}",
                unit="query",
                dynamic_ncols=True,
            )
            for position, (query_idx, x_query, y_true, source, target) in query_iterator:
                started = time.perf_counter()
                error = None
                try:
                    result = method.generate_batch(
                        x=x_query[None, :],
                        target_class=int(target),
                        timeout_s_per_query=timeout_s,
                    )[0]
                except Exception as exc:
                    result = None
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - started
                query_times.append(elapsed)
                x_cf = None if result is None or result.x_cf is None else np.asarray(result.x_cf, dtype=np.float32)
                if x_cf is None:
                    y_cf = -1
                    valid = False
                    distance = float("inf")
                    metadata = {}
                else:
                    with torch.no_grad():
                        y_cf = int(model(torch.from_numpy(x_cf[None, :]).to(device)).argmax(dim=1).item())
                    valid = bool(result.success and y_cf == int(target))
                    distance = float(np.linalg.norm(x_cf - x_query, ord=1))
                    metadata = dict(result.metadata or {})
                row: dict[str, Any] = {
                    "architecture_id": architecture_id(depth, width),
                    "depth": int(depth),
                    "width": int(width),
                    "parameter_count": parameter_count(
                        int(self.config["experiment"]["input_dim"]),
                        [int(width)] * int(depth),
                        int(self.config["experiment"]["num_classes"]),
                    ),
                    "query_position": int(position),
                    "query_idx": int(query_idx),
                    "y_true": int(y_true),
                    "source_class": int(source),
                    "target_class": int(target),
                    "y_cf": int(y_cf),
                    "success": bool(valid),
                    "runtime_s": float(elapsed),
                    "l1_distance": distance,
                    "error": error,
                    "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
                }
                for feature_idx, value in enumerate(x_query):
                    row[f"x_orig_{feature_idx}"] = float(value)
                if x_cf is not None:
                    for feature_idx, value in enumerate(x_cf):
                        row[f"x_cf_{feature_idx}"] = float(value)
                else:
                    for feature_idx in range(x_query.shape[0]):
                        row[f"x_cf_{feature_idx}"] = float("nan")
                query_rows.append(row)
                n_valid += int(valid)
                query_iterator.set_postfix(valid=n_valid, failed=position + 1 - n_valid)
        if method.atlas is not None:
            method.atlas.close_candidate_process_pool()
        query_metrics = query_monitor.metrics

        training_metadata = self._read_json(self.paths.train_metadata(depth, width)) or {}
        atlas_build_profiling = getattr(method.atlas, "build_profiling", {})
        shared_metrics: dict[str, Any] = {
            **build_metrics,
            **query_metrics,
            **diagnostics,
            "query_time_median_s": float(np.median(query_times)),
            "query_time_p95_s": float(np.percentile(query_times, 95)),
            "validity": float(np.mean([row["success"] for row in query_rows])),
            "classifier_test_accuracy": float(training_metadata["test_accuracy"]),
            "classifier_query_accuracy": float(np.mean(query_predictions == y_queries)),
            "training_time_s": float(training_metadata["training_time_s"]),
            "effective_lirpa_batch_size": int(batch_size),
            "epsilon_parallelism": int(self.epsilon_parallelism),
            "build_parallelism": int(self.build_parallelism),
            "lirpa_workers_used": int(
                atlas_build_profiling.get("lirpa_workers_used", 1)
            ),
            "candidate_parallelism": int(self.candidate_parallelism),
            "candidate_parallel_backend": self.candidate_parallel_backend,
            "candidate_parallel_warmup_s": float(warmup_time_s),
            "candidate_parallel_warmup_success": bool(warmup_success),
            "shared_train_pool_size": int(len(x_train)),
            "shared_train_pool_true_class_0": int(np.sum(y_train_true == 0)),
            "shared_train_pool_true_class_1": int(np.sum(y_train_true == 1)),
            "atlas_support_predicted_class_0": int(np.sum(method._y_train == 0)),
            "atlas_support_predicted_class_1": int(np.sum(method._y_train == 1)),
        }
        frame = pd.DataFrame(query_rows)
        for key, value in shared_metrics.items():
            if isinstance(value, (dict, list)):
                frame[key] = json.dumps(value, sort_keys=True)
            else:
                frame[key] = value
        benchmark_fingerprint = self._benchmark_fingerprint(checkpoint_hash, batch_size)
        metadata = {
            "status": "complete",
            "architecture_id": architecture_id(depth, width),
            "depth": int(depth),
            "width": int(width),
            "config_fingerprint": self.fingerprint,
            "benchmark_config_fingerprint": self.benchmark_config_fingerprint,
            "checkpoint_sha256": checkpoint_hash,
            "benchmark_fingerprint": benchmark_fingerprint,
            "effective_lirpa_batch_size": int(batch_size),
            "n_queries": int(len(frame)),
            "metrics": shared_metrics,
        }
        _atomic_parquet(frame, self.paths.benchmark(depth, width))
        _atomic_json(metadata, self.paths.benchmark_metadata(depth, width))
        _log(
            f"[BENCHMARK] {architecture_id(depth, width)} completato: "
            f"validity={shared_metrics['validity']:.2%}, "
            f"build={shared_metrics['build_wall_time_s']:.1f}s, "
            f"query mediana={shared_metrics['query_time_median_s']:.3f}s"
        )
        return frame

    def _record_benchmark_failure(
        self,
        depth: int,
        width: int,
        exc: BaseException,
        *,
        resource_limit: bool,
    ) -> None:
        checkpoint = self.paths.checkpoint(depth, width)
        metadata = {
            "status": "resource_limit" if resource_limit else "failed",
            "architecture_id": architecture_id(depth, width),
            "depth": int(depth),
            "width": int(width),
            "config_fingerprint": self.fingerprint,
            "benchmark_config_fingerprint": self.benchmark_config_fingerprint,
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
            "effective_lirpa_batch_size": self._effective_lirpa_batch_size(),
            "exception_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(metadata, self.paths.benchmark_metadata(depth, width))

    def benchmark(
        self,
        architectures: Iterable[tuple[int, int]] | None = None,
        *,
        force: bool = False,
        calibrate_largest: bool = False,
        enforce_gate: bool = True,
    ) -> pd.DataFrame:
        selected = list(self.grid if architectures is None else architectures)
        if enforce_gate and set(selected) == set(self.grid):
            self.enforce_accuracy_gate()
        largest = max(self.grid, key=lambda cell: (cell[0], cell[1]))
        frames = []
        calibrated_largest = False
        if calibrate_largest and largest in selected:
            _log(
                f"[BENCHMARK 1/{len(selected)}] {architecture_id(*largest)}: "
                "pilot e calibrazione batch LiRPA"
            )
            try:
                frames.append(
                    self.benchmark_architecture(*largest, force=force, calibrate_batch_size=True)
                )
            except Exception as exc:
                self._record_benchmark_failure(
                    *largest,
                    exc,
                    resource_limit=_is_cuda_oom(exc) or "resource" in str(exc).lower(),
                )
                raise
            selected = [cell for cell in selected if cell != largest]
            calibrated_largest = True
        failures: list[tuple[str, BaseException]] = []
        offset = int(calibrated_largest)
        for index, cell in enumerate(selected, start=1 + offset):
            if self._valid_benchmark_artifact(*cell, self._effective_lirpa_batch_size()) and not force:
                _log(
                    f"[BENCHMARK {index}/{len(selected) + offset}] "
                    f"{architecture_id(*cell)}: risultato valido, skip"
                )
            else:
                _log(
                    f"[BENCHMARK {index}/{len(selected) + offset}] "
                    f"{architecture_id(*cell)}: avvio"
                )
            try:
                frames.append(self.benchmark_architecture(*cell, force=force))
            except Exception as exc:
                self._record_benchmark_failure(
                    *cell,
                    exc,
                    resource_limit=_is_cuda_oom(exc) or "resource" in str(exc).lower(),
                )
                failures.append((architecture_id(*cell), exc))
        if failures:
            labels = [(label, f"{type(exc).__name__}: {exc}") for label, exc in failures]
            raise RuntimeError(f"Benchmark failures: {labels}") from failures[0][1]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def combine(self, *, require_complete: bool = True) -> pd.DataFrame:
        batch_size = self._effective_lirpa_batch_size()
        missing = [
            architecture_id(*cell)
            for cell in self.grid
            if not self._valid_benchmark_artifact(*cell, batch_size)
        ]
        if missing and require_complete:
            raise RuntimeError(f"Cannot combine: missing or stale benchmark artifacts: {missing}")
        frames = [
            pd.read_parquet(self.paths.benchmark(*cell))
            for cell in self.grid
            if self._valid_benchmark_artifact(*cell, batch_size)
        ]
        if not frames:
            raise RuntimeError("No valid benchmark artifacts to combine")
        combined = pd.concat(frames, ignore_index=True)
        if require_complete:
            expected_runs = len(self.grid)
            expected_rows = expected_runs * int(self.config["dataset"]["n_queries"])
            if combined["architecture_id"].nunique() != expected_runs or len(combined) != expected_rows:
                raise RuntimeError(
                    f"Combined validation failed: expected {expected_runs} runs/{expected_rows} rows, "
                    f"found {combined['architecture_id'].nunique()} runs/{len(combined)} rows"
                )
        _atomic_parquet(combined, self.paths.combined)
        _log(
            f"[COMBINE] Salvate {len(combined)} righe per "
            f"{combined['architecture_id'].nunique()} architetture in {self.paths.combined}"
        )
        return combined

    @staticmethod
    def summarize_frame(combined: pd.DataFrame) -> pd.DataFrame:
        first_columns = [
            "depth",
            "width",
            "parameter_count",
            "classifier_test_accuracy",
            "classifier_query_accuracy",
            "training_time_s",
            "build_wall_time_s",
            "build_rss_baseline_bytes",
            "build_rss_peak_bytes",
            "build_rss_peak_delta_bytes",
            "build_cuda_peak_allocated_bytes",
            "build_cuda_peak_reserved_bytes",
            "query_wall_time_s",
            "query_rss_baseline_bytes",
            "query_rss_peak_bytes",
            "query_rss_peak_delta_bytes",
            "query_cuda_peak_allocated_bytes",
            "query_cuda_peak_reserved_bytes",
            "query_time_median_s",
            "query_time_p95_s",
            "atlas_region_count",
            "atlas_region_counts_by_class",
            "effective_lirpa_batch_size",
            "eps_initial_min",
            "eps_initial_median",
            "eps_initial_max",
            "eps_final_min",
            "eps_final_median",
            "eps_final_max",
            "adaptive_eps_shrunk_count",
            "adaptive_eps_shrink_total",
            "adaptive_eps_shrink_max",
            "adaptive_eps_center_certified_fraction",
        ]
        aggregations: dict[str, str] = {
            key: "first" for key in first_columns if key in combined.columns
        }
        aggregations.update({"success": "mean", "runtime_s": "count"})
        summary = (
            combined.groupby("architecture_id", as_index=False)
            .agg(aggregations)
            .rename(columns={"success": "validity", "runtime_s": "n_queries"})
        )
        return summary.sort_values(["depth", "width"]).reset_index(drop=True)

    def analyze(self, *, require_complete: bool = True) -> pd.DataFrame:
        combined = self.combine(require_complete=require_complete)
        summary = self.summarize_frame(combined)
        _atomic_parquet(summary, self.paths.summary)
        _log(f"[ANALYZE] Tabella finale salvata in {self.paths.summary}")
        return summary

    def status(self) -> dict[str, Any]:
        batch_size = self._effective_lirpa_batch_size()
        cells = []
        for depth, width in self.grid:
            training_valid = self._valid_training_artifact(depth, width)
            training_metadata = self._read_json(self.paths.train_metadata(depth, width)) or {}
            benchmark_metadata = self._read_json(self.paths.benchmark_metadata(depth, width)) or {}
            cells.append(
                {
                    "architecture_id": architecture_id(depth, width),
                    "depth": depth,
                    "width": width,
                    "training_complete": training_valid,
                    "test_accuracy": training_metadata.get("test_accuracy"),
                    "accuracy_gate_passed": (
                        training_valid
                        and float(training_metadata.get("test_accuracy", -math.inf))
                        >= float(self.config["experiment"]["accuracy_gate"])
                    ),
                    "benchmark_complete": self._valid_benchmark_artifact(depth, width, batch_size),
                    "benchmark_status": benchmark_metadata.get("status"),
                    "benchmark_error": benchmark_metadata.get("error"),
                }
            )
        return {
            "config_fingerprint": self.fingerprint,
            "training_config_fingerprint": self.training_fingerprint,
            "benchmark_config_fingerprint": self.benchmark_config_fingerprint,
            "dataset_complete": self.paths.dataset.exists(),
            "effective_lirpa_batch_size": batch_size,
            "training_complete": sum(cell["training_complete"] for cell in cells),
            "benchmark_complete": sum(cell["benchmark_complete"] for cell in cells),
            "expected_architectures": len(self.grid),
            "cells": cells,
        }

    def all(self, *, force: bool = False) -> pd.DataFrame:
        _log("[PIPELINE 1/4] Preparazione dataset")
        self.prepare(force=force)
        _log("[PIPELINE 2/4] Training completo delle 25 NN e accuracy gate")
        self.train(force=force, enforce_gate=True)
        # No CertCF work starts before every classifier has passed the common
        # accuracy gate. The largest network calibrates the shared LiRPA batch
        # before benchmarking the remaining architectures.
        _log("[PIPELINE 3/4] Benchmark CertCF completo (modello più grande per primo)")
        self.benchmark(force=force, calibrate_largest=True, enforce_gate=True)
        _log("[PIPELINE 4/4] Validazione, combinazione e analisi")
        return self.analyze(require_complete=True)
