"""Resumable CertCF scaling benchmark on pretrained CIFAR-10 ResNets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import time
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

from certcf import CertCFAtlas, NearestOppositeClassClearanceStrategy
from counterfactuals.methods.certcf import CertCF
from counterfactuals.models.torch_model import TorchModelWrapper
from experiments.network_complexity import PhaseResourceMonitor
from models.cifar_resnet import (
    MODEL_SPECS,
    build_cifar10_resnet,
    count_parameters_and_relu_activations,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "networks": ["resnet20", "resnet32", "resnet56"],
        "accuracy_gate": 0.90,
        "require_cuda": True,
    },
    "dataset": {
        "data_dir": "data",
        "support_per_true_class": 1000,
        "queries_per_true_class": 10,
    },
    "pilot": {"queries_per_true_class": 1},
    "certcf": {
        "use_all_support_points": True,
        "eps_reference_scope": "full_support",
        "eps_reference_chunk_size": 128,
        "eps_alpha": 0.20,
        "norm": 1,
        "distance_norm": 1,
        "lirpa_method": "backward",
        "lirpa_batch_size": 1,
        "classification_margin": 1.0e-4,
        "adaptive_eps": True,
        "adaptive_eps_shrink_factor": 0.5,
        "adaptive_eps_max_shrinks": 8,
        "adaptive_eps_min": 1.0e-6,
        "adaptive_eps_center_tol": 1.0e-6,
        "adaptive_eps_binary_search_steps": 0,
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
        "input_bounds": [0.0, 1.0],
        "sparsity_penalty": "none",
    },
    "robustness": {
        "empirical_l1_radii": [0.01, 0.05, 0.1],
        "samples_per_radius": 100,
    },
    "resources": {
        "build_timeout_seconds": 43200,
        "rss_sample_interval_seconds": 0.05,
    },
    "artifacts": {"output_dir": "results/cifar_resnet_scaling"},
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
        config = _deep_merge(DEFAULT_CONFIG, yaml.safe_load(handle) or {})
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    networks = list(config["experiment"]["networks"])
    if not networks or any(name not in MODEL_SPECS for name in networks):
        raise ValueError(f"networks must be selected from {sorted(MODEL_SPECS)}")
    if len(networks) != len(set(networks)):
        raise ValueError("networks must not contain duplicates")
    if int(config["dataset"]["support_per_true_class"]) <= 0:
        raise ValueError("support_per_true_class must be positive")
    if int(config["dataset"]["queries_per_true_class"]) <= 0:
        raise ValueError("queries_per_true_class must be positive")
    certcf = config["certcf"]
    if int(certcf.get("epsilon_parallelism", 1)) <= 0:
        raise ValueError("epsilon_parallelism must be positive")
    if int(certcf.get("build_parallelism", 1)) <= 0:
        raise ValueError("build_parallelism must be positive")
    if int(certcf["norm"]) != 1 or int(certcf["distance_norm"]) != 1:
        raise ValueError("The official CIFAR scaling protocol uses L1 for certification and projection")
    if int(certcf["query_k_candidates"]) != 3:
        raise ValueError("The official CIFAR scaling protocol fixes top-k to 3")
    if not bool(certcf["use_all_support_points"]):
        raise ValueError(
            "The official CIFAR scaling protocol uses every selected support point as an atlas anchor"
        )


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = deepcopy(config)
    payload.pop("artifacts", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
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


def _atomic_link_or_copy(source: Path, target: Path) -> None:
    """Install an existing artifact without duplicating it when hard links are available."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _device(config: dict[str, Any]) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if bool(config["experiment"]["require_cuda"]):
        raise RuntimeError("CUDA is required for the official CIFAR ResNet benchmark")
    return "cpu"


@contextmanager
def _deadline(seconds: int | float | None):
    if not seconds or os.name == "nt":
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise TimeoutError(f"Atlas construction exceeded {seconds} seconds")

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
    def prepared(self) -> Path:
        return self.root / "prepared" / "cifar10.npz"

    @property
    def manifest(self) -> Path:
        return self.root / "prepared" / "manifest.json"

    def checkpoint(self, network: str) -> Path:
        return self.root / "checkpoints" / f"cifar10_{network}.pt"

    def network_dir(self, network: str, *, pilot: bool = False) -> Path:
        return self.root / ("pilot" if pilot else "networks") / network

    def atlas(self, network: str, *, pilot: bool = False) -> Path:
        return self.network_dir(network, pilot=pilot) / "atlas"

    def build(self, network: str, *, pilot: bool = False) -> Path:
        return self.network_dir(network, pilot=pilot) / "build.json"

    def query_dir(self, network: str, *, pilot: bool = False) -> Path:
        return self.network_dir(network, pilot=pilot) / "queries"

    def queries(self, network: str, *, pilot: bool = False) -> Path:
        return self.network_dir(network, pilot=pilot) / "queries.parquet"

    @property
    def combined(self) -> Path:
        return self.root / "cifar_resnet_queries.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "cifar_resnet_summary.parquet"


class CifarResNetScalingRunner:
    def __init__(self, config: dict[str, Any], config_path: str | Path | None = None):
        validate_config(config)
        self.config = config
        self.config_path = None if config_path is None else Path(config_path)
        root = Path(config["artifacts"]["output_dir"])
        if not root.is_absolute():
            root = Path.cwd() / root
        self.paths = Paths(root.resolve())
        self.fingerprint = config_fingerprint(config)
        certcf_config = self.config["certcf"]
        self.candidate_parallelism = int(certcf_config.get("candidate_parallelism", 1))
        self.candidate_parallel_backend = str(
            certcf_config.get("candidate_parallel_backend", "process")
        )
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
    def from_yaml(cls, path: str | Path) -> "CifarResNetScalingRunner":
        return cls(load_config(path), config_path=path)

    def configure_candidate_parallelism(
        self,
        *,
        workers: int | None = None,
        backend: str | None = None,
    ) -> None:
        """Set execution-only query projection parallelism."""
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
        if int(self.config["certcf"]["query_parallelism"]) > 1 and self.candidate_parallelism > 1:
            raise ValueError(
                "query parallelism and candidate parallelism cannot both exceed 1"
            )

    def configure_build_parallelism(self, workers: int | None = None) -> None:
        """Set execution-only LiRPA class-shard parallelism."""
        if workers is None:
            return
        workers = int(workers)
        if workers <= 0:
            raise ValueError("build parallelism must be positive")
        self.build_parallelism = workers

    def configure_epsilon_parallelism(self, workers: int | None = None) -> None:
        """Set execution-only parallelism for initial-radius computation."""
        if workers is None:
            return
        workers = int(workers)
        if workers <= 0:
            raise ValueError("epsilon parallelism must be positive")
        self.epsilon_parallelism = workers

    @property
    def networks(self) -> list[str]:
        return list(self.config["experiment"]["networks"])

    def resolve_networks(self, selected: Iterable[str] | None) -> list[str]:
        names = self.networks if selected is None else list(selected)
        invalid = sorted(set(names).difference(self.networks))
        if invalid:
            raise ValueError(f"Networks not configured: {invalid}")
        return names

    def _download_checkpoint(self, network: str) -> Path:
        target = self.paths.checkpoint(network)
        expected = str(MODEL_SPECS[network]["sha256"])
        if target.exists() and sha256_file(target) == expected:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.download-{os.getpid()}")
        _log(f"[PREPARE] Download checkpoint {network}.")
        urllib.request.urlretrieve(str(MODEL_SPECS[network]["url"]), temporary)
        actual = sha256_file(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Checkpoint hash mismatch for {network}: {actual}")
        os.replace(temporary, target)
        return target

    @staticmethod
    def _images(dataset: CIFAR10) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(dataset.data, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        y = np.asarray(dataset.targets, dtype=np.int64)
        return x, y

    @staticmethod
    def _predict(model: torch.nn.Module, x: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
        model = model.to(device).eval()
        predictions = []
        loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                predictions.append(model(batch.to(device)).argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions).astype(np.int64)

    @staticmethod
    def _balanced_indices(y: np.ndarray, per_class: int, rng: np.random.Generator, eligible=None) -> np.ndarray:
        eligible_mask = np.ones(len(y), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
        parts = []
        for label in range(10):
            candidates = np.flatnonzero((y == label) & eligible_mask)
            if len(candidates) < per_class:
                raise RuntimeError(f"Class {label} has only {len(candidates)} eligible samples")
            parts.append(rng.choice(candidates, size=per_class, replace=False))
        return np.concatenate(parts).astype(np.int64)

    @staticmethod
    def _nearest_other_targets(
        x_query: np.ndarray, y_query: np.ndarray, x_support: np.ndarray, y_support: np.ndarray
    ) -> np.ndarray:
        from scipy.spatial.distance import cdist

        flat_support = x_support.reshape(len(x_support), -1)
        distances = cdist(
            x_query.reshape(len(x_query), -1), flat_support, metric="cityblock"
        )
        distances[y_query[:, None] == y_support[None, :]] = np.inf
        nearest = np.argmin(distances, axis=1)
        return y_support[nearest].astype(np.int64)

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        if self.manifest_valid() and not force:
            _log("[PREPARE] Artifact valido, skip.")
            return json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        seed = int(self.config["experiment"]["seed"])
        rng = np.random.default_rng(seed)
        data_dir = str(self.config["dataset"]["data_dir"])
        _log("[PREPARE] Download/caricamento CIFAR-10.")
        train = CIFAR10(root=data_dir, train=True, download=True)
        test = CIFAR10(root=data_dir, train=False, download=True)
        x_train, y_train = self._images(train)
        x_test, y_test = self._images(test)
        support_indices = self._balanced_indices(
            y_train, int(self.config["dataset"]["support_per_true_class"]), rng
        )
        x_support, y_support = x_train[support_indices], y_train[support_indices]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        common_correct = np.ones(len(y_test), dtype=bool)
        model_metadata = {}
        for network in self.networks:
            checkpoint = self._download_checkpoint(network)
            model = build_cifar10_resnet(network, checkpoint)
            parameters, relus = count_parameters_and_relu_activations(model)
            predictions = self._predict(model, x_test, device)
            accuracy = float(np.mean(predictions == y_test))
            if accuracy < float(self.config["experiment"]["accuracy_gate"]):
                raise RuntimeError(f"{network} accuracy {accuracy:.3%} failed the configured gate")
            common_correct &= predictions == y_test
            model_metadata[network] = {
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_count": parameters,
                "relu_activation_count": relus,
                "test_accuracy": accuracy,
            }
            _log(f"[PREPARE] {network}: accuracy={accuracy:.2%}, ReLU={relus:,}.")
        query_indices = self._balanced_indices(
            y_test,
            int(self.config["dataset"]["queries_per_true_class"]),
            rng,
            eligible=common_correct,
        )
        x_query, y_query = x_test[query_indices], y_test[query_indices]
        targets = self._nearest_other_targets(x_query, y_query, x_support, y_support)
        _atomic_npz(
            self.paths.prepared,
            x_support=x_support.reshape(len(x_support), -1),
            y_support=y_support,
            support_indices=support_indices,
            x_query=x_query.reshape(len(x_query), -1),
            y_query=y_query,
            target_class=targets,
            query_indices=query_indices,
        )
        manifest = {
            "status": "complete",
            "config_fingerprint": self.fingerprint,
            "seed": seed,
            "support_rows": int(len(x_support)),
            "query_rows": int(len(x_query)),
            "support_true_class_counts": {str(i): int(np.sum(y_support == i)) for i in range(10)},
            "query_true_class_counts": {str(i): int(np.sum(y_query == i)) for i in range(10)},
            "models": model_metadata,
            "prepared_sha256": sha256_file(self.paths.prepared),
        }
        _atomic_json(manifest, self.paths.manifest)
        _log(
            f"[PREPARE] Completato: {len(x_support)} punti del training set "
            f"selezionati come anchor, {len(x_query)} query."
        )
        return manifest

    def manifest_valid(self) -> bool:
        if not self.paths.manifest.exists() or not self.paths.prepared.exists():
            return False
        try:
            manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
            return (
                manifest.get("config_fingerprint") == self.fingerprint
                and manifest.get("prepared_sha256") == sha256_file(self.paths.prepared)
            )
        except (OSError, ValueError, KeyError):
            return False

    def _prepared(self) -> dict[str, np.ndarray]:
        if not self.manifest_valid():
            raise RuntimeError("Run prepare before benchmarking")
        with np.load(self.paths.prepared, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def _model(self, network: str, device: str) -> torch.nn.Module:
        checkpoint = self.paths.checkpoint(network)
        if sha256_file(checkpoint) != MODEL_SPECS[network]["sha256"]:
            raise RuntimeError(f"Invalid checkpoint for {network}; rerun prepare")
        return build_cifar10_resnet(network, checkpoint).to(device).eval()

    def _method(
        self,
        model: torch.nn.Module,
        anchors_per_class: int | None = None,
        *,
        bounds_checkpoint_dir: Path | None = None,
    ) -> CertCF:
        config = self.config["certcf"]
        return CertCF(
            model=TorchModelWrapper(model=model, device=next(model.parameters()).device),
            norm=1,
            distance_norm=1,
            lirpa_method=str(config["lirpa_method"]),
            eps_strategy=NearestOppositeClassClearanceStrategy(
                alpha=float(config["eps_alpha"]),
                chunk_size=int(config["eps_reference_chunk_size"]),
            ),
            batch_size=int(config["lirpa_batch_size"]),
            epsilon_parallelism=self.epsilon_parallelism,
            build_parallelism=self.build_parallelism,
            cnn=True,
            cnn_input_shape=(3, 32, 32),
            reuse_lirpa_graph=True,
            bounds_checkpoint_dir=bounds_checkpoint_dir,
            default_query_method=str(config["query_method"]),
            query_k_candidates=3,
            query_parallelism=int(config["query_parallelism"]),
            candidate_parallelism=self.candidate_parallelism,
            candidate_parallel_backend=self.candidate_parallel_backend,
            solver_maxiter=int(config["solver_maxiter"]),
            cvxpy_solvers=list(config["cvxpy_solvers"]),
            cvxpy_solver_options=deepcopy(config["cvxpy_solver_options"]),
            cvxpy_accept_statuses=deepcopy(config["cvxpy_accept_statuses"]),
            classification_margin=float(config["classification_margin"]),
            adaptive_eps=bool(config["adaptive_eps"]),
            adaptive_eps_shrink_factor=float(config["adaptive_eps_shrink_factor"]),
            adaptive_eps_max_shrinks=int(config["adaptive_eps_max_shrinks"]),
            adaptive_eps_min=float(config["adaptive_eps_min"]),
            adaptive_eps_center_tol=float(config["adaptive_eps_center_tol"]),
            adaptive_eps_binary_search_steps=int(config["adaptive_eps_binary_search_steps"]),
            input_bounds=list(config["input_bounds"]),
            sparsity_penalty=str(config["sparsity_penalty"]),
            k_per_class=anchors_per_class,
            subsample_method="random",
            random_seed=int(self.config["experiment"]["seed"]),
            eps_reference_scope=str(config["eps_reference_scope"]),
        )

    def _valid_build(self, network: str, *, pilot: bool = False) -> bool:
        metadata_path = self.paths.build(network, pilot=pilot)
        atlas_manifest = self.paths.atlas(network, pilot=pilot) / "manifest.json"
        if not metadata_path.exists() or not atlas_manifest.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return (
                metadata.get("status") == "complete"
                and metadata.get("config_fingerprint") == self.fingerprint
                and metadata.get("checkpoint_sha256") == MODEL_SPECS[network]["sha256"]
            )
        except (OSError, ValueError):
            return False

    def _promote_pilot_artifacts(self, network: str) -> bool:
        """Reuse an equivalent full-size pilot atlas for the official query run."""
        if self._valid_build(network) or not self._valid_build(network, pilot=True):
            return False

        source_atlas = self.paths.atlas(network, pilot=True)
        source_manifest_path = source_atlas / "manifest.json"
        try:
            manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            atlas_files = [source_atlas / name for name in manifest["files"].values()]
            if not atlas_files or any(not path.exists() for path in atlas_files):
                return False
            build_metadata = json.loads(
                self.paths.build(network, pilot=True).read_text(encoding="utf-8")
            )
        except (OSError, ValueError, KeyError):
            return False

        target_atlas = self.paths.atlas(network)
        for source in atlas_files:
            _atomic_link_or_copy(source, target_atlas / source.name)
        _atomic_link_or_copy(source_manifest_path, target_atlas / "manifest.json")

        build_metadata["pilot"] = False
        build_metadata["reused_from_pilot"] = True
        build_metadata["pilot_artifact_path"] = str(
            self.paths.network_dir(network, pilot=True)
        )
        _atomic_json(build_metadata, self.paths.build(network))

        # The pilot evaluates positions 0, q, 2q, ... and those are also part
        # of the official 100-query run. Reuse them while keeping official
        # metadata separate from the pilot artifacts.
        per_class = int(self.config["dataset"]["queries_per_true_class"])
        for position in (label * per_class for label in range(10)):
            if not self._query_artifact_valid(network, position, pilot=True):
                continue
            source_json = self.paths.query_dir(network, pilot=True) / f"query_{position:03d}.json"
            target_json = self.paths.query_dir(network) / f"query_{position:03d}.json"
            row = json.loads(source_json.read_text(encoding="utf-8"))
            row["pilot"] = False
            row["reused_from_pilot"] = True
            _atomic_json(row, target_json)
            source_npz = self.paths.query_dir(network, pilot=True) / f"query_{position:03d}.npz"
            if source_npz.exists():
                _atomic_link_or_copy(
                    source_npz,
                    self.paths.query_dir(network) / f"query_{position:03d}.npz",
                )
        return True

    def build(self, network: str, *, force: bool = False, pilot: bool = False) -> dict[str, Any]:
        self.prepare()
        if self._valid_build(network, pilot=pilot) and not force:
            _log(f"[BUILD] {network}: atlas valido, skip.")
            return json.loads(self.paths.build(network, pilot=pilot).read_text(encoding="utf-8"))
        if not pilot and not force and self._promote_pilot_artifacts(network):
            _log(f"[BUILD] {network}: riuso dell'atlas completo costruito dal pilot.")
            return json.loads(self.paths.build(network).read_text(encoding="utf-8"))
        prepared = self._prepared()
        device = _device(self.config)
        model = self._model(network, device)
        y_pred = self._predict(
            model,
            prepared["x_support"].reshape(-1, 3, 32, 32),
            device,
        )
        # The selected 10,000 training images are the atlas anchors. Passing
        # k_per_class=None disables CertCF's second, class-wise subsampling.
        anchors_per_class = None
        partial_dir = self.paths.network_dir(network, pilot=pilot) / "partial_bounds"
        partial_state = partial_dir / "state.json"
        state: dict[str, Any] = {}
        if force and partial_dir.exists():
            shutil.rmtree(partial_dir)
        if partial_state.exists():
            state = json.loads(partial_state.read_text(encoding="utf-8"))
            if state.get("config_fingerprint") != self.fingerprint:
                shutil.rmtree(partial_dir)
                state = {}
        previous_build_time_s = float(state.get("accumulated_build_wall_time_s", 0.0))
        previous_rss_peak = int(state.get("build_rss_peak_bytes", 0))
        previous_cuda_allocated = int(state.get("build_cuda_peak_allocated_bytes", 0))
        previous_cuda_reserved = int(state.get("build_cuda_peak_reserved_bytes", 0))
        partial_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            {
                "config_fingerprint": self.fingerprint,
                "network": network,
                "accumulated_build_wall_time_s": previous_build_time_s,
                "build_rss_peak_bytes": previous_rss_peak,
                "build_cuda_peak_allocated_bytes": previous_cuda_allocated,
                "build_cuda_peak_reserved_bytes": previous_cuda_reserved,
            },
            partial_state,
        )
        method = self._method(model, anchors_per_class, bounds_checkpoint_dir=partial_dir)
        _log(
            f"[BUILD] {network}: {len(prepared['x_support'])} punti del training set, "
            f"tutti usati come anchor, device={device}, "
            f"epsilon workers={self.epsilon_parallelism}, "
            f"LiRPA workers={self.build_parallelism}."
        )
        interval = float(self.config["resources"]["rss_sample_interval_seconds"])
        timeout = float(self.config["resources"]["build_timeout_seconds"])
        remaining_timeout = timeout - previous_build_time_s
        if remaining_timeout <= 0.0:
            raise TimeoutError(
                f"Atlas construction already exhausted its cumulative {timeout:g}-second budget"
            )
        try:
            with PhaseResourceMonitor("build", device, interval) as monitor, _deadline(remaining_timeout):
                method.fit(prepared["x_support"].astype(np.float32), y_pred)
        except BaseException as exc:
            partial_metrics = {
                "config_fingerprint": self.fingerprint,
                "network": network,
                "accumulated_build_wall_time_s": previous_build_time_s
                + float(monitor.metrics.get("build_wall_time_s", 0.0)),
                "build_rss_peak_bytes": max(
                    previous_rss_peak,
                    int(monitor.metrics.get("build_rss_peak_bytes", 0)),
                ),
                "build_cuda_peak_allocated_bytes": max(
                    previous_cuda_allocated,
                    int(monitor.metrics.get("build_cuda_peak_allocated_bytes", 0)),
                ),
                "build_cuda_peak_reserved_bytes": max(
                    previous_cuda_reserved,
                    int(monitor.metrics.get("build_cuda_peak_reserved_bytes", 0)),
                ),
            }
            _atomic_json(partial_metrics, partial_state)
            _atomic_json(
                {
                    "status": "resource_limit" if isinstance(exc, TimeoutError) else "failed",
                    "config_fingerprint": self.fingerprint,
                    "checkpoint_sha256": MODEL_SPECS[network]["sha256"],
                    "network": network,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                self.paths.build(network, pilot=pilot),
            )
            raise
        atlas = method.atlas
        if atlas is None or atlas.bounds is None:
            raise RuntimeError("CertCF did not produce an atlas")
        serialization_started = time.perf_counter()
        atlas.save_bounds(self.paths.atlas(network, pilot=pilot))
        serialization_time_s = time.perf_counter() - serialization_started
        bounds = [atlas.bounds[label] for label in atlas.class_labels]
        eps_initial = np.concatenate([item["eps_initial"] for item in bounds])
        eps_final = np.concatenate([item["eps"] for item in bounds])
        shrinks = np.concatenate([item["adaptive_eps_n_shrinks"] for item in bounds])
        attempted = sum(int(np.asarray(item["attempted_anchor_count"]).item()) for item in bounds)
        uncertified = sum(int(np.asarray(item["uncertified_anchor_count"]).item()) for item in bounds)
        manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        metrics = {
            "status": "complete",
            "network": network,
            "pilot": pilot,
            "config_fingerprint": self.fingerprint,
            "checkpoint_sha256": MODEL_SPECS[network]["sha256"],
            "parameter_count": manifest["models"][network]["parameter_count"],
            "relu_activation_count": manifest["models"][network]["relu_activation_count"],
            "classifier_test_accuracy": manifest["models"][network]["test_accuracy"],
            "support_rows": int(len(prepared["x_support"])),
            "attempted_anchor_count": attempted,
            "atlas_region_count": attempted - uncertified,
            "uncertified_anchor_count": uncertified,
            "uncertified_anchor_fraction": float(uncertified / attempted),
            "atlas_region_counts_by_class": {
                str(label): int(len(atlas.bounds[label]["X"])) for label in atlas.class_labels
            },
            "eps_initial_median": float(np.median(eps_initial)),
            "eps_initial_p25": float(np.quantile(eps_initial, 0.25)),
            "eps_initial_p75": float(np.quantile(eps_initial, 0.75)),
            "eps_final_median": float(np.median(eps_final)),
            "eps_final_p25": float(np.quantile(eps_final, 0.25)),
            "eps_final_p75": float(np.quantile(eps_final, 0.75)),
            "eps_ratio_median": float(np.median(eps_final / np.maximum(eps_initial, 1e-30))),
            "adaptive_shrinks_mean": float(np.mean(shrinks)),
            "adaptive_shrinks_max": int(np.max(shrinks)),
            "epsilon_time_s": float(atlas.build_profiling["epsilon_time_s"]),
            "lirpa_time_s": float(atlas.build_profiling["lirpa_time_s"]),
            "epsilon_parallelism": int(atlas.build_profiling["epsilon_parallelism"]),
            "build_parallelism": int(atlas.build_profiling["build_parallelism"]),
            "lirpa_workers_used": int(atlas.build_profiling["lirpa_workers_used"]),
            "serialization_time_s": float(serialization_time_s),
            **monitor.metrics,
        }
        metrics["build_attempt_wall_time_s"] = float(monitor.metrics["build_wall_time_s"])
        metrics["build_wall_time_s"] = previous_build_time_s + float(
            monitor.metrics["build_wall_time_s"]
        )
        metrics["build_rss_peak_bytes"] = max(
            previous_rss_peak, int(monitor.metrics["build_rss_peak_bytes"])
        )
        metrics["build_cuda_peak_allocated_bytes"] = max(
            previous_cuda_allocated,
            int(monitor.metrics["build_cuda_peak_allocated_bytes"]),
        )
        metrics["build_cuda_peak_reserved_bytes"] = max(
            previous_cuda_reserved,
            int(monitor.metrics["build_cuda_peak_reserved_bytes"]),
        )
        _atomic_json(metrics, self.paths.build(network, pilot=pilot))
        _log(
            f"[BUILD] {network}: {metrics['atlas_region_count']}/{attempted} regioni "
            f"in {metrics['build_wall_time_s']:.1f}s."
        )
        return metrics

    def _load_method(self, network: str, *, pilot: bool = False) -> tuple[CertCF, torch.nn.Module]:
        if not self._valid_build(network, pilot=pilot):
            raise RuntimeError(f"Missing valid atlas for {network}; run build first")
        device = _device(self.config)
        model = self._model(network, device)
        atlas_dir = self.paths.atlas(network, pilot=pilot)
        manifest = json.loads((atlas_dir / "manifest.json").read_text(encoding="utf-8"))
        bounds = {}
        x_parts, y_parts = [], []
        for raw_label in manifest["class_labels"]:
            label = int(raw_label)
            with np.load(atlas_dir / manifest["files"][str(label)], allow_pickle=False) as data:
                bounds[label] = {key: data[key] for key in data.files}
            x_parts.append(bounds[label]["X"].astype(np.float32))
            y_parts.append(np.full(len(bounds[label]["X"]), label, dtype=np.int64))
        dataset = TensorDataset(
            torch.from_numpy(np.concatenate(x_parts)), torch.from_numpy(np.concatenate(y_parts))
        )
        config = self.config["certcf"]
        atlas = CertCFAtlas(
            model,
            dataset,
            device,
            cnn=True,
            model_input_shape=(3, 32, 32),
            reuse_lirpa_graph=True,
            norm=1,
            distance_norm=1,
            lirpa_method=str(config["lirpa_method"]),
            input_bounds=list(config["input_bounds"]),
            default_query_method=str(config["query_method"]),
            query_parallelism=int(config["query_parallelism"]),
            candidate_parallelism=self.candidate_parallelism,
            candidate_parallel_backend=self.candidate_parallel_backend,
            solver_maxiter=int(config["solver_maxiter"]),
            cvxpy_solvers=list(config["cvxpy_solvers"]),
            cvxpy_solver_options=deepcopy(config["cvxpy_solver_options"]),
            cvxpy_accept_statuses=deepcopy(config["cvxpy_accept_statuses"]),
            classification_margin=float(config["classification_margin"]),
        )
        atlas.bounds = bounds
        method = self._method(model)
        method.atlas = atlas
        method._is_fitted = True
        return method, model

    @staticmethod
    def _certified_radii(atlas: CertCFAtlas, x_cf: np.ndarray, target: int, anchor: int) -> dict[str, float]:
        if anchor < 0:
            return {"certified_l1_radius": 0.0, "certified_l2_radius": 0.0, "certified_linf_radius": 0.0}
        bounds = atlas.bounds[target]
        A = np.asarray(bounds["lA"][anchor], dtype=np.float64)
        b = np.asarray(bounds["lbias"][anchor], dtype=np.float64)
        x = np.asarray(x_cf, dtype=np.float64)
        slack = np.maximum(A @ x + b - float(atlas.classification_margin), 0.0)
        center = np.asarray(bounds["X"][anchor], dtype=np.float64)
        remaining_l1 = max(0.0, float(bounds["eps"][anchor]) - float(np.linalg.norm(x - center, 1)))
        box_radius = max(0.0, float(np.min(np.minimum(x, 1.0 - x))))
        output = {}
        dimension = x.size
        for name, primal, dual, conversion in (
            ("l1", 1, np.inf, 1.0),
            ("l2", 2, 2, math.sqrt(dimension)),
            ("linf", np.inf, 1, float(dimension)),
        ):
            denominator = np.linalg.norm(A, ord=dual, axis=1)
            halfspace = float(np.min(slack / np.maximum(denominator, 1e-30)))
            output[f"certified_{name}_radius"] = max(
                0.0, min(halfspace, remaining_l1 / conversion, box_radius)
            )
        return output

    def _query_artifact_valid(self, network: str, position: int, *, pilot: bool) -> bool:
        path = self.paths.query_dir(network, pilot=pilot) / f"query_{position:03d}.json"
        if not path.exists():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("config_fingerprint") == self.fingerprint
        except (OSError, ValueError):
            return False

    def query(self, network: str, *, force: bool = False, pilot: bool = False) -> pd.DataFrame:
        prepared = self._prepared()
        load_started = time.perf_counter()
        method, model = self._load_method(network, pilot=pilot)
        atlas_load_time_s = time.perf_counter() - load_started
        if pilot:
            per_class = int(self.config["dataset"]["queries_per_true_class"])
            selected_positions = [label * per_class for label in range(10)]
        else:
            selected_positions = list(range(len(prepared["x_query"])))
        rows = []
        warmup_time_s = 0.0
        warmup_success = False
        if (
            self.candidate_parallelism > 1
            and self.candidate_parallel_backend == "process"
            and self.candidate_parallel_warmup
            and selected_positions
        ):
            warmup_position = selected_positions[0]
            warmup_x = prepared["x_query"][warmup_position].astype(np.float32)
            warmup_target = int(prepared["target_class"][warmup_position])
            warmup_started = time.perf_counter()
            warmup = method.generate_batch(
                warmup_x[None, :],
                target_class=warmup_target,
                timeout_s_per_query=float(self.config["certcf"]["timeout_s_per_query"]),
            )[0]
            warmup_time_s = time.perf_counter() - warmup_started
            warmup_success = bool(warmup.success)
            _log(
                f"[QUERY WARMUP] {network}: {warmup_time_s:.3f}s, "
                f"workers={self.candidate_parallelism}, success={warmup_success}."
            )
        try:
            for progress, position in enumerate(selected_positions, start=1):
                json_path = (
                    self.paths.query_dir(network, pilot=pilot)
                    / f"query_{position:03d}.json"
                )
                npz_path = (
                    self.paths.query_dir(network, pilot=pilot)
                    / f"query_{position:03d}.npz"
                )
                if self._query_artifact_valid(network, position, pilot=pilot) and not force:
                    rows.append(json.loads(json_path.read_text(encoding="utf-8")))
                    continue
                x = prepared["x_query"][position].astype(np.float32)
                target = int(prepared["target_class"][position])
                _log(f"[QUERY {progress}/{len(selected_positions)}] {network}: target={target}.")
                started = time.perf_counter()
                result = method.generate_batch(
                    x[None, :],
                    target_class=target,
                    timeout_s_per_query=float(
                        self.config["certcf"]["timeout_s_per_query"]
                    ),
                )[0]
                elapsed = time.perf_counter() - started
                success = bool(result.success and result.x_cf is not None)
                x_cf = np.asarray(
                    result.x_cf if success else np.empty(0), dtype=np.float32
                )
                row: dict[str, Any] = {
                    "config_fingerprint": self.fingerprint,
                    "network": network,
                    "pilot": pilot,
                    "query_position": position,
                    "test_index": int(prepared["query_indices"][position]),
                    "true_class": int(prepared["y_query"][position]),
                    "target_class": target,
                    "success": success,
                    "timeout": str(result.metadata.get("reason", "")).lower()
                    == "timeout",
                    "runtime_s": float(elapsed),
                    "atlas_load_time_s": float(atlas_load_time_s),
                    "distance_l0": np.nan,
                    "distance_l1": np.nan,
                    "distance_l2": np.nan,
                    "distance_linf": np.nan,
                    "psnr": np.nan,
                    "ssim": np.nan,
                    "validity": False,
                    "anchor_idx": int(result.metadata.get("anchor_idx", -1)),
                    "fallback_used": bool(
                        result.metadata.get("nearest_anchor_fallback_used", False)
                    ),
                    "n_candidates_considered": float(
                        result.metadata.get("n_candidates_considered", np.nan)
                    ),
                }
                if success:
                    difference = x_cf - x
                    mse = float(np.mean(difference * difference))
                    row.update(
                        {
                            "distance_l0": int(
                                np.count_nonzero(np.abs(difference) > 1e-6)
                            ),
                            "distance_l1": float(np.linalg.norm(difference, 1)),
                            "distance_l2": float(np.linalg.norm(difference, 2)),
                            "distance_linf": float(np.linalg.norm(difference, np.inf)),
                            "psnr": (
                                float("inf")
                                if mse == 0
                                else float(10.0 * np.log10(1.0 / mse))
                            ),
                        }
                    )
                    try:
                        from skimage.metrics import structural_similarity

                        original = x.reshape(3, 32, 32).transpose(1, 2, 0)
                        candidate = x_cf.reshape(3, 32, 32).transpose(1, 2, 0)
                        row["ssim"] = float(
                            structural_similarity(
                                original,
                                candidate,
                                channel_axis=2,
                                data_range=1.0,
                            )
                        )
                    except ImportError:
                        row["ssim"] = np.nan
                    with torch.no_grad():
                        prediction = int(
                            model(
                                torch.from_numpy(x_cf.reshape(1, 3, 32, 32)).to(
                                    next(model.parameters()).device
                                )
                            )
                            .argmax(dim=1)
                            .item()
                        )
                    row["validity"] = prediction == target
                    rng = np.random.default_rng(
                        int(self.config["experiment"]["seed"])
                        + 1000 * self.networks.index(network)
                        + position
                    )
                    robustness_values = []
                    robustness_started = time.perf_counter()
                    sample_count = int(
                        self.config["robustness"]["samples_per_radius"]
                    )
                    for radius in self.config["robustness"]["empirical_l1_radii"]:
                        weights = rng.exponential(size=(sample_count, x_cf.size))
                        weights /= np.maximum(
                            weights.sum(axis=1, keepdims=True), 1e-30
                        )
                        signs = rng.choice(np.array([-1.0, 1.0]), size=weights.shape)
                        scales = rng.random(sample_count) ** (1.0 / x_cf.size)
                        noise = signs * weights * (float(radius) * scales[:, None])
                        perturbed = np.clip(
                            x_cf[None, :] + noise, 0.0, 1.0
                        ).astype(np.float32)
                        predictions = self._predict(
                            model,
                            perturbed.reshape(-1, 3, 32, 32),
                            str(next(model.parameters()).device),
                        )
                        value = float(np.mean(predictions == target))
                        row[f"empirical_robustness_l1_{float(radius):g}"] = value
                        robustness_values.append(value)
                    row["empirical_robustness"] = float(np.mean(robustness_values))
                    row["robustness_time_s"] = float(
                        time.perf_counter() - robustness_started
                    )
                    row.update(
                        self._certified_radii(
                            method.atlas, x_cf, target, int(row["anchor_idx"])
                        )
                    )
                else:
                    row.update(
                        {
                            "empirical_robustness": 0.0,
                            "robustness_time_s": 0.0,
                            "certified_l1_radius": 0.0,
                            "certified_l2_radius": 0.0,
                            "certified_linf_radius": 0.0,
                        }
                    )
                row.update(
                    {
                        "candidate_parallelism": int(self.candidate_parallelism),
                        "candidate_parallel_backend": self.candidate_parallel_backend,
                        "candidate_parallel_warmup_s": float(warmup_time_s),
                        "candidate_parallel_warmup_success": bool(warmup_success),
                    }
                )
                _atomic_npz(npz_path, x=x, x_cf=x_cf)
                _atomic_json(row, json_path)
                rows.append(row)
        finally:
            if method.atlas is not None:
                method.atlas.close_candidate_process_pool()
        frame = pd.DataFrame(rows)
        _atomic_parquet(frame, self.paths.queries(network, pilot=pilot))
        return frame

    def benchmark(self, network: str, *, force: bool = False, pilot: bool = False) -> pd.DataFrame:
        self.build(network, force=force, pilot=pilot)
        return self.query(network, force=force, pilot=pilot)

    def aggregate(self, *, allow_partial: bool = False) -> pd.DataFrame:
        frames = []
        missing = []
        expected_queries = 10 * int(self.config["dataset"]["queries_per_true_class"])
        for network in self.networks:
            path = self.paths.queries(network)
            if not path.exists() or not self._valid_build(network):
                missing.append(network)
                continue
            frame = pd.read_parquet(path)
            if (
                len(frame) != expected_queries
                or set(frame["config_fingerprint"].astype(str)) != {self.fingerprint}
                or set(frame["network"].astype(str)) != {network}
            ):
                missing.append(network)
                continue
            frames.append(frame)
        if missing and not allow_partial:
            raise RuntimeError(f"Missing or stale network results: {missing}")
        if not frames:
            raise RuntimeError("No complete network results to aggregate")
        combined = pd.concat(frames, ignore_index=True)
        _atomic_parquet(combined, self.paths.combined)
        _log(f"[AGGREGATE] {len(combined)} query per {combined['network'].nunique()} reti.")
        return combined

    def analyze(self, *, allow_partial: bool = False) -> pd.DataFrame:
        combined = self.aggregate(allow_partial=allow_partial)
        build_rows = [
            json.loads(self.paths.build(network).read_text(encoding="utf-8"))
            for network in sorted(combined["network"].unique())
        ]
        query_summary = (
            combined.groupby("network", as_index=False)
            .agg(
                query_count=("success", "size"),
                success_rate=("success", "mean"),
                validity=("validity", "mean"),
                timeout_rate=("timeout", "mean"),
                query_time_median_s=("runtime_s", "median"),
                query_time_p95_s=("runtime_s", lambda values: float(np.quantile(values, 0.95))),
                mean_l0=("distance_l0", "mean"),
                mean_l1=("distance_l1", "mean"),
                mean_l2=("distance_l2", "mean"),
                mean_linf=("distance_linf", "mean"),
                mean_psnr=("psnr", "mean"),
                mean_ssim=("ssim", "mean"),
                empirical_robustness=("empirical_robustness", "mean"),
                mean_certified_l1_radius=("certified_l1_radius", "mean"),
                mean_certified_l2_radius=("certified_l2_radius", "mean"),
                mean_certified_linf_radius=("certified_linf_radius", "mean"),
                fallback_rate=("fallback_used", "mean"),
            )
        )
        summary = pd.DataFrame(build_rows).merge(query_summary, on="network", how="left")
        order = {name: position for position, name in enumerate(self.networks)}
        summary["network_order"] = summary["network"].map(order)
        summary = summary.sort_values("network_order").drop(columns="network_order")
        _atomic_parquet(summary, self.paths.summary)
        _log(f"[ANALYZE] Tabella salvata in {self.paths.summary}.")
        return summary

    def status(self) -> dict[str, Any]:
        expected = 10 * int(self.config["dataset"]["queries_per_true_class"])
        networks = []
        for network in self.networks:
            build_path = self.paths.build(network)
            build = json.loads(build_path.read_text(encoding="utf-8")) if build_path.exists() else {}
            completed_queries = sum(
                self._query_artifact_valid(network, position, pilot=False)
                for position in range(expected)
            )
            networks.append(
                {
                    "network": network,
                    "build_status": build.get("status", "missing"),
                    "atlas_complete": self._valid_build(network),
                    "completed_queries": int(completed_queries),
                    "expected_queries": expected,
                    "error": build.get("error"),
                }
            )
        return {
            "config_fingerprint": self.fingerprint,
            "prepared": self.manifest_valid(),
            "networks": networks,
            "combined_exists": self.paths.combined.exists(),
            "summary_exists": self.paths.summary.exists(),
        }

    def all(self, *, force: bool = False) -> pd.DataFrame:
        self.prepare(force=force)
        self.benchmark("resnet20", force=force, pilot=True)
        for network in self.networks:
            self.benchmark(network, force=force)
        return self.analyze()
