"""Paired VERIX/CertCF benchmark on the existing 32D synthetic dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from experiments.network_complexity import (
    NetworkComplexityRunner,
    architecture_id,
    load_config as load_network_complexity_config,
    sha256_file,
)
from experiments.verix_certcf_mnist import (
    ResourceMonitor,
    _atomic_json,
    _atomic_npz,
    _atomic_parquet,
    _deep_merge,
    _json_value,
    _normalize_device,
    _read_npz_metadata,
)
from verix import VeriX
from verix.lenet5_export import onnx_operator_types, validate_marabou_parse
from verix.resume import (
    CheckpointingChecker,
    WallClockBudgetChecker,
    load_check_cache,
    save_check_cache,
)
from verix.traversal import deletion_transform, occlusion_sensitivity_order


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "source_grid_config": "configs/experiments/network_complexity_grid.yaml",
        "architectures": [
            {"depth": 1, "width": 16},
            {"depth": 5, "width": 256},
        ],
    },
    "queries": {"count": 10, "per_true_class": 5},
    "export": {
        "opset_version": 13,
        "validation_batch_size": 256,
        "atol": 1.0e-5,
        "rtol": 1.0e-5,
    },
    "verix": {
        "epsilon": 0.5,
        "traversal": "deletion_sensitivity",
        "norm": "inf",
        "discrepancy": 0.0,
        "classification_margin": 1.0e-6,
        "num_workers": 16,
        "timeout_seconds": 300,
        "solve_with_milp": False,
    },
    "certcf": {"device": "auto"},
    "resources": {"rss_sample_interval_seconds": 0.05},
    "artifacts": {"output_dir": "results/verix_certcf_synthetic32"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def architecture_cells(config: dict[str, Any]) -> list[tuple[int, int]]:
    experiment = config["experiment"]
    configured = experiment.get("architectures")
    if configured is not None:
        return [
            (int(item["depth"]), int(item["width"]))
            for item in configured
        ]
    depths = [int(value) for value in experiment.get("depths", [])]
    widths = [int(value) for value in experiment.get("widths", [])]
    return [(depth, width) for depth in depths for width in widths]


def validate_config(config: dict[str, Any]) -> None:
    cells = architecture_cells(config)
    if not cells or len(cells) != len(set(cells)):
        raise ValueError(
            "the configured architecture list or depth/width grid must be "
            "non-empty and unique"
        )
    if any(depth <= 0 or width <= 0 for depth, width in cells):
        raise ValueError("architecture depth and width must be positive")
    queries = config["queries"]
    if int(queries["count"]) != 2 * int(queries["per_true_class"]):
        raise ValueError("queries.count must equal 2 * queries.per_true_class")
    if int(queries["per_true_class"]) <= 0:
        raise ValueError("queries.per_true_class must be positive")
    verix = config["verix"]
    if verix["traversal"] != "deletion_sensitivity":
        raise ValueError("the standardized 32D protocol uses deletion_sensitivity")
    if str(verix["norm"]).lower() not in {"inf", "infinity"}:
        raise ValueError("the Marabou VERIX backend requires norm=inf")
    if float(verix["epsilon"]) < 0.0:
        raise ValueError("verix.epsilon must be non-negative")
    if float(verix["discrepancy"]) != 0.0:
        raise ValueError("classification VERIX requires discrepancy=0")
    if int(verix["timeout_seconds"]) <= 0:
        raise ValueError("verix.timeout_seconds must be positive")
    query_timeout = verix.get("query_timeout_seconds")
    if query_timeout is not None and float(query_timeout) <= 0.0:
        raise ValueError("verix.query_timeout_seconds must be positive")


def config_fingerprint(config: dict[str, Any]) -> str:
    protocol = copy.deepcopy(config)
    protocol.pop("artifacts", None)
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_balanced_queries(
    query_indices: np.ndarray,
    y_test: np.ndarray,
    *,
    per_true_class: int,
) -> np.ndarray:
    """Select the earliest prepared grid queries with exact class balance."""

    candidates = np.asarray(query_indices, dtype=np.int64)
    selected: list[int] = []
    for label in (0, 1):
        matching = candidates[np.asarray(y_test)[candidates] == label]
        if len(matching) < int(per_true_class):
            raise ValueError(f"not enough prepared queries for true class {label}")
        selected.extend(matching[: int(per_true_class)].tolist())
    selected_set = set(selected)
    return np.asarray(
        [index for index in candidates if int(index) in selected_set],
        dtype=np.int64,
    )


class ONNXTabularScorer:
    """ONNX Runtime logits adapter for flat numerical inputs."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path))
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != 32:
            raise ValueError("synthetic inputs must have shape [N, 32]")
        return np.asarray(
            self.session.run(None, {self.input_name: array})[0],
            dtype=np.float64,
        )

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.logits(values).argmax(axis=1)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def queries(self) -> Path:
        return self.root / "data" / "queries.npz"

    @property
    def combined(self) -> Path:
        return self.root / "verix_certcf_queries.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "summary.parquet"

    def model_dir(self, cell: tuple[int, int]) -> Path:
        return self.root / "models" / architecture_id(*cell)

    def onnx(self, cell: tuple[int, int]) -> Path:
        return self.model_dir(cell) / "model.onnx"

    def export_validation(self, cell: tuple[int, int]) -> Path:
        return self.model_dir(cell) / "equivalence.json"

    def verix_query(self, cell: tuple[int, int], query_index: int) -> Path:
        return (
            self.root
            / "verix"
            / architecture_id(*cell)
            / f"query_{int(query_index):05d}.npz"
        )

    def verix_partial(self, cell: tuple[int, int], query_index: int) -> Path:
        return (
            self.root
            / "verix"
            / architecture_id(*cell)
            / f"query_{int(query_index):05d}.partial.npz"
        )

    def certcf_query(self, cell: tuple[int, int], query_index: int) -> Path:
        return (
            self.root
            / "certcf"
            / architecture_id(*cell)
            / f"query_{int(query_index):05d}.npz"
        )

    def certcf_build(self, cell: tuple[int, int]) -> Path:
        return self.root / "certcf" / architecture_id(*cell) / "build.json"


def _export_onnx(model: torch.nn.Module, path: Path, *, opset_version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    example = torch.zeros((1, 32), dtype=torch.float32)
    torch.onnx.export(
        copy.deepcopy(model).eval().cpu(),
        example,
        str(temporary),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(opset_version),
        do_constant_folding=True,
        dynamo=False,
    )
    os.replace(temporary, path)


def _counterfactual_metrics(
    query: np.ndarray,
    counterfactual: np.ndarray | None,
    *,
    source: int,
    target: int | None,
    logits_function: Callable[[np.ndarray], np.ndarray],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> dict[str, Any]:
    if counterfactual is None or target is None:
        return {
            "success": False,
            "predicted_class": None,
            "l1_distance": np.nan,
            "l2_distance": np.nan,
            "l0_changed": np.nan,
            "target_margin": np.nan,
            "domain_feasible": False,
        }
    x = np.asarray(query, dtype=np.float64).reshape(-1)
    x_cf = np.asarray(counterfactual, dtype=np.float64).reshape(-1)
    logits = np.asarray(logits_function(x_cf[None, :]), dtype=np.float64)[0]
    predicted = int(logits.argmax())
    diff = np.abs(x_cf - x)
    feasible = bool(
        np.isfinite(x_cf).all()
        and np.all(x_cf >= lower_bounds - 1.0e-6)
        and np.all(x_cf <= upper_bounds + 1.0e-6)
    )
    margin = float(logits[int(target)] - logits[int(source)])
    return {
        "success": bool(predicted == int(target) and feasible),
        "predicted_class": predicted,
        "l1_distance": float(np.linalg.norm(diff, ord=1)),
        "l2_distance": float(np.linalg.norm(diff, ord=2)),
        "l0_changed": int(np.count_nonzero(diff > 1.0e-6)),
        "target_margin": margin,
        "domain_feasible": feasible,
    }


def _binary_target(
    source: int,
    verix_target: int | None,
) -> tuple[int, str]:
    """Return the unique binary target while auditing any VERIX witness."""

    source_class = int(source)
    if source_class not in {0, 1}:
        raise ValueError(f"binary source class must be 0 or 1, got {source_class}")
    expected = 1 - source_class
    if verix_target is not None and int(verix_target) != expected:
        raise RuntimeError(
            "VERIX witness target disagrees with the unique opposite binary class"
        )
    return expected, (
        "verix_witness" if verix_target is not None else "binary_opposite"
    )


class Synthetic32Runner:
    """Run the paired experiment on selected network-grid architectures."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        config_path: str | Path | None = None,
    ) -> None:
        validate_config(config)
        self.config = copy.deepcopy(config)
        self.config_path = None if config_path is None else Path(config_path)
        self.fingerprint = config_fingerprint(config)
        repository_root = Path(__file__).resolve().parents[2]
        output_dir = Path(config["artifacts"]["output_dir"])
        if not output_dir.is_absolute():
            output_dir = repository_root / output_dir
        self.paths = ArtifactPaths(output_dir.resolve())
        source_grid_config = Path(
            config["experiment"]["source_grid_config"]
        )
        if not source_grid_config.is_absolute() and not source_grid_config.exists():
            repository_candidate = repository_root / source_grid_config
            if repository_candidate.exists():
                source_grid_config = repository_candidate
        grid_config = load_network_complexity_config(source_grid_config)
        grid_output_dir = Path(grid_config["artifacts"]["output_dir"])
        if not grid_output_dir.is_absolute():
            grid_config["artifacts"]["output_dir"] = str(
                (repository_root / grid_output_dir).resolve()
            )
        self.grid_runner = NetworkComplexityRunner(
            grid_config,
            config_path=source_grid_config,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Synthetic32Runner":
        return cls(load_config(path), config_path=path)

    @property
    def architectures(self) -> list[tuple[int, int]]:
        return architecture_cells(self.config)

    def resolve_architectures(
        self,
        identifiers: Sequence[str] | None,
    ) -> list[tuple[int, int]]:
        if identifiers is None:
            return self.architectures
        mapping = {architecture_id(*cell): cell for cell in self.architectures}
        unknown = sorted(set(identifiers) - set(mapping))
        if unknown:
            raise ValueError(f"unknown configured architectures: {unknown}")
        return [mapping[value] for value in identifiers]

    @staticmethod
    def _log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _source_data(self) -> dict[str, np.ndarray]:
        with np.load(self.grid_runner.paths.dataset, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files if key != "metadata_json"}

    def _load_model(self, cell: tuple[int, int], device: str) -> torch.nn.Module:
        return self.grid_runner._load_model(*cell, device=device)

    @staticmethod
    def _logits_function(
        model: torch.nn.Module,
        *,
        device: str,
    ) -> Callable[[np.ndarray], np.ndarray]:
        device_object = torch.device(device)

        @torch.no_grad()
        def logits(values: np.ndarray) -> np.ndarray:
            array = np.asarray(values, dtype=np.float32)
            if array.ndim == 1:
                array = array[None, :]
            tensor = torch.from_numpy(array).to(device_object)
            return model(tensor).detach().cpu().numpy()

        return logits

    def _read_manifest(self) -> dict[str, Any]:
        if not self.paths.manifest.exists():
            raise RuntimeError("prepare stage has not completed")
        manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != self.fingerprint:
            raise RuntimeError("prepared artifacts use a different configuration")
        if manifest.get("dataset_sha256") != sha256_file(self.grid_runner.paths.dataset):
            raise RuntimeError("source dataset changed; rerun prepare --force")
        for cell in self.architectures:
            identifier = architecture_id(*cell)
            checkpoint = self.grid_runner.paths.checkpoint(*cell)
            if manifest["checkpoint_sha256"].get(identifier) != sha256_file(checkpoint):
                raise RuntimeError(f"checkpoint changed for {identifier}")
            if manifest["onnx_sha256"].get(identifier) != sha256_file(
                self.paths.onnx(cell)
            ):
                raise RuntimeError(f"ONNX artifact changed for {identifier}")
        return manifest

    def _artifact_valid(self, path: Path, stage: str) -> bool:
        if not path.exists():
            return False
        try:
            metadata = _read_npz_metadata(path)
            manifest = self._read_manifest()
        except Exception:
            return False
        return bool(
            metadata.get("status") == "complete"
            and metadata.get("stage") == stage
            and metadata.get("run_fingerprint") == manifest["run_fingerprint"]
        )

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        if self.paths.manifest.exists() and not force:
            try:
                manifest = self._read_manifest()
                self._log("[PREPARE] Artifact validi già presenti.")
                return manifest
            except RuntimeError:
                pass
        if not self.grid_runner.paths.dataset.exists():
            raise FileNotFoundError(
                "missing network-complexity dataset; run its prepare stage first"
            )
        data = self._source_data()
        selected_queries = select_balanced_queries(
            data["query_indices"],
            data["y_test"],
            per_true_class=int(self.config["queries"]["per_true_class"]),
        )
        all_inputs = np.concatenate(
            [data["x_train"], data["x_validation"], data["x_test"]],
            axis=0,
        )
        epsilon = float(self.config["verix"]["epsilon"])
        lower_bounds = all_inputs.min(axis=0).astype(np.float64) - epsilon
        upper_bounds = all_inputs.max(axis=0).astype(np.float64) + epsilon
        _atomic_npz(
            self.paths.queries,
            indices=selected_queries,
            x=data["x_test"][selected_queries].astype(np.float32),
            y_true=data["y_test"][selected_queries].astype(np.int64),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        checkpoint_hashes: dict[str, str] = {}
        onnx_hashes: dict[str, str] = {}
        validations: dict[str, Any] = {}
        export = self.config["export"]
        for cell in self.architectures:
            identifier = architecture_id(*cell)
            checkpoint = self.grid_runner.paths.checkpoint(*cell)
            if not self.grid_runner._valid_training_artifact(*cell):
                raise RuntimeError(f"missing or stale checkpoint for {identifier}")
            model = self._load_model(cell, "cpu")
            self._log(f"[PREPARE] Export ONNX e validazione {identifier}.")
            _export_onnx(
                model,
                self.paths.onnx(cell),
                opset_version=int(export["opset_version"]),
            )
            operators = onnx_operator_types(self.paths.onnx(cell))
            unsupported = set(operators) - {"Gemm", "Relu"}
            if unsupported:
                raise RuntimeError(
                    f"unsupported ONNX operators for {identifier}: {unsupported}"
                )
            parse = validate_marabou_parse(self.paths.onnx(cell))
            if parse != {"input_variables": 32, "output_variables": 2}:
                raise RuntimeError(f"unexpected Marabou graph for {identifier}: {parse}")
            scorer = ONNXTabularScorer(self.paths.onnx(cell))
            torch_logits = self._logits_function(model, device="cpu")
            maximum_error = 0.0
            argmax_identical = True
            allclose = True
            batch_size = int(export["validation_batch_size"])
            for start in range(0, len(data["x_test"]), batch_size):
                batch = data["x_test"][start : start + batch_size]
                expected = torch_logits(batch)
                actual = scorer.logits(batch)
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(expected - actual))),
                )
                allclose &= bool(
                    np.allclose(
                        expected,
                        actual,
                        atol=float(export["atol"]),
                        rtol=float(export["rtol"]),
                    )
                )
                argmax_identical &= bool(
                    np.array_equal(expected.argmax(axis=1), actual.argmax(axis=1))
                )
            report = {
                "n_samples": int(len(data["x_test"])),
                "operators": operators,
                "marabou": parse,
                "maximum_absolute_logit_error": maximum_error,
                "allclose": allclose,
                "argmax_identical": argmax_identical,
            }
            if not allclose or not argmax_identical:
                raise RuntimeError(f"ONNX equivalence failed for {identifier}: {report}")
            _atomic_json(report, self.paths.export_validation(cell))
            checkpoint_hashes[identifier] = sha256_file(checkpoint)
            onnx_hashes[identifier] = sha256_file(self.paths.onnx(cell))
            validations[identifier] = report

        prepared_at_ns = time.time_ns()
        payload = {
            "config": self.fingerprint,
            "dataset": sha256_file(self.grid_runner.paths.dataset),
            "checkpoints": checkpoint_hashes,
            "onnx": onnx_hashes,
            "queries": sha256_file(self.paths.queries),
            "prepared_at_ns": prepared_at_ns,
        }
        run_fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = {
            "status": "complete",
            "config_fingerprint": self.fingerprint,
            "run_fingerprint": run_fingerprint,
            "prepared_at_ns": prepared_at_ns,
            "dataset_sha256": payload["dataset"],
            "queries_sha256": payload["queries"],
            "checkpoint_sha256": checkpoint_hashes,
            "onnx_sha256": onnx_hashes,
            "architectures": [architecture_id(*cell) for cell in self.architectures],
            "query_indices": selected_queries.tolist(),
            "query_true_class_counts": {
                str(label): int(np.sum(data["y_test"][selected_queries] == label))
                for label in (0, 1)
            },
            "domain_bounds_source": "min/max over cached train, validation and test splits, expanded by epsilon",
            "export_validation": validations,
        }
        _atomic_json(manifest, self.paths.manifest)
        self._log(
            f"[PREPARE] Complete: {len(selected_queries)} balanced queries, "
            f"{len(self.architectures)} architetture."
        )
        return manifest

    def _queries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.load(self.paths.queries, allow_pickle=False) as data:
            return (
                data["indices"].copy(),
                data["x"].copy(),
                data["y_true"].copy(),
                data["lower_bounds"].copy(),
                data["upper_bounds"].copy(),
            )

    def _query(self, query_index: int) -> tuple[np.ndarray, int]:
        indices, x, y, _, _ = self._queries()
        positions = np.flatnonzero(indices == int(query_index))
        if len(positions) != 1:
            raise KeyError(f"query {query_index} is not prepared")
        position = int(positions[0])
        return x[position].copy(), int(y[position])

    def run_verix(
        self,
        architectures: Sequence[tuple[int, int]] | None = None,
        query_subset: Sequence[int] | None = None,
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        from verix.backends.marabou import (
            MarabouClassificationChecker,
            MarabouOptions,
        )

        manifest = self._read_manifest()
        cells = self.architectures if architectures is None else list(architectures)
        query_indices_prepared, _, _, lower_bounds, upper_bounds = self._queries()
        selected_queries = (
            query_indices_prepared
            if query_subset is None
            else np.asarray(query_subset, dtype=np.int64)
        )
        config = self.config["verix"]
        summaries: list[dict[str, Any]] = []
        total_tasks = len(cells) * len(selected_queries)
        task_number = 0

        for cell in cells:
            identifier = architecture_id(*cell)
            model = self._load_model(cell, "cpu")
            logits = self._logits_function(model, device="cpu")
            scorer = ONNXTabularScorer(self.paths.onnx(cell))
            checker_native = MarabouClassificationChecker(
                self.paths.onnx(cell),
                input_lower_bounds=lower_bounds,
                input_upper_bounds=upper_bounds,
                options=MarabouOptions(
                    num_workers=int(config["num_workers"]),
                    timeout_seconds=int(config["timeout_seconds"]),
                    solve_with_milp=bool(config["solve_with_milp"]),
                    classification_margin=float(config["classification_margin"]),
                ),
            )
            for query_index_value in selected_queries:
                task_number += 1
                query_index = int(query_index_value)
                output = self.paths.verix_query(cell, query_index)
                if not force and self._artifact_valid(output, "verix"):
                    summaries.append(_read_npz_metadata(output))
                    self._log(
                        f"[VERIX {task_number}/{total_tasks}] {identifier} "
                        f"query={query_index}: completa, skip."
                    )
                    continue
                x_query, y_true = self._query(query_index)
                source = int(scorer.predict(x_query)[0])
                sensitivity_started = time.perf_counter()
                traversal, sensitivity = occlusion_sensitivity_order(
                    x_query,
                    score_function=scorer.logits,
                    predicted_class=source,
                    transform=deletion_transform,
                )
                sensitivity_runtime_seconds = (
                    time.perf_counter() - sensitivity_started
                )
                partial = self.paths.verix_partial(cell, query_index)
                partial_identity = {
                    "run_fingerprint": manifest["run_fingerprint"],
                    "architecture_id": identifier,
                    "query_index": query_index,
                    "traversal_sha256": hashlib.sha256(
                        np.asarray(traversal, dtype=np.int64).tobytes()
                    ).hexdigest(),
                }
                cached = (
                    []
                    if force
                    else load_check_cache(
                        partial,
                        expected_metadata=partial_identity,
                    )
                )

                cached_solver_seconds = float(
                    sum(
                        float(
                            record.metadata.get(
                                "verix_check_elapsed_seconds",
                                0.0,
                            )
                        )
                        for record in cached
                    )
                )
                query_timeout = config.get("query_timeout_seconds")
                budgeted_checker = (
                    checker_native
                    if query_timeout is None
                    else WallClockBudgetChecker(
                        checker_native,
                        timeout_seconds=float(query_timeout),
                        spent_seconds=(
                            cached_solver_seconds
                            + sensitivity_runtime_seconds
                        ),
                        max_solver_timeout_seconds=float(
                            config["timeout_seconds"]
                        ),
                    )
                )

                def persist(records) -> None:
                    save_check_cache(
                        partial,
                        records,
                        input_dimension=32,
                        metadata={
                            **partial_identity,
                            "completed_checks": len(records),
                            "total_checks": 32,
                        },
                    )

                checker = CheckpointingChecker(
                    budgeted_checker,
                    cached=cached,
                    on_update=persist,
                )
                self._log(
                    f"[VERIX {task_number}/{total_tasks}] {identifier} "
                    f"query={query_index}: ripresa da {len(cached)}/32 feature."
                )
                progress = tqdm(
                    total=32,
                    desc=f"  VERIX {identifier} q={query_index}",
                    unit="feature",
                    leave=False,
                )
                interval = float(
                    self.config["resources"]["rss_sample_interval_seconds"]
                )
                try:
                    with ResourceMonitor("query", "cpu", interval) as monitor:
                        result = VeriX(
                            checker,
                            epsilon=float(config["epsilon"]),
                            norm=np.inf,
                            discrepancy=0.0,
                        ).explain(
                            x_query,
                            traversal_order=traversal,
                            step_callback=lambda step: progress.update(1),
                        )
                        validations: list[dict[str, Any]] = []
                        valid_indices: list[int] = []
                        for witness_index, witness in enumerate(result.counterfactuals):
                            x_cf = np.asarray(witness.x, dtype=np.float64)
                            diff = np.abs(x_cf - x_query)
                            free = np.asarray(
                                witness.free_input_indices,
                                dtype=np.int64,
                            )
                            fixed = np.ones(32, dtype=bool)
                            fixed[free] = False
                            prediction = int(logits(x_cf)[0].argmax())
                            free_linf = (
                                float(np.max(diff[free])) if len(free) else 0.0
                            )
                            fixed_max = (
                                float(np.max(diff[fixed])) if np.any(fixed) else 0.0
                            )
                            in_domain = bool(
                                np.all(x_cf >= lower_bounds - 1.0e-6)
                                and np.all(x_cf <= upper_bounds + 1.0e-6)
                            )
                            valid = bool(
                                prediction != source
                                and in_domain
                                and free_linf <= float(config["epsilon"]) + 1.0e-6
                                and fixed_max <= 1.0e-6
                            )
                            validation = {
                                "witness_index": witness_index,
                                "feature": int(witness.feature),
                                "predicted_class": prediction,
                                "canonical_valid": valid,
                                "free_linf_distance": free_linf,
                                "fixed_max_abs_difference": fixed_max,
                                "l1_distance": float(np.linalg.norm(diff, ord=1)),
                                "l2_distance": float(np.linalg.norm(diff, ord=2)),
                                "l0_changed": int(np.count_nonzero(diff > 1.0e-6)),
                                "domain_feasible": in_domain,
                            }
                            validations.append(validation)
                            if valid:
                                valid_indices.append(witness_index)
                        selected_index = (
                            min(
                                valid_indices,
                                key=lambda index: (
                                    validations[index]["l1_distance"],
                                    index,
                                ),
                            )
                            if valid_indices
                            else None
                        )
                        selected_cf = (
                            None
                            if selected_index is None
                            else np.asarray(
                                result.counterfactuals[selected_index].x,
                                dtype=np.float32,
                            )
                        )
                    witnesses = (
                        np.stack([item.x for item in result.counterfactuals]).astype(
                            np.float32
                        )
                        if result.counterfactuals
                        else np.empty((0, 32), dtype=np.float32)
                    )
                    target = (
                        None
                        if selected_index is None
                        else int(validations[selected_index]["predicted_class"])
                    )
                    query_timed_out = bool(
                        isinstance(budgeted_checker, WallClockBudgetChecker)
                        and budgeted_checker.budget_exhausted
                    ) or any(
                        bool(
                            step.metadata.get(
                                "query_budget_exhausted",
                                False,
                            )
                        )
                        for step in result.steps
                    )
                    budget_skipped_checks = int(
                        sum(
                            bool(
                                step.metadata.get(
                                    "query_budget_solver_skipped",
                                    False,
                                )
                            )
                            for step in result.steps
                        )
                    )
                    solver_check_time_seconds = float(
                        sum(
                            float(
                                step.metadata.get(
                                    "verix_check_elapsed_seconds",
                                    step.elapsed_seconds,
                                )
                            )
                            for step in result.steps
                        )
                    )
                    metadata = {
                        "status": "complete",
                        "stage": "verix",
                        "outcome": (
                            "query_timeout" if query_timed_out else "complete"
                        ),
                        "run_fingerprint": manifest["run_fingerprint"],
                        "architecture_id": identifier,
                        "depth": cell[0],
                        "width": cell[1],
                        "query_index": query_index,
                        "true_class": y_true,
                        "source_class": source,
                        "target_class": target,
                        "paired_eligible": selected_index is not None,
                        "selected_witness_index": selected_index,
                        "runtime_seconds": (
                            sensitivity_runtime_seconds
                            + solver_check_time_seconds
                        ),
                        "sensitivity_runtime_seconds": sensitivity_runtime_seconds,
                        "solver_check_time_seconds": solver_check_time_seconds,
                        "current_invocation_wall_time_seconds": monitor.metrics[
                            "query_wall_time_seconds"
                        ],
                        "replayed_feature_checks": checker.replayed_count,
                        "query_timeout_seconds": query_timeout,
                        "query_timed_out": query_timed_out,
                        "budget_skipped_feature_checks": budget_skipped_checks,
                        "feature_checks": len(result.steps),
                        "explanation_size": len(result.explanation),
                        "irrelevant_size": len(result.irrelevant),
                        "unknown_size": len(result.unknown),
                        "native_witness_count": len(result.counterfactuals),
                        "valid_witness_count": len(valid_indices),
                        "locally_minimal": result.locally_minimal,
                        "validations": validations,
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
                        output,
                        metadata_json=np.asarray(
                            json.dumps(_json_value(metadata))
                        ),
                        query=x_query.astype(np.float32),
                        sensitivity=sensitivity.astype(np.float64),
                        traversal_order=np.asarray(traversal, dtype=np.int64),
                        explanation=np.asarray(result.explanation, dtype=np.int64),
                        irrelevant=np.asarray(result.irrelevant, dtype=np.int64),
                        unknown=np.asarray(result.unknown, dtype=np.int64),
                        witnesses=witnesses,
                        selected_counterfactual=(
                            np.empty((0,), dtype=np.float32)
                            if selected_cf is None
                            else selected_cf
                        ),
                    )
                    summaries.append(metadata)
                    self._log(
                        f"[VERIX] {identifier} query={query_index}: "
                        f"target={target}, witness validi={len(valid_indices)}, "
                        f"timeout={query_timed_out}."
                    )
                finally:
                    progress.close()
        return summaries

    def run_certcf(
        self,
        architectures: Sequence[tuple[int, int]] | None = None,
        query_subset: Sequence[int] | None = None,
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        manifest = self._read_manifest()
        cells = self.architectures if architectures is None else list(architectures)
        prepared_indices, _, _, lower_bounds, upper_bounds = self._queries()
        selected_queries = (
            prepared_indices
            if query_subset is None
            else np.asarray(query_subset, dtype=np.int64)
        )
        summaries: list[dict[str, Any]] = []
        device_override = _normalize_device(self.config["certcf"]["device"])
        source_data = self.grid_runner._shared_benchmark_data()
        x_train, _, _, _, _ = source_data

        for cell in cells:
            identifier = architecture_id(*cell)
            missing = [
                int(index)
                for index in selected_queries
                if not self._artifact_valid(
                    self.paths.verix_query(cell, int(index)),
                    "verix",
                )
            ]
            if missing:
                raise RuntimeError(
                    f"VERIX must complete for {identifier}; missing {missing}"
                )
            pending = [
                int(index)
                for index in selected_queries
                if force
                or not self._artifact_valid(
                    self.paths.certcf_query(cell, int(index)),
                    "certcf",
                )
            ]
            if not pending:
                continue
            model = self._load_model(cell, device_override)
            logits = self._logits_function(model, device=device_override)
            with torch.no_grad():
                y_support = logits(x_train).argmax(axis=1).astype(np.int64)
            self._log(f"[CERTCF BUILD] {identifier}: costruzione atlas.")
            method, build_metrics = self.grid_runner._make_certcf(
                model,
                x_train,
                y_support,
                device=device_override,
                batch_size=self.grid_runner._effective_lirpa_batch_size(),
            )
            diagnostics = self.grid_runner._atlas_diagnostics(method)
            _atomic_json(
                {
                    "status": "complete",
                    "run_fingerprint": manifest["run_fingerprint"],
                    "architecture_id": identifier,
                    **build_metrics,
                    **diagnostics,
                },
                self.paths.certcf_build(cell),
            )
            for query_index_value in selected_queries:
                query_index = int(query_index_value)
                output = self.paths.certcf_query(cell, query_index)
                if not force and self._artifact_valid(output, "certcf"):
                    summaries.append(_read_npz_metadata(output))
                    continue
                x_query, y_true = self._query(query_index)
                verix_metadata = _read_npz_metadata(
                    self.paths.verix_query(cell, query_index)
                )
                verix_target = verix_metadata.get("target_class")
                target, target_source = _binary_target(
                    int(verix_metadata["source_class"]),
                    verix_target,
                )
                result = None
                error = None
                started = time.perf_counter()
                try:
                    result = method.generate_batch(
                        x=x_query[None, :],
                        target_class=np.asarray([int(target)]),
                        timeout_s_per_query=float(
                            self.grid_runner.config["certcf"][
                                "timeout_s_per_query"
                            ]
                        ),
                    )[0]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                runtime = time.perf_counter() - started
                x_cf = (
                    None
                    if result is None or result.x_cf is None
                    else np.asarray(result.x_cf, dtype=np.float32)
                )
                metrics = _counterfactual_metrics(
                    x_query,
                    x_cf,
                    source=int(verix_metadata["source_class"]),
                    target=target,
                    logits_function=logits,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )
                metadata = {
                    "status": "complete",
                    "stage": "certcf",
                    "run_fingerprint": manifest["run_fingerprint"],
                    "architecture_id": identifier,
                    "depth": cell[0],
                    "width": cell[1],
                    "query_index": query_index,
                    "true_class": y_true,
                    "source_class": verix_metadata["source_class"],
                    "target_class": target,
                    "target_source": target_source,
                    "verix_witness_available": verix_target is not None,
                    "paired_eligible": True,
                    "method_success": bool(result is not None and result.success),
                    "benchmark_success": bool(
                        result is not None and result.success and metrics["success"]
                    ),
                    "runtime_seconds": runtime,
                    "failure_reason": (
                        error
                        if error is not None
                        else (
                            None
                            if result is not None and result.success
                            else "no_counterfactual"
                        )
                    ),
                    "metrics": metrics,
                    "method_metadata": (
                        {} if result is None else _json_value(result.metadata)
                    ),
                }
                _atomic_npz(
                    output,
                    metadata_json=np.asarray(json.dumps(_json_value(metadata))),
                    query=x_query.astype(np.float32),
                    counterfactual=(
                        np.empty((0,), dtype=np.float32)
                        if x_cf is None
                        else x_cf
                    ),
                )
                summaries.append(metadata)
                self._log(
                    f"[CERTCF] {identifier} query={query_index}: "
                    f"success={metadata['benchmark_success']}."
                )
        return summaries

    def analyze(self, *, allow_partial: bool = False) -> pd.DataFrame:
        manifest = self._read_manifest()
        del manifest
        indices, _, _, lower_bounds, upper_bounds = self._queries()
        rows: list[dict[str, Any]] = []
        for cell in self.architectures:
            identifier = architecture_id(*cell)
            model = self._load_model(cell, "cpu")
            logits = self._logits_function(model, device="cpu")
            build_metadata = (
                json.loads(
                    self.paths.certcf_build(cell).read_text(encoding="utf-8")
                )
                if self.paths.certcf_build(cell).exists()
                else {}
            )
            training_metadata = json.loads(
                self.grid_runner.paths.train_metadata(*cell).read_text(
                    encoding="utf-8"
                )
            )
            for query_index_value in indices:
                query_index = int(query_index_value)
                verix_path = self.paths.verix_query(cell, query_index)
                certcf_path = self.paths.certcf_query(cell, query_index)
                if not (
                    self._artifact_valid(verix_path, "verix")
                    and self._artifact_valid(certcf_path, "certcf")
                ):
                    if allow_partial:
                        continue
                    raise RuntimeError(
                        f"missing paired artifact for {identifier} query={query_index}"
                    )
                x_query, y_true = self._query(query_index)
                with np.load(verix_path, allow_pickle=False) as data:
                    verix_metadata = json.loads(
                        str(data["metadata_json"].item())
                    )
                    selected = data["selected_counterfactual"]
                    verix_cf = None if selected.size == 0 else selected
                verix_target = verix_metadata.get("target_class")
                target, target_source = _binary_target(
                    int(verix_metadata["source_class"]),
                    verix_target,
                )
                verix_metrics = _counterfactual_metrics(
                    x_query,
                    verix_cf,
                    source=int(verix_metadata["source_class"]),
                    target=target,
                    logits_function=logits,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )
                with np.load(certcf_path, allow_pickle=False) as data:
                    certcf_metadata = json.loads(
                        str(data["metadata_json"].item())
                    )
                    candidate = data["counterfactual"]
                    certcf_cf = None if candidate.size == 0 else candidate
                certcf_metrics = _counterfactual_metrics(
                    x_query,
                    certcf_cf,
                    source=int(verix_metadata["source_class"]),
                    target=target,
                    logits_function=logits,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )
                common = {
                    "architecture_id": identifier,
                    "depth": cell[0],
                    "width": cell[1],
                    "parameter_count": training_metadata["parameter_count"],
                    "query_index": query_index,
                    "true_class": y_true,
                    "source_class": verix_metadata["source_class"],
                    "target_class": target,
                    "target_source": target_source,
                    "verix_witness_available": verix_target is not None,
                    "paired_eligible": True,
                    "verix_outcome": verix_metadata.get(
                        "outcome",
                        "complete",
                    ),
                    "verix_query_timed_out": bool(
                        verix_metadata.get("query_timed_out", False)
                    ),
                    "verix_query_timeout_seconds": verix_metadata.get(
                        "query_timeout_seconds"
                    ),
                    "verix_feature_checks": verix_metadata.get(
                        "feature_checks",
                        32,
                    ),
                    "verix_budget_skipped_feature_checks": verix_metadata.get(
                        "budget_skipped_feature_checks",
                        0,
                    ),
                    "verix_explanation_size": verix_metadata.get(
                        "explanation_size",
                        np.nan,
                    ),
                    "verix_unknown_size": verix_metadata.get(
                        "unknown_size",
                        np.nan,
                    ),
                    "verix_valid_witness_count": verix_metadata.get(
                        "valid_witness_count",
                        0,
                    ),
                    "verix_locally_minimal": bool(
                        verix_metadata.get("locally_minimal", False)
                    ),
                }
                rows.append(
                    {
                        **common,
                        "method": "verix",
                        "offline_build_seconds": 0.0,
                        "runtime_seconds": verix_metadata["runtime_seconds"],
                        **verix_metrics,
                    }
                )
                rows.append(
                    {
                        **common,
                        "method": "certcf",
                        "offline_build_seconds": build_metadata.get(
                            "build_wall_time_s",
                            np.nan,
                        ),
                        "runtime_seconds": certcf_metadata["runtime_seconds"],
                        **certcf_metrics,
                        "success": bool(
                            certcf_metadata.get("method_success", False)
                            and certcf_metrics["success"]
                        ),
                    }
                )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(
                ["architecture_id", "query_index", "method"]
            ).reset_index(drop=True)
        _atomic_parquet(frame, self.paths.combined)
        summary = (
            frame.groupby(["architecture_id", "depth", "width", "method"])
            .agg(
                queries=("query_index", "count"),
                success_rate=("success", "mean"),
                mean_l1=("l1_distance", "mean"),
                median_l1=("l1_distance", "median"),
                mean_l0=("l0_changed", "mean"),
                mean_runtime_seconds=("runtime_seconds", "mean"),
                offline_build_seconds=("offline_build_seconds", "first"),
                paired_eligibility_rate=("paired_eligible", "mean"),
                verix_query_timeout_rate=("verix_query_timed_out", "mean"),
                mean_verix_unknown_size=("verix_unknown_size", "mean"),
            )
            .reset_index()
            if not frame.empty
            else pd.DataFrame()
        )
        if not frame.empty:
            distances = frame.pivot(
                index=["architecture_id", "query_index"],
                columns="method",
                values="l1_distance",
            )
            if {"certcf", "verix"}.issubset(distances.columns):
                ratios = (
                    distances["certcf"] / distances["verix"]
                ).replace([np.inf, -np.inf], np.nan)
                ratio_summary = ratios.groupby(level="architecture_id").agg(
                    certcf_over_verix_l1_mean="mean",
                    certcf_over_verix_l1_median="median",
                )
                summary = summary.merge(
                    ratio_summary,
                    left_on="architecture_id",
                    right_index=True,
                    how="left",
                )
        _atomic_parquet(summary, self.paths.summary)
        self._log(
            f"[ANALYZE] Saved {len(frame)} rows and {len(summary)} summaries."
        )
        return frame

    def status(self) -> dict[str, Any]:
        try:
            manifest = self._read_manifest()
        except RuntimeError:
            return {"prepared": False}
        indices, _, _, _, _ = self._queries()
        architectures: dict[str, Any] = {}
        for cell in self.architectures:
            identifier = architecture_id(*cell)
            partial_checks = 0
            for query_index in indices:
                partial = self.paths.verix_partial(cell, int(query_index))
                records = load_check_cache(
                    partial,
                    expected_metadata={
                        "run_fingerprint": manifest["run_fingerprint"],
                        "architecture_id": identifier,
                        "query_index": int(query_index),
                    },
                )
                partial_checks += len(records)
            architectures[identifier] = {
                "verix_complete": sum(
                    self._artifact_valid(
                        self.paths.verix_query(cell, int(index)),
                        "verix",
                    )
                    for index in indices
                ),
                "verix_feature_checks_cached": partial_checks,
                "verix_feature_checks_total": int(len(indices) * 32),
                "verix_query_timeouts": sum(
                    bool(
                        _read_npz_metadata(
                            self.paths.verix_query(cell, int(index))
                        ).get("query_timed_out", False)
                    )
                    for index in indices
                    if self._artifact_valid(
                        self.paths.verix_query(cell, int(index)),
                        "verix",
                    )
                ),
                "certcf_complete": sum(
                    self._artifact_valid(
                        self.paths.certcf_query(cell, int(index)),
                        "certcf",
                    )
                    for index in indices
                ),
            }
        return {
            "prepared": True,
            "run_fingerprint": manifest["run_fingerprint"],
            "queries": len(indices),
            "architectures": architectures,
            "combined_exists": self.paths.combined.exists(),
        }

    def all(self, *, force: bool = False) -> pd.DataFrame:
        self.prepare(force=force)
        self.run_verix(force=force)
        self.run_certcf(force=force)
        return self.analyze()
