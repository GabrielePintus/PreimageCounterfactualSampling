"""Full-grid query-time ablation of CertCF sparsity refinement.

The experiment reuses the dataset and trained classifiers of the controlled
network-complexity grid, rebuilds each atlas with the optimized offline path,
and benchmarks parallel top-k querying with reweighted-L1 refinement disabled.
Artifacts are kept separate from the paper-standard sparsity-enabled results.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from experiments.network_complexity import (
    NetworkComplexityRunner,
    PhaseResourceMonitor,
    _atomic_json,
    _atomic_parquet,
    _normalize_device,
    architecture_id,
    parameter_count,
    sha256_file,
)


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SparsityAblationPaths:
    root: Path

    @property
    def benchmark_dir(self) -> Path:
        return self.root / "benchmarks"

    def benchmark(self, depth: int, width: int) -> Path:
        return self.benchmark_dir / f"{architecture_id(depth, width)}.parquet"

    def partial(self, depth: int, width: int) -> Path:
        return self.benchmark_dir / f".{architecture_id(depth, width)}.partial.parquet"

    def metadata(self, depth: int, width: int) -> Path:
        return self.benchmark_dir / f"{architecture_id(depth, width)}.json"

    @property
    def combined(self) -> Path:
        return self.root / "network_complexity_no_sparsity_queries.parquet"

    @property
    def summary(self) -> Path:
        return self.root / "network_complexity_no_sparsity_summary.parquet"

    @property
    def comparison(self) -> Path:
        return self.root / "network_complexity_sparsity_comparison.parquet"

    @property
    def report(self) -> Path:
        return self.root / "network_complexity_sparsity_comparison.json"


class NetworkComplexitySparsityAblation:
    """Run the optimized no-sparsity query variant over the complete FCNN grid."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        source_config = raw.get("source_config", "configs/experiments/network_complexity_grid.yaml")
        self.source = NetworkComplexityRunner.from_yaml(source_config)
        self.variant = NetworkComplexityRunner.from_yaml(self.config_path)
        cfg = self.variant.config["certcf"]
        if str(cfg["sparsity_penalty"]).lower() != "none":
            raise ValueError("This ablation requires certcf.sparsity_penalty='none'")
        if int(cfg["sparsity_reweight_iters"]) != 0:
            raise ValueError("This ablation requires certcf.sparsity_reweight_iters=0")
        if int(cfg.get("candidate_parallelism", 1)) <= 1:
            raise ValueError("This ablation requires parallel candidate projection")
        self.variant.configure_candidate_parallelism(
            workers=int(cfg["candidate_parallelism"]),
            backend=str(cfg["candidate_parallel_backend"]),
        )
        self.variant.configure_epsilon_parallelism(int(cfg.get("epsilon_parallelism", 1)))
        self.variant.configure_build_parallelism(int(cfg.get("build_parallelism", 1)))
        self.paths = SparsityAblationPaths(
            Path(self.variant.config["artifacts"]["output_dir"]).resolve()
        )
        self.checkpoint_every = max(1, int(raw.get("checkpoint_every_queries", 50)))
        self.protocol_fingerprint = _fingerprint(
            {
                "source_training": self.source.training_fingerprint,
                "source_dataset": deepcopy(self.source.config["dataset"]),
                "variant_benchmark": self.variant.benchmark_config_fingerprint,
                "candidate_parallelism": self.variant.candidate_parallelism,
                "candidate_parallel_backend": self.variant.candidate_parallel_backend,
                "checkpoint_every_queries": self.checkpoint_every,
            }
        )

    @property
    def grid(self) -> list[tuple[int, int]]:
        return self.source.grid

    def _artifact_valid(self, depth: int, width: int) -> bool:
        parquet = self.paths.benchmark(depth, width)
        metadata_path = self.paths.metadata(depth, width)
        if not parquet.exists() or not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame = pd.read_parquet(parquet, columns=["query_position", "query_idx"])
        except Exception:
            return False
        expected = int(self.variant.config["dataset"]["n_queries"])
        return bool(
            metadata.get("status") == "complete"
            and metadata.get("protocol_fingerprint") == self.protocol_fingerprint
            and metadata.get("checkpoint_sha256")
            == sha256_file(self.source.paths.checkpoint(depth, width))
            and len(frame) == expected
            and frame["query_position"].nunique() == expected
            and frame["query_idx"].nunique() == expected
        )

    def _partial(self, depth: int, width: int) -> pd.DataFrame:
        path = self.paths.partial(depth, width)
        if not path.exists():
            return pd.DataFrame()
        try:
            frame = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
        if "protocol_fingerprint" not in frame:
            return pd.DataFrame()
        if not frame["protocol_fingerprint"].eq(self.protocol_fingerprint).all():
            return pd.DataFrame()
        return frame.sort_values("query_position").drop_duplicates("query_position", keep="last")

    def _save_partial(self, rows: list[dict[str, Any]], depth: int, width: int) -> None:
        if rows:
            _atomic_parquet(pd.DataFrame(rows).sort_values("query_position"), self.paths.partial(depth, width))

    def benchmark_architecture(self, depth: int, width: int, *, force: bool = False) -> pd.DataFrame:
        label = architecture_id(depth, width)
        if self._artifact_valid(depth, width) and not force:
            _log(f"[NO-SPARSITY] {label}: artifact complete, skip")
            return pd.read_parquet(self.paths.benchmark(depth, width))
        if not self.source._valid_training_artifact(depth, width):
            raise RuntimeError(f"Missing or stale source checkpoint for {label}")

        device = _normalize_device(self.variant.config["certcf"]["device"])
        if device != "cuda":
            raise RuntimeError("The paper no-sparsity grid requires CUDA")
        model = self.source._load_model(depth, width, device)
        x_train, _, x_test, y_test, query_indices = self.source._shared_benchmark_data()
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

        method, build_metrics = self.variant._make_certcf(
            model,
            x_train,
            y_support,
            device=device,
            batch_size=int(self.variant.config["certcf"]["lirpa_batch_size"]),
        )
        x_queries = x_test[query_indices]
        y_queries = y_test[query_indices]
        with torch.no_grad():
            source_classes = (
                model(torch.from_numpy(x_queries).to(device)).argmax(dim=1).cpu().numpy()
            )
        targets = 1 - source_classes

        previous = pd.DataFrame() if force else self._partial(depth, width)
        rows = previous.to_dict(orient="records")
        completed = set(previous.get("query_position", pd.Series(dtype=int)).astype(int).tolist())

        timeout_s = float(self.variant.config["certcf"]["timeout_s_per_query"])
        warmup_started = time.perf_counter()
        warmup = method.generate_batch(
            x=x_queries[:1], target_class=int(targets[0]), timeout_s_per_query=timeout_s
        )[0]
        warmup_s = time.perf_counter() - warmup_started
        _log(
            f"[NO-SPARSITY] {label}: build={build_metrics['build_wall_time_s']:.3f}s, "
            f"warmup={warmup_s:.3f}s, resume={len(completed)}/{len(query_indices)}"
        )

        interval = float(self.variant.config["resources"]["rss_sample_interval_s"])
        new_since_checkpoint = 0
        try:
            with PhaseResourceMonitor("query", device, interval) as query_monitor:
                iterator = tqdm(
                    enumerate(zip(query_indices, x_queries, y_queries, source_classes, targets)),
                    total=len(query_indices),
                    initial=len(completed),
                    desc=f"NO-SPARSITY {label}",
                    unit="query",
                    dynamic_ncols=True,
                )
                for position, (query_idx, x_query, y_true, source_class, target) in iterator:
                    if position in completed:
                        continue
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
                    x_cf = (
                        None
                        if result is None or result.x_cf is None
                        else np.asarray(result.x_cf, dtype=np.float32)
                    )
                    if x_cf is None:
                        y_cf = -1
                        success = False
                        l0 = l1 = l2 = float("nan")
                        metadata: dict[str, Any] = {}
                    else:
                        difference = x_cf - x_query
                        l0 = float(np.count_nonzero(np.abs(difference) > 1.0e-6))
                        l1 = float(np.linalg.norm(difference, ord=1))
                        l2 = float(np.linalg.norm(difference, ord=2))
                        with torch.no_grad():
                            y_cf = int(
                                model(torch.from_numpy(x_cf[None, :]).to(device))
                                .argmax(dim=1)
                                .item()
                            )
                        success = bool(result.success and y_cf == int(target))
                        metadata = dict(result.metadata or {})
                    row: dict[str, Any] = {
                        "protocol_fingerprint": self.protocol_fingerprint,
                        "architecture_id": label,
                        "depth": int(depth),
                        "width": int(width),
                        "parameter_count": parameter_count(32, [width] * depth, 2),
                        "query_position": int(position),
                        "query_idx": int(query_idx),
                        "y_true": int(y_true),
                        "source_class": int(source_class),
                        "target_class": int(target),
                        "y_cf": int(y_cf),
                        "success": bool(success),
                        "runtime_s": float(elapsed),
                        "l0_sparsity": l0,
                        "l1_distance": l1,
                        "l2_distance": l2,
                        "error": error,
                        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
                        "sparsity_penalty": "none",
                        "candidate_parallelism": int(self.variant.candidate_parallelism),
                        "candidate_parallel_backend": self.variant.candidate_parallel_backend,
                    }
                    for feature_idx, value in enumerate(x_query):
                        row[f"x_orig_{feature_idx}"] = float(value)
                    for feature_idx in range(x_query.shape[0]):
                        row[f"x_cf_{feature_idx}"] = (
                            float(x_cf[feature_idx]) if x_cf is not None else float("nan")
                        )
                    rows.append(row)
                    completed.add(position)
                    new_since_checkpoint += 1
                    if new_since_checkpoint >= self.checkpoint_every:
                        self._save_partial(rows, depth, width)
                        new_since_checkpoint = 0
                    iterator.set_postfix(valid=sum(bool(item["success"]) for item in rows))
            query_metrics = query_monitor.metrics
        except BaseException:
            self._save_partial(rows, depth, width)
            if method.atlas is not None:
                method.atlas.close_candidate_process_pool()
            raise

        if method.atlas is not None:
            method.atlas.close_candidate_process_pool()
        frame = pd.DataFrame(rows).sort_values("query_position").reset_index(drop=True)
        expected = int(self.variant.config["dataset"]["n_queries"])
        if len(frame) != expected or frame["query_position"].nunique() != expected:
            self._save_partial(rows, depth, width)
            raise RuntimeError(f"Incomplete {label}: expected {expected} unique queries, found {len(frame)}")
        _atomic_parquet(frame, self.paths.benchmark(depth, width))
        metadata = {
            "status": "complete",
            "protocol_fingerprint": self.protocol_fingerprint,
            "checkpoint_sha256": sha256_file(self.source.paths.checkpoint(depth, width)),
            "architecture_id": label,
            "n_queries": len(frame),
            "successes": int(frame["success"].sum()),
            "build_metrics": build_metrics,
            "query_metrics": query_metrics,
            "warmup_s": warmup_s,
            "runtime_mean_s": float(frame["runtime_s"].mean()),
            "runtime_median_s": float(frame["runtime_s"].median()),
            "runtime_p95_s": float(frame["runtime_s"].quantile(0.95)),
        }
        _atomic_json(metadata, self.paths.metadata(depth, width))
        self.paths.partial(depth, width).unlink(missing_ok=True)
        _log(
            f"[NO-SPARSITY] {label}: complete, validity={frame['success'].mean():.2%}, "
            f"median={frame['runtime_s'].median() * 1000:.2f} ms"
        )
        return frame

    def benchmark(
        self,
        architectures: Iterable[tuple[int, int]] | None = None,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        selected = list(self.grid if architectures is None else architectures)
        frames = []
        for index, (depth, width) in enumerate(selected, start=1):
            _log(f"[NO-SPARSITY {index}/{len(selected)}] {architecture_id(depth, width)}")
            frames.append(self.benchmark_architecture(depth, width, force=force))
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
        successful = frame[frame["success"]]
        return {
            "n_queries": int(len(frame)),
            "success_rate": float(frame["success"].mean()),
            "runtime_mean_s": float(frame["runtime_s"].mean()),
            "runtime_median_s": float(frame["runtime_s"].median()),
            "runtime_p95_s": float(frame["runtime_s"].quantile(0.95)),
            "l0_mean": float(successful["l0_sparsity"].mean()),
            "l1_mean": float(successful["l1_distance"].mean()),
            "l2_mean": float(successful["l2_distance"].mean()),
        }

    @staticmethod
    def _add_distances(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        original = result.filter(regex=r"^x_orig_\d+$").to_numpy(dtype=float)
        counterfactual = result.filter(regex=r"^x_cf_\d+$").to_numpy(dtype=float)
        difference = counterfactual - original
        result["l0_sparsity"] = np.count_nonzero(np.abs(difference) > 1.0e-6, axis=1)
        result["l2_distance"] = np.linalg.norm(difference, ord=2, axis=1)
        return result

    def analyze(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        missing = [architecture_id(*cell) for cell in self.grid if not self._artifact_valid(*cell)]
        if missing:
            raise RuntimeError(f"Incomplete no-sparsity grid: {missing}")
        no_sparsity = pd.concat(
            [pd.read_parquet(self.paths.benchmark(*cell)) for cell in self.grid],
            ignore_index=True,
        )
        _atomic_parquet(no_sparsity, self.paths.combined)
        summary_rows = []
        for label, group in no_sparsity.groupby("architecture_id", sort=False):
            summary_rows.append({"architecture_id": label, **self._metrics(group)})
        summary = pd.DataFrame(summary_rows).sort_values("architecture_id").reset_index(drop=True)
        _atomic_parquet(summary, self.paths.summary)

        with_sparsity = pd.read_parquet(self.source.paths.combined)
        with_sparsity = self._add_distances(with_sparsity)
        with_summary = pd.DataFrame(
            [
                {"architecture_id": label, **self._metrics(group)}
                for label, group in with_sparsity.groupby("architecture_id", sort=False)
            ]
        )
        comparison = with_summary.merge(
            summary,
            on="architecture_id",
            suffixes=("_with_sparsity", "_without_sparsity"),
            validate="one_to_one",
        )
        comparison["runtime_median_speedup_without_vs_with"] = (
            comparison["runtime_median_s_with_sparsity"]
            / comparison["runtime_median_s_without_sparsity"]
        )
        _atomic_parquet(comparison, self.paths.comparison)
        report = {
            "protocol_fingerprint": self.protocol_fingerprint,
            "architectures": len(self.grid),
            "with_sparsity": self._metrics(with_sparsity),
            "without_sparsity": self._metrics(no_sparsity),
            "median_architecture_speedup": float(
                comparison["runtime_median_speedup_without_vs_with"].median()
            ),
            "minimum_architecture_speedup": float(
                comparison["runtime_median_speedup_without_vs_with"].min()
            ),
            "maximum_architecture_speedup": float(
                comparison["runtime_median_speedup_without_vs_with"].max()
            ),
        }
        _atomic_json(report, self.paths.report)
        return summary, comparison, report

    def status(self) -> dict[str, Any]:
        complete = [architecture_id(*cell) for cell in self.grid if self._artifact_valid(*cell)]
        partial = {
            architecture_id(*cell): len(self._partial(*cell))
            for cell in self.grid
            if not self._artifact_valid(*cell) and not self._partial(*cell).empty
        }
        return {
            "protocol_fingerprint": self.protocol_fingerprint,
            "complete": len(complete),
            "expected": len(self.grid),
            "complete_architectures": complete,
            "partial_queries": partial,
        }
