"""Top-k versus exhaustive CertCF search on the seven tabular benchmarks."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from experiments.lirpa_refinement_ablation import (
    LiRPARefinementAblationRunner,
    _prediction,
)
from experiments.topk_heuristic_ablation import (
    add_minimal_k_columns,
    one_sided_binomial_upper_bound,
    strict_prefix_search,
)


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
        "queries_per_case": 200,
        "k_values": [1, 2, 3, 4, 5, 6, 7],
        "absolute_tolerance": 1.0e-5,
        "relative_tolerance": 1.0e-6,
        "confidence_level": 0.95,
    },
    "pilot": {"queries_per_case": 1, "cases": ["heloc"]},
    "artifacts": {"output_dir": "results/appendix_d_tabular/topk"},
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
    k_values = [int(value) for value in config["experiment"]["k_values"]]
    if not k_values or min(k_values) <= 0 or k_values != sorted(set(k_values)):
        raise ValueError("experiment.k_values must be unique, positive, and increasing")
    if int(config["experiment"]["queries_per_case"]) <= 0:
        raise ValueError("experiment.queries_per_case must be positive")
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

    def query(self, case_id: str, position: int) -> Path:
        return self.root / "queries" / case_id / f"query_{position:04d}.json"

    @property
    def combined(self) -> Path:
        return self.root / "topk_queries.parquet"

    @property
    def per_dataset_summary(self) -> Path:
        return self.root / "topk_summary_by_dataset.parquet"

    @property
    def macro_summary(self) -> Path:
        return self.root / "topk_summary_macro.parquet"

    @property
    def minimal_k(self) -> Path:
        return self.root / "topk_minimal_k.parquet"


class TabularTopKAblationRunner:
    """Run strict top-k and exhaustive search on shared tabular atlases."""

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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TabularTopKAblationRunner":
        return cls(load_config(path), config_path=path)

    @property
    def k_values(self) -> list[int]:
        return [int(value) for value in self.config["experiment"]["k_values"]]

    def resolve_cases(self, identifiers: Sequence[str] | None = None) -> list[dict[str, Any]]:
        requested = (
            list(self.config["experiment"]["cases"])
            if identifiers is None
            else [str(value) for value in identifiers]
        )
        cases = self.source.resolve_cases(requested)
        non_tabular = [str(case["id"]) for case in cases if case["kind"] != "tabular"]
        if non_tabular:
            raise ValueError(f"top-k tabular ablation received non-tabular cases: {non_tabular}")
        return cases

    def _query_count(self, case: dict[str, Any]) -> int:
        geometry = self.source._geometry(str(case["id"]))
        return min(
            int(self.config["experiment"]["queries_per_case"]),
            len(geometry["x_query"]),
        )

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        source_manifest = self.source.prepare(force=force)
        cases = self.resolve_cases()
        manifest = {
            "status": "complete",
            "config_fingerprint": self.fingerprint,
            "source_config_fingerprint": self.source.fingerprint,
            "cases": {
                str(case["id"]): {
                    "queries": self._query_count(case),
                    "atlas_regions_planned": int(
                        len(self.source._geometry(str(case["id"]))["x_anchor"])
                    ),
                }
                for case in cases
            },
            "k_values": self.k_values,
            "source_manifest_status": source_manifest.get("status"),
        }
        _atomic_json(manifest, self.paths.manifest)
        return manifest

    def build(
        self,
        cases: Iterable[dict[str, Any]] | None = None,
        *,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        self.prepare(force=False)
        selected = self.resolve_cases() if cases is None else list(cases)
        return self.source.build_all(selected, force=force)

    def _query_valid(self, case: dict[str, Any], position: int) -> bool:
        path = self.paths.query(str(case["id"]), position)
        if not path.exists():
            return False
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return bool(
            len(rows) == len(self.k_values)
            and all(row.get("config_fingerprint") == self.fingerprint for row in rows)
            and {int(row["k"]) for row in rows} == set(self.k_values)
        )

    def benchmark_case(
        self,
        case: dict[str, Any],
        *,
        query_positions: Sequence[int] | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        self.source.build(case, force=False)
        identifier = str(case["id"])
        geometry = self.source._geometry(identifier)
        available = self._query_count(case)
        positions = (
            list(range(available))
            if query_positions is None
            else [int(value) for value in query_positions]
        )
        if any(position < 0 or position >= available for position in positions):
            raise ValueError(f"{identifier}: query position outside [0, {available})")
        from experiments.lirpa_refinement_ablation import _device

        device = _device(self.source.config)
        _, _, _, _, model, _ = self.source._load_case(case, device)
        method = self.source._load_certcf(case, model, geometry, device)
        atlas = method.atlas
        fixed_dims = self.source._fixed_dims(case)
        absolute_tolerance = float(self.config["experiment"]["absolute_tolerance"])
        relative_tolerance = float(self.config["experiment"]["relative_tolerance"])
        rows: list[dict[str, Any]] = []
        interrupted = [False]
        previous_handler = signal.getsignal(signal.SIGINT)

        def request_stop(signum, frame):
            del signum, frame
            if interrupted[0]:
                raise KeyboardInterrupt
            interrupted[0] = True

        signal.signal(signal.SIGINT, request_stop)
        try:
            for position in tqdm(
                positions,
                desc=f"TOP-K {identifier}",
                unit="query",
                dynamic_ncols=True,
            ):
                path = self.paths.query(identifier, position)
                if self._query_valid(case, position) and not force:
                    rows.extend(json.loads(path.read_text(encoding="utf-8")))
                    continue
                query = geometry["x_query"][position]
                target = int(geometry["target"][position])
                states = strict_prefix_search(
                    atlas,
                    query,
                    target,
                    self.k_values,
                    fixed_dims=fixed_dims,
                )
                exact_started = time.perf_counter()
                exact = atlas.find_counterfactual(
                    query,
                    target_class=target,
                    method="sorted",
                    fixed_dims=fixed_dims,
                )
                exact_runtime_ms = 1.0e3 * (time.perf_counter() - exact_started)
                exact_point = None if exact.x_cf is None else np.asarray(exact.x_cf)
                exact_prediction = (
                    -1
                    if exact_point is None
                    else int(_prediction(model, exact_point[None, :], device)[0])
                )
                exact_success = bool(exact.success and exact_prediction == target)
                exact_distance = float(exact.distance) if exact_success else math.inf
                query_rows: list[dict[str, Any]] = []
                for state in states:
                    strict_prediction = (
                        -1
                        if state.point is None
                        else int(_prediction(model, state.point[None, :], device)[0])
                    )
                    strict_success = bool(state.success and strict_prediction == target)
                    strict_distance = float(state.distance) if strict_success else math.inf
                    shared_success = bool(strict_success and exact_success)
                    raw_gap = strict_distance - exact_distance
                    comparison_tolerance = absolute_tolerance + relative_tolerance * max(
                        1.0, abs(exact_distance)
                    )
                    gap = max(0.0, raw_gap) if shared_success else math.inf
                    relative_gap = (
                        gap / max(abs(exact_distance), absolute_tolerance)
                        if shared_success
                        else math.inf
                    )
                    query_rows.append(
                        {
                            "config_fingerprint": self.fingerprint,
                            "case_id": identifier,
                            "dataset": str(case["dataset"]),
                            "query_position": int(position),
                            "query_index": int(geometry["query_indices"][position]),
                            "source_class": int(geometry["y_query_pred"][position]),
                            "target_class": target,
                            "k": int(state.k),
                            "strict_success": strict_success,
                            "strict_target_valid": strict_success,
                            "strict_distance": strict_distance,
                            "strict_runtime_ms": float(state.runtime_ms),
                            "strict_projection_time_ms": float(state.projection_time_ms),
                            "strict_n_projections": int(state.n_projections),
                            "strict_n_pruned_by_bound": int(state.n_pruned_by_bound),
                            "strict_anchor_rank": state.anchor_rank,
                            "exact_success": exact_success,
                            "exact_target_valid": exact_success,
                            "exact_distance": exact_distance,
                            "exact_n_projections": int(exact.n_qp_solved),
                            "exact_runtime_ms": float(exact_runtime_ms),
                            "exact_recovery": bool(
                                shared_success and raw_gap <= comparison_tolerance
                            ),
                            "within_1pct": bool(
                                shared_success
                                and strict_distance
                                <= exact_distance * 1.01 + absolute_tolerance
                            ),
                            "within_5pct": bool(
                                shared_success
                                and strict_distance
                                <= exact_distance * 1.05 + absolute_tolerance
                            ),
                            "gap_absolute": float(gap),
                            "gap_relative": float(relative_gap),
                        }
                    )
                _atomic_json(query_rows, path)
                rows.extend(query_rows)
                if interrupted[0]:
                    raise KeyboardInterrupt
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            if atlas is not None:
                atlas.close_candidate_process_pool()
        return rows

    def benchmark(
        self,
        cases: Iterable[dict[str, Any]] | None = None,
        *,
        query_positions: Sequence[int] | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        selected = self.resolve_cases() if cases is None else list(cases)
        rows: list[dict[str, Any]] = []
        for case in selected:
            rows.extend(
                self.benchmark_case(
                    case,
                    query_positions=query_positions,
                    force=force,
                )
            )
        return pd.DataFrame(rows)

    def pilot(self, *, force: bool = False) -> dict[str, Any]:
        cases = self.resolve_cases(self.config["pilot"]["cases"])
        count = int(self.config["pilot"]["queries_per_case"])
        self.prepare(force=False)
        self.build(cases, force=force)
        rows: list[dict[str, Any]] = []
        for case in cases:
            rows.extend(
                self.benchmark_case(
                    case,
                    query_positions=list(range(min(count, self._query_count(case)))),
                    force=force,
                )
            )
        return {"cases": len(cases), "queries_per_case": count, "rows": len(rows)}

    def aggregate(self, *, allow_partial: bool = False) -> pd.DataFrame:
        self.prepare(force=False)
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for case in self.resolve_cases():
            identifier = str(case["id"])
            for position in range(self._query_count(case)):
                path = self.paths.query(identifier, position)
                if not self._query_valid(case, position):
                    missing.append(f"{identifier}/{position}")
                    continue
                rows.extend(json.loads(path.read_text(encoding="utf-8")))
        if missing and not allow_partial:
            raise RuntimeError(
                f"missing {len(missing)} complete query artifacts; first: {missing[0]}"
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(["case_id", "query_position", "k"]).reset_index(
                drop=True
            )
            _atomic_parquet(frame, self.paths.combined)
        return frame

    @staticmethod
    def _summarize_group(group: pd.DataFrame, confidence: float) -> dict[str, Any]:
        eligible = group[group["exact_success"].astype(bool)]
        shared = eligible[eligible["strict_success"].astype(bool)]
        misses = ~eligible["exact_recovery"].astype(bool)
        missed_shared = shared[~shared["exact_recovery"].astype(bool)]
        failures = int(misses.sum())
        trials = int(len(eligible))
        finite_gaps = shared["gap_absolute"].replace([np.inf, -np.inf], np.nan)
        finite_relative = shared["gap_relative"].replace([np.inf, -np.inf], np.nan)
        return {
            "n_queries_total": int(len(group)),
            "n_queries_exact_success": trials,
            "strict_success_rate": float(group["strict_success"].mean()),
            "exact_success_rate": float(group["exact_success"].mean()),
            "exact_recovery_rate": float(eligible["exact_recovery"].mean()),
            "miss_probability_empirical": (
                float(failures / trials) if trials else math.nan
            ),
            "miss_probability_upper_confidence": one_sided_binomial_upper_bound(
                failures, trials, confidence
            ),
            "gap_absolute_mean": float(finite_gaps.mean()),
            "gap_absolute_conditional_mean": float(
                missed_shared["gap_absolute"].mean()
            ),
            "gap_absolute_p95": float(finite_gaps.quantile(0.95)),
            "gap_relative_mean": float(finite_relative.mean()),
            "gap_relative_conditional_mean": float(
                missed_shared["gap_relative"].mean()
            ),
            "within_1pct_rate": float(eligible["within_1pct"].mean()),
            "within_5pct_rate": float(eligible["within_5pct"].mean()),
            "strict_runtime_ms_median": float(group["strict_runtime_ms"].median()),
            "exact_runtime_ms_median": float(group["exact_runtime_ms"].median()),
            "strict_projections_mean": float(group["strict_n_projections"].mean()),
            "exact_projections_mean": float(group["exact_n_projections"].mean()),
        }

    def analyze(self, *, allow_partial: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = self.aggregate(allow_partial=allow_partial)
        if frame.empty:
            raise RuntimeError("no top-k query results available")
        confidence = float(self.config["experiment"]["confidence_level"])
        per_dataset_rows: list[dict[str, Any]] = []
        for (dataset, k), group in frame.groupby(["dataset", "k"], sort=True):
            per_dataset_rows.append(
                {
                    "dataset": dataset,
                    "k": int(k),
                    **self._summarize_group(group, confidence),
                }
            )
        per_dataset = pd.DataFrame(per_dataset_rows)
        metric_columns = [
            column
            for column in per_dataset.columns
            if column
            not in {
                "dataset",
                "k",
                "n_queries_total",
                "n_queries_exact_success",
            }
        ]
        macro = (
            per_dataset.groupby("k", as_index=False)[metric_columns]
            .mean(numeric_only=True)
            .sort_values("k")
        )
        macro.insert(1, "n_datasets", per_dataset["dataset"].nunique())
        ranks_frames: list[pd.DataFrame] = []
        maximum_k = max(self.k_values)
        for dataset, group in frame.groupby("dataset", sort=True):
            augmented, ranks = add_minimal_k_columns(
                group.rename(columns={"query_position": "query_position", "query_index": "query_idx"}),
                maximum_k,
            )
            del augmented
            ranks.insert(0, "dataset", dataset)
            ranks_frames.append(ranks)
        ranks = pd.concat(ranks_frames, ignore_index=True)
        _atomic_parquet(per_dataset, self.paths.per_dataset_summary)
        _atomic_parquet(macro, self.paths.macro_summary)
        _atomic_parquet(ranks, self.paths.minimal_k)
        return per_dataset, macro

    def status(self) -> dict[str, Any]:
        self.prepare(force=False)
        return {
            "cases": {
                str(case["id"]): {
                    "atlas_complete": self.source._valid_build(case),
                    "queries_complete": sum(
                        self._query_valid(case, position)
                        for position in range(self._query_count(case))
                    ),
                    "queries_expected": self._query_count(case),
                }
                for case in self.resolve_cases()
            }
        }

    def all(self, *, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.prepare(force=False)
        self.build(force=force)
        self.benchmark(force=force)
        return self.analyze()


__all__ = ["DEFAULT_CONFIG", "TabularTopKAblationRunner", "load_config"]
