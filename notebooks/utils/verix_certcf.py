"""Analysis helpers for the paired VERIX/CertCF synthetic-32 benchmark."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


_HELPERS_DIR = Path(__file__).resolve().parent
_NOTEBOOKS_DIR = _HELPERS_DIR.parent
_REPO_ROOT = _NOTEBOOKS_DIR.parent
_SRC_ROOT = _REPO_ROOT / "src"

for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


METHOD_ORDER = ["verix", "certcf"]
PAPER_ROBUSTNESS_SIGMAS = (0.00, 0.01, 0.03, 0.05, 0.10)
PAPER_ROBUSTNESS_SAMPLES = 10
LOF_NEIGHBORS = 20
IFOREST_ESTIMATORS = 200
IFOREST_RANDOM_STATE = 0
DIAGNOSTICS_VERSION = 4
CERTIFIED_NORMS = {
    "l1": 1.0,
    "l2": 2.0,
    "linf": float("inf"),
}


def load_synthetic32_results(
    result_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load query rows, runner summary, and preparation manifest."""

    root = Path(result_dir)
    queries_path = root / "verix_certcf_queries.parquet"
    summary_path = root / "summary.parquet"
    manifest_path = root / "manifest.json"
    if not queries_path.exists():
        raise FileNotFoundError(
            f"Missing {queries_path}. Run the synthetic32 analyze stage first."
        )
    queries = pd.read_parquet(queries_path)
    summary = (
        pd.read_parquet(summary_path)
        if summary_path.exists()
        else pd.DataFrame()
    )
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    validate_query_rows(queries)
    return queries, summary, manifest


def validate_query_rows(frame: pd.DataFrame) -> None:
    required = {
        "architecture_id",
        "query_index",
        "method",
        "paired_eligible",
        "success",
        "l1_distance",
        "l2_distance",
        "l0_changed",
        "runtime_seconds",
        "offline_build_seconds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"query result is missing columns: {missing}")
    duplicates = frame.duplicated(
        ["architecture_id", "query_index", "method"],
        keep=False,
    )
    if duplicates.any():
        raise ValueError("duplicate architecture/query/method rows detected")
    unknown = sorted(set(frame["method"].dropna()) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"unexpected methods: {unknown}")


def completion_table(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Report method completion for every configured architecture."""

    expected_queries = int(len(manifest.get("query_indices", [])))
    architectures = manifest.get(
        "architectures",
        sorted(frame["architecture_id"].unique().tolist()),
    )
    rows = []
    for architecture in architectures:
        for method in METHOD_ORDER:
            subset = frame[
                (frame["architecture_id"] == architecture)
                & (frame["method"] == method)
            ]
            unique_queries = int(subset["query_index"].nunique())
            rows.append(
                {
                    "architecture_id": architecture,
                    "method": method,
                    "rows": int(len(subset)),
                    "unique_queries": unique_queries,
                    "expected_queries": expected_queries,
                    "complete": bool(
                        expected_queries > 0
                        and len(subset) == expected_queries
                        and unique_queries == expected_queries
                    ),
                }
            )
    return pd.DataFrame(rows)


def method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute per-architecture method metrics with explicit denominators."""

    validate_query_rows(frame)
    rows = []
    for (architecture, depth, width, parameters, method), group in frame.groupby(
        ["architecture_id", "depth", "width", "parameter_count", "method"],
        observed=True,
        sort=True,
    ):
        eligible = group[group["paired_eligible"].astype(bool)]
        successful = eligible[eligible["success"].astype(bool)]
        rows.append(
            {
                "architecture_id": architecture,
                "depth": int(depth),
                "width": int(width),
                "parameter_count": int(parameters),
                "method": method,
                "attempted_queries": int(len(group)),
                "paired_eligible_queries": int(len(eligible)),
                "successful_queries": int(len(successful)),
                "success_rate": (
                    float(eligible["success"].mean()) if len(eligible) else np.nan
                ),
                "mean_l1": float(successful["l1_distance"].mean()),
                "median_l1": float(successful["l1_distance"].median()),
                "mean_l2": float(successful["l2_distance"].mean()),
                "mean_l0": float(successful["l0_changed"].mean()),
                "mean_query_seconds": float(group["runtime_seconds"].mean()),
                "median_query_seconds": float(group["runtime_seconds"].median()),
                "offline_build_seconds": float(
                    group["offline_build_seconds"].iloc[0]
                ),
                "mean_target_margin": float(successful["target_margin"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    result["method"] = pd.Categorical(
        result["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    return result.sort_values(["depth", "width", "method"]).reset_index(drop=True)


def paired_query_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return shared-success query pairs and method deltas."""

    validate_query_rows(frame)
    paired_source = frame[
        frame["paired_eligible"].astype(bool) & frame["success"].astype(bool)
    ]
    metrics = [
        "l1_distance",
        "l2_distance",
        "l0_changed",
        "runtime_seconds",
        "target_margin",
    ]
    paired = paired_source.pivot(
        index=[
            "architecture_id",
            "depth",
            "width",
            "parameter_count",
            "query_index",
        ],
        columns="method",
        values=metrics,
    )
    if not {"verix", "certcf"}.issubset(
        set(paired.columns.get_level_values("method"))
    ):
        return pd.DataFrame()
    paired = paired.dropna(
        subset=[
            ("l1_distance", "verix"),
            ("l1_distance", "certcf"),
        ]
    ).copy()
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    paired["certcf_minus_verix_l1"] = (
        paired["l1_distance_certcf"] - paired["l1_distance_verix"]
    )
    paired["certcf_over_verix_l1"] = (
        paired["l1_distance_certcf"] / paired["l1_distance_verix"]
    ).replace([np.inf, -np.inf], np.nan)
    paired["certcf_minus_verix_l0"] = (
        paired["l0_changed_certcf"] - paired["l0_changed_verix"]
    )
    paired["certcf_minus_verix_query_seconds"] = (
        paired["runtime_seconds_certcf"] - paired["runtime_seconds_verix"]
    )
    return paired


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired proximity, sparsity, and runtime differences."""

    if paired.empty:
        return pd.DataFrame()
    return (
        paired.groupby(
            ["architecture_id", "depth", "width", "parameter_count"],
            observed=True,
        )
        .agg(
            shared_successes=("query_index", "count"),
            mean_certcf_over_verix_l1=("certcf_over_verix_l1", "mean"),
            median_certcf_over_verix_l1=("certcf_over_verix_l1", "median"),
            mean_certcf_minus_verix_l1=("certcf_minus_verix_l1", "mean"),
            mean_certcf_minus_verix_l0=("certcf_minus_verix_l0", "mean"),
            mean_certcf_minus_verix_query_seconds=(
                "certcf_minus_verix_query_seconds",
                "mean",
            ),
        )
        .reset_index()
    )


def load_counterfactual_vectors(
    result_dir: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the saved query and counterfactual vectors to result rows."""

    validate_query_rows(frame)
    root = Path(result_dir)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        method = str(record["method"])
        architecture = str(record["architecture_id"])
        query_index = int(record["query_index"])
        path = (
            root
            / method
            / architecture
            / f"query_{query_index:05d}.npz"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing counterfactual artifact: {path}")
        with np.load(path, allow_pickle=False) as data:
            query = np.asarray(data["query"], dtype=np.float32).reshape(-1)
            key = (
                "selected_counterfactual"
                if method == "verix"
                else "counterfactual"
            )
            counterfactual = np.asarray(data[key], dtype=np.float32).reshape(-1)
        if query.shape != (32,):
            raise ValueError(f"Unexpected query shape in {path}: {query.shape}")
        if counterfactual.size == 0:
            counterfactual = np.full(32, np.nan, dtype=np.float32)
        elif counterfactual.shape != (32,):
            raise ValueError(
                f"Unexpected counterfactual shape in {path}: "
                f"{counterfactual.shape}"
            )
        rows.append(
            {
                **record,
                **{f"x_orig_{idx}": float(value) for idx, value in enumerate(query)},
                **{
                    f"x_cf_{idx}": float(value)
                    for idx, value in enumerate(counterfactual)
                },
            }
        )
    return pd.DataFrame(rows)


def _feature_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(32)]


def _training_nn_reference(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict[int, NearestNeighbors], float]:
    """Fit target-class L1 indices and return the natural NN distance mean."""

    y = np.asarray(y_train, dtype=np.int64).reshape(-1)
    indices: dict[int, NearestNeighbors] = {}
    reference_distances: list[np.ndarray] = []
    for label in np.unique(y):
        class_x = np.asarray(x_train[y == int(label)], dtype=np.float32)
        if len(class_x) < 2:
            continue
        leave_one_out = NearestNeighbors(
            n_neighbors=2,
            metric="manhattan",
            n_jobs=-1,
        ).fit(class_x)
        distances, _ = leave_one_out.kneighbors(class_x)
        reference_distances.append(distances[:, 1])
        indices[int(label)] = NearestNeighbors(
            n_neighbors=1,
            metric="manhattan",
            n_jobs=-1,
        ).fit(class_x)
    if not reference_distances:
        raise ValueError("Training data do not contain a valid NN reference")
    natural_mean = float(np.mean(np.concatenate(reference_distances)))
    return indices, natural_mean


def attach_manifoldness_metrics(
    frame: pd.DataFrame,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    lof_neighbors: int = LOF_NEIGHBORS,
    include_relative_proximity: bool = True,
) -> pd.DataFrame:
    """Attach paper-aligned LOF and Isolation Forest metrics.

    Relative Proximity Ratio can optionally be attached for analyses that use
    it; the Synthetic32 VERIX comparison explicitly disables that calculation.
    """

    result = frame.copy()
    cf_columns = _feature_columns("x_cf")
    missing = sorted(set(cf_columns) - set(result.columns))
    if missing:
        raise ValueError(f"Missing counterfactual columns: {missing[:3]}")

    x_train_array = np.asarray(x_train, dtype=np.float32)
    y_train_array = np.asarray(y_train, dtype=np.int64).reshape(-1)
    if x_train_array.ndim != 2 or x_train_array.shape[1] != 32:
        raise ValueError("Synthetic32 training data must have shape [N, 32]")

    target_indices: dict[int, NearestNeighbors] = {}
    natural_mean = np.nan
    if include_relative_proximity:
        target_indices, natural_mean = _training_nn_reference(
            x_train_array,
            y_train_array,
        )
    k_eff = max(2, min(int(lof_neighbors), len(x_train_array) - 1))
    lof = LocalOutlierFactor(
        n_neighbors=k_eff,
        novelty=True,
        n_jobs=-1,
    ).fit(x_train_array)
    isolation_forest = IsolationForest(
        n_estimators=IFOREST_ESTIMATORS,
        contamination="auto",
        random_state=IFOREST_RANDOM_STATE,
        n_jobs=-1,
    ).fit(x_train_array)

    target_nn = np.full(len(result), np.nan, dtype=float)
    negative_lof = np.full(len(result), np.nan, dtype=float)
    positive_lof = np.full(len(result), np.nan, dtype=float)
    iforest_score = np.full(len(result), np.nan, dtype=float)
    valid = (
        result["success"].astype(bool).to_numpy()
        & np.isfinite(result[cf_columns].to_numpy(dtype=float)).all(axis=1)
    )
    valid_positions = np.flatnonzero(valid)
    if len(valid_positions):
        valid_cf = result.iloc[valid_positions][cf_columns].to_numpy(
            dtype=np.float32
        )
        negative_lof[valid_positions] = lof.score_samples(valid_cf)
        positive_lof[valid_positions] = np.maximum(
            -negative_lof[valid_positions],
            1.0e-12,
        )
        iforest_score[valid_positions] = isolation_forest.score_samples(
            valid_cf
        )
        if include_relative_proximity:
            valid_targets = result.iloc[valid_positions][
                "target_class"
            ].to_numpy(dtype=np.int64)
            for target in np.unique(valid_targets):
                local = np.flatnonzero(valid_targets == int(target))
                index = target_indices.get(int(target))
                if index is None:
                    continue
                distances, _ = index.kneighbors(valid_cf[local])
                target_nn[valid_positions[local]] = distances[:, 0]

    if include_relative_proximity:
        result["target_class_nn_l1"] = target_nn
        result["natural_same_class_nn_l1"] = natural_mean
        result["relative_proximity_ratio"] = target_nn / natural_mean
    result["negative_lof_score"] = negative_lof
    result["positive_lof"] = positive_lof
    result["log10_lof"] = np.log10(positive_lof)
    result["isolation_forest_score"] = iforest_score
    return result


def evaluate_paper_empirical_robustness(
    frame: pd.DataFrame,
    *,
    models_by_architecture: dict[str, Any],
    sigmas: tuple[float, ...] = PAPER_ROBUSTNESS_SIGMAS,
    samples_per_sigma: int = PAPER_ROBUSTNESS_SAMPLES,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate the paper's numerical Gaussian robustness protocol.

    Common Gaussian draws are used for the two methods on each
    architecture/query/sigma tuple, making the comparison paired while
    preserving the paper's perturbation magnitudes and sample count.
    """

    if samples_per_sigma <= 0:
        raise ValueError("samples_per_sigma must be positive")
    cf_columns = _feature_columns("x_cf")
    rows: list[dict[str, Any]] = []
    successful = frame[
        frame["paired_eligible"].astype(bool) & frame["success"].astype(bool)
    ].copy()
    if successful.empty:
        return pd.DataFrame()

    def predict(model: Any, values: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict"):
            return np.asarray(model.predict(values), dtype=np.int64).reshape(-1)
        try:
            model_device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            model_device = torch.device("cpu")
        with torch.no_grad():
            tensor = torch.as_tensor(
                values,
                dtype=torch.float32,
                device=model_device,
            )
            return (
                model(tensor)
                .argmax(dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

    for architecture, architecture_frame in successful.groupby(
        "architecture_id",
        observed=True,
        sort=True,
    ):
        model = models_by_architecture.get(str(architecture))
        if model is None:
            raise KeyError(f"Missing model for architecture {architecture}")
        for query_index, query_frame in architecture_frame.groupby(
            "query_index",
            observed=True,
            sort=True,
        ):
            for sigma in sigmas:
                sigma_value = float(sigma)
                sigma_key = int(round(sigma_value * 1.0e9))
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [int(seed), int(query_index), sigma_key]
                    )
                )
                standard_noise = rng.normal(
                    0.0,
                    1.0,
                    size=(int(samples_per_sigma), 32),
                ).astype(np.float32)
                for record in query_frame.to_dict(orient="records"):
                    x_cf = np.asarray(
                        [record[column] for column in cf_columns],
                        dtype=np.float32,
                    )
                    perturbed = x_cf[None, :] + sigma_value * standard_noise
                    predictions = predict(model, perturbed)
                    preserved = predictions == int(record["target_class"])
                    rows.append(
                        {
                            "architecture_id": str(architecture),
                            "query_index": int(query_index),
                            "method": str(record["method"]),
                            "target_class": int(record["target_class"]),
                            "sigma": sigma_value,
                            "n_perturbations": int(samples_per_sigma),
                            "empirical_target_rate": float(
                                np.mean(preserved)
                            ),
                            "empirical_all_preserved": bool(
                                np.all(preserved)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


class FlatPostHocLiRPACertifier:
    """Common target-class L-infinity certifier for flat binary networks."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str,
        method: str = "backward",
    ) -> None:
        from counterfactuals.methods.certcf import _strip_dropout_modules

        self.model = _strip_dropout_modules(model).eval().to(device)
        self.device = torch.device(device)
        self.method = str(method)

    def certified_batch(
        self,
        x: np.ndarray,
        target: int,
        radii: np.ndarray,
        *,
        norm: float = float("inf"),
    ) -> np.ndarray:
        """Return LiRPA certification outcomes for one target-class batch."""

        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        from certcf.certification.wrapping import WrappedModel

        centers = torch.as_tensor(
            np.asarray(x, dtype=np.float32),
            device=self.device,
        )
        radius_tensor = torch.as_tensor(
            np.asarray(radii, dtype=np.float32).reshape(-1, 1),
            device=self.device,
        )
        if float(norm) == float("inf"):
            perturbation = PerturbationLpNorm(
                norm=np.inf,
                x_L=centers - radius_tensor,
                x_U=centers + radius_tensor,
            )
        else:
            perturbation = PerturbationLpNorm(
                norm=float(norm),
                eps=radius_tensor,
            )
        bounded_x = BoundedTensor(centers, perturbation)
        wrapped = WrappedModel(
            self.model,
            int(target),
            self.device,
            n_labels=2,
        ).to(self.device).eval()
        bounded_model = BoundedModule(
            wrapped,
            bounded_x,
            device=self.device,
        )
        lower_bounds, _ = bounded_model.compute_bounds(
            x=(bounded_x,),
            method=self.method,
        )
        return (
            torch.all(lower_bounds >= 0.0, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

    def radii(
        self,
        x: np.ndarray,
        targets: np.ndarray,
        *,
        maximum: float,
        steps: int,
        norm: float = float("inf"),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Binary-search certified radii, returning values and cap flags."""

        values = np.asarray(x, dtype=np.float32)
        labels = np.asarray(targets, dtype=np.int64).reshape(-1)
        if values.ndim != 2 or values.shape[1] != 32:
            raise ValueError("Certification inputs must have shape [N, 32]")
        if len(values) != len(labels):
            raise ValueError("Certification inputs and targets must align")
        if maximum < 0.0 or steps < 0:
            raise ValueError("maximum and steps must be non-negative")

        output = np.zeros(len(values), dtype=float)
        capped = np.zeros(len(values), dtype=bool)
        for target in np.unique(labels):
            positions = np.flatnonzero(labels == int(target))
            centers = values[positions]
            zeros = np.zeros(len(positions), dtype=np.float32)
            center_ok = self.certified_batch(
                centers,
                int(target),
                zeros,
                norm=norm,
            )
            if not center_ok.any():
                continue
            active_local = np.flatnonzero(center_ok)
            active_centers = centers[active_local]
            low = np.zeros(len(active_local), dtype=np.float32)
            high = np.full(
                len(active_local),
                float(maximum),
                dtype=np.float32,
            )
            maximum_ok = self.certified_batch(
                active_centers,
                int(target),
                high,
                norm=norm,
            )
            low[maximum_ok] = float(maximum)
            capped[positions[active_local[maximum_ok]]] = True
            search = ~maximum_ok
            for _ in range(int(steps)):
                if not search.any():
                    break
                search_positions = np.flatnonzero(search)
                middle = 0.5 * (
                    low[search_positions] + high[search_positions]
                )
                middle_ok = self.certified_batch(
                    active_centers[search_positions],
                    int(target),
                    middle,
                    norm=norm,
                )
                accepted = search_positions[middle_ok]
                rejected = search_positions[~middle_ok]
                low[accepted] = middle[middle_ok]
                high[rejected] = middle[~middle_ok]
            output[positions[active_local]] = low
        return output, capped


def attach_certified_robustness(
    frame: pd.DataFrame,
    *,
    models_by_architecture: dict[str, torch.nn.Module],
    device: str,
    maximum: float = 5.0,
    steps: int = 14,
    lirpa_method: str = "backward",
    norms: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach common post-hoc LiRPA L1, L2, and Linf radii."""

    result = frame.copy()
    norm_map = CERTIFIED_NORMS if norms is None else dict(norms)
    for norm_label in norm_map:
        result[f"certified_{norm_label}_radius"] = np.nan
        result[f"certified_{norm_label}_at_cap"] = False
    cf_columns = _feature_columns("x_cf")
    for architecture, architecture_frame in result.groupby(
        "architecture_id",
        observed=True,
        sort=True,
    ):
        model = models_by_architecture.get(str(architecture))
        if model is None:
            raise KeyError(f"Missing model for architecture {architecture}")
        valid = (
            architecture_frame["paired_eligible"].astype(bool).to_numpy()
            & architecture_frame["success"].astype(bool).to_numpy()
            & np.isfinite(
                architecture_frame[cf_columns].to_numpy(dtype=float)
            ).all(axis=1)
        )
        if not valid.any():
            continue
        selected = architecture_frame.iloc[np.flatnonzero(valid)]
        certifier = FlatPostHocLiRPACertifier(
            model,
            device=device,
            method=lirpa_method,
        )
        for norm_label, norm_value in norm_map.items():
            radii, capped = certifier.radii(
                selected[cf_columns].to_numpy(dtype=np.float32),
                selected["target_class"].to_numpy(dtype=np.int64),
                maximum=float(maximum),
                steps=int(steps),
                norm=float(norm_value),
            )
            result.loc[
                selected.index,
                f"certified_{norm_label}_radius",
            ] = radii
            result.loc[
                selected.index,
                f"certified_{norm_label}_at_cap",
            ] = capped
    return result


def summarize_quality_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paper-aligned manifoldness and certified robustness."""

    successful = frame[
        frame["paired_eligible"].astype(bool) & frame["success"].astype(bool)
    ]
    if successful.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "n_success": ("query_index", "count"),
        "mean_log10_lof": ("log10_lof", "mean"),
        "mean_negative_lof_score": ("negative_lof_score", "mean"),
        "mean_isolation_forest_score": (
            "isolation_forest_score",
            "mean",
        ),
        "mean_certified_linf_radius": ("certified_linf_radius", "mean"),
        "median_certified_linf_radius": ("certified_linf_radius", "median"),
        "certified_linf_cap_fraction": ("certified_linf_at_cap", "mean"),
        "mean_certified_l2_radius": ("certified_l2_radius", "mean"),
        "median_certified_l2_radius": ("certified_l2_radius", "median"),
        "certified_l2_cap_fraction": ("certified_l2_at_cap", "mean"),
        "mean_certified_l1_radius": ("certified_l1_radius", "mean"),
        "median_certified_l1_radius": ("certified_l1_radius", "median"),
        "certified_l1_cap_fraction": ("certified_l1_at_cap", "mean"),
    }
    if "relative_proximity_ratio" in successful.columns:
        aggregations.update(
            {
                "mean_target_class_nn_l1": (
                    "target_class_nn_l1",
                    "mean",
                ),
                "mean_relative_proximity_ratio": (
                    "relative_proximity_ratio",
                    "mean",
                ),
            }
        )
    return (
        successful.groupby(
            [
                "architecture_id",
                "depth",
                "width",
                "parameter_count",
                "method",
            ],
            observed=True,
            sort=True,
        )
        .agg(**aggregations)
        .reset_index()
    )


def summarize_empirical_robustness(
    robustness: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the paper robustness score and strict preservation rate."""

    if robustness.empty:
        return pd.DataFrame()
    return (
        robustness.groupby(
            ["architecture_id", "method"],
            observed=True,
            sort=True,
        )
        .agg(
            empirical_robustness_pct=(
                "empirical_target_rate",
                lambda values: 100.0 * float(np.mean(values)),
            ),
            all_perturbations_preserved_pct=(
                "empirical_all_preserved",
                lambda values: 100.0 * float(np.mean(values)),
            ),
            robustness_grid_rows=("sigma", "size"),
        )
        .reset_index()
    )


def _diagnostics_fingerprint(
    frame: pd.DataFrame,
    manifest: dict,
    *,
    certified: bool,
    maximum: float,
    steps: int,
    lirpa_method: str,
) -> str:
    keys = (
        frame[
            [
                "architecture_id",
                "query_index",
                "method",
                "success",
                "target_class",
            ]
        ]
        .sort_values(["architecture_id", "query_index", "method"])
        .astype(str)
        .to_dict(orient="records")
    )
    payload = {
        "version": DIAGNOSTICS_VERSION,
        "run_fingerprint": manifest.get("run_fingerprint"),
        "keys": keys,
        "sigmas": PAPER_ROBUSTNESS_SIGMAS,
        "samples": PAPER_ROBUSTNESS_SAMPLES,
        "lof_neighbors": LOF_NEIGHBORS,
        "iforest_estimators": IFOREST_ESTIMATORS,
        "iforest_random_state": IFOREST_RANDOM_STATE,
        "certified": bool(certified),
        "maximum": float(maximum),
        "steps": int(steps),
        "lirpa_method": str(lirpa_method),
        "certified_norms": CERTIFIED_NORMS,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def compute_synthetic32_diagnostics(
    *,
    config_path: str | Path,
    result_dir: str | Path,
    device: str = "auto",
    certified: bool = True,
    certified_maximum: float = 5.0,
    certified_steps: int = 14,
    lirpa_method: str = "backward",
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute and cache paper-aligned diagnostics from saved CF artifacts."""

    from experiments.verix_certcf_mnist import _normalize_device
    from experiments.verix_certcf_synthetic32 import Synthetic32Runner

    queries, _, manifest = load_synthetic32_results(result_dir)
    vectors = load_counterfactual_vectors(result_dir, queries)
    fingerprint = _diagnostics_fingerprint(
        vectors,
        manifest,
        certified=certified,
        maximum=certified_maximum,
        steps=certified_steps,
        lirpa_method=lirpa_method,
    )
    cache_root = Path(result_dir) / "analysis"
    point_path = cache_root / "point_metrics.parquet"
    robustness_path = cache_root / "robustness_curves.parquet"
    metadata_path = cache_root / "metadata.json"
    if (
        not force
        and point_path.exists()
        and robustness_path.exists()
        and metadata_path.exists()
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return (
                pd.read_parquet(point_path),
                pd.read_parquet(robustness_path),
                metadata,
            )

    runner = Synthetic32Runner.from_yaml(config_path)
    source = runner._source_data()
    points = attach_manifoldness_metrics(
        vectors,
        x_train=source["x_train"],
        y_train=source["y_train"],
        include_relative_proximity=False,
    )
    resolved_device = _normalize_device(device)
    models: dict[str, torch.nn.Module] = {}
    for record in (
        points[["architecture_id", "depth", "width"]]
        .drop_duplicates()
        .to_dict(orient="records")
    ):
        cell = (int(record["depth"]), int(record["width"]))
        models[str(record["architecture_id"])] = runner._load_model(
            cell,
            resolved_device,
        )

    robustness = evaluate_paper_empirical_robustness(
        points,
        models_by_architecture=models,
        seed=int(runner.config["experiment"]["seed"]),
    )
    certification_seconds = 0.0
    if certified:
        started = time.perf_counter()
        points = attach_certified_robustness(
            points,
            models_by_architecture=models,
            device=resolved_device,
            maximum=certified_maximum,
            steps=certified_steps,
            lirpa_method=lirpa_method,
        )
        certification_seconds = time.perf_counter() - started
    else:
        for norm_label in CERTIFIED_NORMS:
            points[f"certified_{norm_label}_radius"] = np.nan
            points[f"certified_{norm_label}_at_cap"] = False

    metadata = {
        "status": "complete",
        "fingerprint": fingerprint,
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "run_fingerprint": manifest.get("run_fingerprint"),
        "device": resolved_device,
        "paper_robustness_sigmas": list(PAPER_ROBUSTNESS_SIGMAS),
        "paper_robustness_samples_per_sigma": PAPER_ROBUSTNESS_SAMPLES,
        "lof_neighbors": LOF_NEIGHBORS,
        "iforest_estimators": IFOREST_ESTIMATORS,
        "iforest_random_state": IFOREST_RANDOM_STATE,
        "relative_proximity_ratio_enabled": False,
        "certified_enabled": bool(certified),
        "certified_norms": list(CERTIFIED_NORMS),
        "certified_maximum": float(certified_maximum),
        "certified_binary_search_steps": int(certified_steps),
        "certified_lirpa_method": str(lirpa_method),
        "certification_wall_time_seconds": float(certification_seconds),
        "architectures": sorted(points["architecture_id"].unique().tolist()),
        "rows": int(len(points)),
    }
    _atomic_parquet(points, point_path)
    _atomic_parquet(robustness, robustness_path)
    _atomic_json(metadata, metadata_path)
    return points, robustness, metadata


__all__ = [
    "CERTIFIED_NORMS",
    "DIAGNOSTICS_VERSION",
    "FlatPostHocLiRPACertifier",
    "IFOREST_ESTIMATORS",
    "IFOREST_RANDOM_STATE",
    "LOF_NEIGHBORS",
    "METHOD_ORDER",
    "PAPER_ROBUSTNESS_SAMPLES",
    "PAPER_ROBUSTNESS_SIGMAS",
    "attach_certified_robustness",
    "attach_manifoldness_metrics",
    "completion_table",
    "compute_synthetic32_diagnostics",
    "evaluate_paper_empirical_robustness",
    "load_counterfactual_vectors",
    "load_synthetic32_results",
    "method_summary",
    "paired_query_table",
    "paired_summary",
    "summarize_empirical_robustness",
    "summarize_quality_metrics",
    "validate_query_rows",
]
