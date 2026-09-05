"""End-to-end stress test for CertCF's nearest-anchor top-k heuristic.

The binary data geometry and ReLU classifier are deliberately controlled, but
the atlas is not: radii come from Equation (1), affine bounds come from
auto-LiRPA, anchors are sampled by CertCF, and both online searches use the
production implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
DEFAULT_REPOSITORY = HERE.parents[1]


@dataclass(frozen=True)
class Configuration:
    alpha: float = 0.2
    decoy_distance: float = 9.0
    useful_distance: float = 9.1
    decoy_true_radius: float = 0.03
    useful_true_radius: float = 5.0
    source_std: float = 0.02
    target_std: float = 0.005
    n_source_support: int = 120
    n_target_per_component: int = 20
    n_queries: int = 100
    anchors_per_class: int = 24
    top_k: int = 5
    shrink_factor: float = 0.5
    max_shrinks: int = 8
    seed: int = 0


class UnionOfL1Balls(nn.Module):
    """Binary ReLU classifier whose target set is a union of L1 balls."""

    def __init__(self, centers: np.ndarray, radii: np.ndarray):
        super().__init__()
        centers = np.asarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        n_components, dimension = centers.shape
        self.n_components = int(n_components)

        # ReLU(x-c) + ReLU(c-x) computes each absolute coordinate offset.
        self.absolute_linear = nn.Linear(dimension, 2 * n_components * dimension)
        absolute_weight = np.zeros((2 * n_components * dimension, dimension), dtype=np.float32)
        absolute_bias = np.zeros(2 * n_components * dimension, dtype=np.float32)
        for component in range(n_components):
            for feature in range(dimension):
                base = 2 * (component * dimension + feature)
                absolute_weight[base, feature] = 1.0
                absolute_bias[base] = -centers[component, feature]
                absolute_weight[base + 1, feature] = -1.0
                absolute_bias[base + 1] = centers[component, feature]

        # Each output is radius_i - ||x-center_i||_1.
        self.margin_linear = nn.Linear(2 * n_components * dimension, n_components)
        margin_weight = np.zeros((n_components, 2 * n_components * dimension), dtype=np.float32)
        for component in range(n_components):
            start = 2 * component * dimension
            margin_weight[component, start : start + 2 * dimension] = -1.0

        self.output_linear = nn.Linear(1, 2)
        with torch.no_grad():
            self.absolute_linear.weight.copy_(torch.from_numpy(absolute_weight))
            self.absolute_linear.bias.copy_(torch.from_numpy(absolute_bias))
            self.margin_linear.weight.copy_(torch.from_numpy(margin_weight))
            self.margin_linear.bias.copy_(torch.from_numpy(radii))
            self.output_linear.weight.copy_(torch.tensor([[0.0], [1.0]]))
            self.output_linear.bias.zero_()

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        absolute_parts = self.relu(self.absolute_linear(x))
        component_margins = self.margin_linear(absolute_parts)
        union_margin = component_margins[:, 0:1]
        for component_index in range(1, self.n_components):
            candidate = component_margins[:, component_index : component_index + 1]
            union_margin = self.relu(union_margin - candidate) + candidate
        return self.output_linear(union_margin)


def component_geometry(config: Configuration) -> tuple[np.ndarray, np.ndarray, list[str]]:
    distance = config.decoy_distance
    decoys = np.array(
        [
            [-distance, 0.0],
            [0.0, distance],
            [0.0, -distance],
            [-0.5 * distance, 0.5 * distance],
            [-0.5 * distance, -0.5 * distance],
        ],
        dtype=np.float32,
    )
    useful = np.array([[config.useful_distance, 0.0]], dtype=np.float32)
    centers = np.concatenate((decoys, useful), axis=0)
    radii = np.array(
        [config.decoy_true_radius] * len(decoys) + [config.useful_true_radius],
        dtype=np.float32,
    )
    roles = [f"decoy_{index + 1}" for index in range(len(decoys))] + ["useful"]
    return centers, radii, roles


def sample_inside_l1_ball(
    rng: np.random.Generator,
    center: np.ndarray,
    n_samples: int,
    std: float,
    radius: float,
) -> np.ndarray:
    samples = rng.normal(center, std, size=(n_samples, len(center))).astype(np.float32)
    offsets = samples - center
    norms = np.sum(np.abs(offsets), axis=1)
    maximum = max(1e-6, 0.5 * float(radius))
    oversized = norms > maximum
    if np.any(oversized):
        offsets[oversized] *= (maximum / norms[oversized])[:, None]
    return (center + offsets).astype(np.float32)


def make_dataset(config: Configuration) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    centers, radii, roles = component_geometry(config)
    source_support = rng.normal(
        0.0,
        config.source_std,
        size=(config.n_source_support, 2),
    ).astype(np.float32)
    target_parts = [
        sample_inside_l1_ball(
            rng,
            center,
            config.n_target_per_component,
            config.target_std,
            float(radius),
        )
        for center, radius in zip(centers, radii)
    ]
    target_support = np.concatenate(target_parts, axis=0)
    x_support = np.concatenate((source_support, target_support), axis=0)
    y_support = np.concatenate(
        (
            np.zeros(len(source_support), dtype=np.int64),
            np.ones(len(target_support), dtype=np.int64),
        )
    )
    queries = rng.normal(0.0, config.source_std, size=(config.n_queries, 2)).astype(np.float32)
    return {
        "x_support": x_support,
        "y_support": y_support,
        "queries": queries,
        "centers": centers,
        "radii": radii,
        "roles": np.asarray(roles),
    }


def predict(model: nn.Module, points: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.as_tensor(points, dtype=torch.float32))
    return logits.argmax(dim=1).cpu().numpy().astype(np.int64)


def assign_roles(
    points: np.ndarray,
    centers: np.ndarray,
    roles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sum(np.abs(points[:, None, :] - centers[None, :, :]), axis=2)
    indices = np.argmin(distances, axis=1)
    return roles[indices], indices


def build_method(
    config: Configuration,
    model: nn.Module,
    data: dict[str, np.ndarray],
    repository: Path,
):
    source = str(repository / "src")
    if source not in sys.path:
        sys.path.insert(0, source)

    from certcf.eps_strategies import NearestOppositeClassClearanceStrategy
    from counterfactuals.methods.certcf import CertCF
    from counterfactuals.models.torch_model import TorchModelWrapper

    strategy = NearestOppositeClassClearanceStrategy(alpha=config.alpha, chunk_size=128)
    method = CertCF(
        model=TorchModelWrapper(model=model, device="cpu"),
        norm=1,
        distance_norm=1,
        lirpa_method="backward",
        eps_strategy=strategy,
        batch_size=32,
        default_query_method="nearest_anchor",
        query_k_candidates=config.top_k,
        solver_maxiter=500,
        cvxpy_solvers=["CLARABEL"],
        cvxpy_solver_options={"CLARABEL": {}},
        cvxpy_accept_statuses={"CLARABEL": ["optimal"]},
        classification_margin=0.0,
        adaptive_eps=True,
        adaptive_eps_shrink_factor=config.shrink_factor,
        adaptive_eps_max_shrinks=config.max_shrinks,
        adaptive_eps_min=1e-6,
        adaptive_eps_center_tol=1e-6,
        adaptive_eps_binary_search_steps=0,
        sparsity_penalty="none",
        k_per_class=config.anchors_per_class,
        subsample_method="random",
        random_seed=config.seed,
        eps_reference_scope="full_support",
    )
    started = time.perf_counter()
    method.fit(data["x_support"], data["y_support"])
    return method, time.perf_counter() - started


def atlas_diagnostics(
    method,
    data: dict[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    bounds = method.atlas.bounds[1]
    roles, role_indices = assign_roles(bounds["X"], data["centers"], data["roles"])
    rows: list[dict[str, object]] = []
    for index in range(len(bounds["X"])):
        initial = float(bounds["eps_initial"][index])
        final = float(bounds["eps"][index])
        rows.append(
            {
                "anchor_index": index,
                "role": str(roles[index]),
                "component_index": int(role_indices[index]),
                "anchor_x": float(bounds["X"][index, 0]),
                "anchor_y": float(bounds["X"][index, 1]),
                "initial_epsilon": initial,
                "final_epsilon": final,
                "epsilon_ratio": final / initial,
                "n_shrinks": int(bounds["adaptive_eps_n_shrinks"][index]),
                "center_slack": float(bounds["adaptive_eps_center_slack"][index]),
            }
        )

    aggregate: dict[str, object] = {
        "target_anchor_count": len(rows),
        "role_counts": {},
        "role_epsilon": {},
    }
    for role in sorted(set(roles.tolist())):
        selected = [row for row in rows if row["role"] == role]
        aggregate["role_counts"][role] = len(selected)
        aggregate["role_epsilon"][role] = {
            "mean_initial": float(np.mean([row["initial_epsilon"] for row in selected])),
            "mean_final": float(np.mean([row["final_epsilon"] for row in selected])),
            "mean_ratio": float(np.mean([row["epsilon_ratio"] for row in selected])),
            "mean_shrinks": float(np.mean([row["n_shrinks"] for row in selected])),
        }
    return rows, aggregate


def evaluate_queries(
    method,
    data: dict[str, np.ndarray],
    top_k: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    bounds = method.atlas.bounds[1]
    anchor_roles, _ = assign_roles(bounds["X"], data["centers"], data["roles"])
    rows: list[dict[str, object]] = []
    for query_index, query in enumerate(data["queries"]):
        anchor_distances = np.sum(np.abs(bounds["X"] - query[None, :]), axis=1)
        order = np.argsort(anchor_distances)
        approximate = method.atlas.find_counterfactual(
            query,
            target_class=1,
            method="nearest_anchor",
            query_k_candidates=top_k,
        )
        exhaustive = method.atlas.find_counterfactual(query, target_class=1, method="sorted")
        approximate_prediction = (
            -1
            if approximate.x_cf is None
            else int(method.model.predict(np.asarray(approximate.x_cf).reshape(1, -1))[0])
        )
        exhaustive_prediction = (
            -1
            if exhaustive.x_cf is None
            else int(method.model.predict(np.asarray(exhaustive.x_cf).reshape(1, -1))[0])
        )
        gap = float(approximate.distance - exhaustive.distance)
        exhaustive_index = -1 if exhaustive.anchor_idx is None else int(exhaustive.anchor_idx)
        approximate_index = -1 if approximate.anchor_idx is None else int(approximate.anchor_idx)
        exhaustive_rank = (
            -1 if exhaustive_index < 0 else int(np.where(order == exhaustive_index)[0][0] + 1)
        )
        rows.append(
            {
                "query_index": query_index,
                "query_x": float(query[0]),
                "query_y": float(query[1]),
                "top_k_success": bool(approximate.success),
                "exhaustive_success": bool(exhaustive.success),
                "top_k_distance": float(approximate.distance),
                "exhaustive_distance": float(exhaustive.distance),
                "top_k_prediction": approximate_prediction,
                "exhaustive_prediction": exhaustive_prediction,
                "top_k_cf_x": np.nan if approximate.x_cf is None else float(approximate.x_cf[0]),
                "top_k_cf_y": np.nan if approximate.x_cf is None else float(approximate.x_cf[1]),
                "exhaustive_cf_x": np.nan if exhaustive.x_cf is None else float(exhaustive.x_cf[0]),
                "exhaustive_cf_y": np.nan if exhaustive.x_cf is None else float(exhaustive.x_cf[1]),
                "absolute_gap": gap,
                "distance_ratio": float(approximate.distance / exhaustive.distance),
                "failure": bool(gap > 1e-5),
                "fallback_used": bool(approximate.profiling["nearest_anchor_fallback_used"]),
                "top_k_anchor_index": approximate_index,
                "top_k_anchor_role": (
                    "none" if approximate_index < 0 else str(anchor_roles[approximate_index])
                ),
                "exhaustive_anchor_index": exhaustive_index,
                "exhaustive_anchor_role": (
                    "none" if exhaustive_index < 0 else str(anchor_roles[exhaustive_index])
                ),
                "exhaustive_anchor_rank": exhaustive_rank,
                "nearest_anchor_role": str(anchor_roles[int(order[0])]),
            }
        )

    failures = np.array([row["failure"] for row in rows], dtype=bool)
    gaps = np.array([row["absolute_gap"] for row in rows], dtype=float)
    ratios = np.array([row["distance_ratio"] for row in rows], dtype=float)
    aggregate = {
        "n_queries": len(rows),
        "failure_rate": float(np.mean(failures)),
        "fallback_rate": float(np.mean([row["fallback_used"] for row in rows])),
        "top_k_target_validity": float(np.mean([row["top_k_prediction"] == 1 for row in rows])),
        "exhaustive_target_validity": float(
            np.mean([row["exhaustive_prediction"] == 1 for row in rows])
        ),
        "mean_top_k_distance": float(np.mean([row["top_k_distance"] for row in rows])),
        "mean_exhaustive_distance": float(
            np.mean([row["exhaustive_distance"] for row in rows])
        ),
        "mean_gap": float(np.mean(gaps)),
        "mean_gap_on_failure": float(np.mean(gaps[failures])) if np.any(failures) else 0.0,
        "minimum_gap_on_failure": float(np.min(gaps[failures])) if np.any(failures) else 0.0,
        "mean_ratio": float(np.mean(ratios)),
        "mean_ratio_on_failure": (
            float(np.mean(ratios[failures])) if np.any(failures) else 1.0
        ),
        "useful_exhaustive_rate": float(
            np.mean([row["exhaustive_anchor_role"] == "useful" for row in rows])
        ),
        "exhaustive_rank_mean": float(np.mean([row["exhaustive_anchor_rank"] for row in rows])),
    }
    return rows, aggregate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(
    data: dict[str, np.ndarray],
    anchor_rows: list[dict[str, object]],
    query_rows: list[dict[str, object]],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    source = data["x_support"][data["y_support"] == 0]
    target = data["x_support"][data["y_support"] == 1]
    ax.scatter(source[:, 0], source[:, 1], s=12, alpha=0.35, color="#37474f", label="source support")
    ax.scatter(target[:, 0], target[:, 1], s=12, alpha=0.28, color="#c96a19", label="target support")
    anchors = np.array([[row["anchor_x"], row["anchor_y"]] for row in anchor_rows])
    useful = np.array([row["role"] == "useful" for row in anchor_rows])
    ax.scatter(
        anchors[~useful, 0],
        anchors[~useful, 1],
        s=55,
        color="#d26a00",
        edgecolor="white",
        label="sampled decoy anchors",
    )
    if np.any(useful):
        ax.scatter(
            anchors[useful, 0],
            anchors[useful, 1],
            s=75,
            marker="*",
            color="#087f8c",
            edgecolor="white",
            label="sampled useful anchors",
        )
    failures = np.array([row["failure"] for row in query_rows])
    queries = data["queries"]
    ax.scatter(queries[~failures, 0], queries[~failures, 1], s=18, color="#4c78a8", label="top-k exact query")
    ax.scatter(queries[failures, 0], queries[failures, 1], s=20, color="#d62728", label="top-k failure query")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("End-to-end synthetic top-k stress test")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(config: Configuration, repository: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    data = make_dataset(config)
    centers, radii, _ = component_geometry(config)
    model = UnionOfL1Balls(centers, radii).eval()

    support_predictions = predict(model, data["x_support"])
    query_predictions = predict(model, data["queries"])
    if not np.array_equal(support_predictions, data["y_support"]):
        mismatch = int(np.count_nonzero(support_predictions != data["y_support"]))
        raise RuntimeError(f"Teacher misclassifies {mismatch} support points")
    if np.any(query_predictions != 0):
        raise RuntimeError("At least one query is already predicted as the target class")

    method, build_time_s = build_method(config, model, data, repository)
    anchor_rows, anchor_summary = atlas_diagnostics(method, data)
    if anchor_summary["role_counts"].get("useful", 0) == 0:
        raise RuntimeError("Random anchor sampling selected no useful target anchor")
    query_rows, query_summary = evaluate_queries(method, data, config.top_k)

    result = {
        "configuration": asdict(config),
        "build_time_s": build_time_s,
        "support_prediction_accuracy": float(np.mean(support_predictions == data["y_support"])),
        "query_source_rate": float(np.mean(query_predictions == 0)),
        "atlas": anchor_summary,
        "queries": query_summary,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "anchors.csv", anchor_rows)
    write_csv(output / "queries.csv", query_rows)
    np.savez(
        output / "data.npz",
        x_support=data["x_support"],
        y_support=data["y_support"],
        queries=data["queries"],
        centers=data["centers"],
        radii=data["radii"],
    )
    plot_case(data, anchor_rows, query_rows, output / "case.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "single_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--decoy-distance", type=float, default=9.0)
    parser.add_argument("--useful-distance", type=float, default=9.1)
    parser.add_argument("--decoy-radius", type=float, default=0.03)
    parser.add_argument("--useful-radius", type=float, default=5.0)
    parser.add_argument("--source-std", type=float, default=0.02)
    parser.add_argument("--target-std", type=float, default=0.005)
    parser.add_argument("--anchors-per-class", type=int, default=24)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    config = Configuration(
        seed=args.seed,
        alpha=args.alpha,
        decoy_distance=args.decoy_distance,
        useful_distance=args.useful_distance,
        decoy_true_radius=args.decoy_radius,
        useful_true_radius=args.useful_radius,
        source_std=args.source_std,
        target_std=args.target_std,
        anchors_per_class=args.anchors_per_class,
        n_queries=args.queries,
    )
    result = run(config, args.repository.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
