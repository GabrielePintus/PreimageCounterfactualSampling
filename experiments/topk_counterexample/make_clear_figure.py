"""Create a publication-style explanation of the end-to-end counterexample."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from end_to_end import (
    DEFAULT_REPOSITORY,
    HERE,
    Configuration,
    UnionOfL1Balls,
    build_method,
    component_geometry,
    make_dataset,
)


DECOY = "#D55E00"
USEFUL = "#0072B2"
QUERY = "#202124"
SOURCE = "#6C7A80"
PALE = "#B8C4C9"


def l1_diamond(center: np.ndarray, radius: float) -> np.ndarray:
    x, y = map(float, center)
    return np.array([[x - radius, y], [x, y + radius], [x + radius, y], [x, y - radius]])


def draw_distance_row(
    ax,
    *,
    y: float,
    color: str,
    label: str,
    center_distance: float,
    initial_epsilon: float,
    projection_distance: float,
    annotation: str,
) -> None:
    initial_left = max(0.0, center_distance - initial_epsilon)
    initial_right = center_distance + initial_epsilon
    ax.plot(
        [initial_left, initial_right],
        [y, y],
        color=PALE,
        lw=13,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        [projection_distance, center_distance],
        [y, y],
        color=color,
        lw=13,
        solid_capstyle="round",
        zorder=2,
    )
    ax.scatter(center_distance, y, s=145, color=color, edgecolor="white", linewidth=1.5, zorder=4)
    ax.scatter(projection_distance, y, s=110, marker="D", color=color, edgecolor="white", linewidth=1.3, zorder=4)
    ax.annotate(
        "",
        xy=(projection_distance, y - 0.17),
        xytext=(0.0, y - 0.17),
        arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.8},
    )
    ax.text(
        projection_distance / 2,
        y - 0.29,
        f"projection = {projection_distance:.2f}",
        ha="center",
        va="top",
        color=color,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.text(-0.12, y, label, ha="right", va="center", color=color, fontsize=11, fontweight="bold")
    ax.text(
        center_distance,
        y + 0.20,
        f"anchor  {center_distance:.2f}",
        ha="center",
        va="bottom",
        color=color,
        fontsize=9.2,
        fontweight="bold",
    )
    ax.text(0.03, y + 0.20, annotation, ha="left", va="bottom", fontsize=9.1, color="#4A555A")
    ax.text(
        initial_right,
        y - 0.02,
        f"  $\\epsilon_0={initial_epsilon:.2f}$",
        ha="left",
        va="center",
        color="#66777E",
        fontsize=8.8,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs" / "counterexample_explained.png",
    )
    args = parser.parse_args()

    config = Configuration(seed=args.seed)
    data = make_dataset(config)
    centers, radii, _ = component_geometry(config)
    model = UnionOfL1Balls(centers, radii).eval()
    method, _ = build_method(config, model, data, args.repository.resolve())

    bounds = method.atlas.bounds[1]
    query = data["queries"][0]
    anchor_distances = np.sum(np.abs(bounds["X"] - query[None, :]), axis=1)
    order = np.argsort(anchor_distances)
    top_indices = order[: config.top_k]
    top_result = method.atlas.find_counterfactual(
        query,
        target_class=1,
        method="nearest_anchor",
        query_k_candidates=config.top_k,
    )
    exact_result = method.atlas.find_counterfactual(query, target_class=1, method="sorted")
    top_index = int(top_result.anchor_idx)
    exact_index = int(exact_result.anchor_idx)
    exact_rank = int(np.where(order == exact_index)[0][0] + 1)

    top_center_distance = float(anchor_distances[top_index])
    exact_center_distance = float(anchor_distances[exact_index])
    top_distance = float(top_result.distance)
    exact_distance = float(exact_result.distance)
    gap = top_distance - exact_distance
    penalty = 100.0 * (top_distance / exact_distance - 1.0)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "font.family": "DejaVu Sans",
        }
    )
    fig = plt.figure(figsize=(14.2, 6.3), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0])
    ax_geo = fig.add_subplot(grid[0, 0])
    ax_dist = fig.add_subplot(grid[0, 1])

    # Actual model target preimage: the boundary is the union of six diamonds.
    for index, (center, radius) in enumerate(zip(centers, radii)):
        color = USEFUL if index == len(centers) - 1 else DECOY
        alpha = 0.16 if index == len(centers) - 1 else 0.35
        ax_geo.add_patch(
            Polygon(
                l1_diamond(center, float(radius)),
                closed=True,
                facecolor=color,
                edgecolor=color,
                lw=1.5,
                alpha=alpha,
                zorder=1,
            )
        )

    source = data["x_support"][data["y_support"] == 0]
    ax_geo.scatter(source[:, 0], source[:, 1], s=10, color=SOURCE, alpha=0.35, zorder=2)
    ax_geo.scatter(query[0], query[1], s=180, marker="*", color=QUERY, edgecolor="white", linewidth=1.2, zorder=8)
    ax_geo.text(query[0] + 0.18, query[1] - 0.48, "query", color=QUERY, fontsize=10.5, fontweight="bold")

    target_anchors = bounds["X"]
    ax_geo.scatter(target_anchors[:, 0], target_anchors[:, 1], s=20, color="#9BABB1", alpha=0.65, zorder=3)
    top_groups: dict[int, list[int]] = {}
    for rank, index in enumerate(top_indices, start=1):
        point = target_anchors[int(index)]
        ax_geo.scatter(point[0], point[1], s=100, color=DECOY, edgecolor="white", linewidth=1.4, zorder=6)
        component = int(np.argmin(np.sum(np.abs(centers - point[None, :]), axis=1)))
        top_groups.setdefault(component, []).append(rank)

    for component, ranks in top_groups.items():
        point = centers[component]
        rank_text = ", ".join(map(str, ranks))
        ax_geo.annotate(
            f"top-5 ranks {rank_text}",
            xy=point,
            xytext=(15, 16),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=DECOY,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": DECOY, "lw": 1.1},
            arrowprops={"arrowstyle": "->", "color": DECOY, "lw": 1.1},
            zorder=9,
        )

    exact_anchor = target_anchors[exact_index]
    ax_geo.scatter(
        exact_anchor[0],
        exact_anchor[1],
        s=180,
        marker="*",
        color=USEFUL,
        edgecolor="white",
        linewidth=1.5,
        zorder=7,
    )
    ax_geo.annotate(
        f"useful anchor\nrank {exact_rank}",
        xy=exact_anchor,
        xytext=(-42, 30),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=USEFUL,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": USEFUL, "lw": 1.4},
    )

    top_cf = np.asarray(top_result.x_cf)
    exact_cf = np.asarray(exact_result.x_cf)
    ax_geo.plot([query[0], top_cf[0]], [query[1], top_cf[1]], color=DECOY, lw=2.0, ls=(0, (4, 3)), zorder=4)
    ax_geo.plot([query[0], exact_cf[0]], [query[1], exact_cf[1]], color=USEFUL, lw=2.3, zorder=5)
    ax_geo.scatter(top_cf[0], top_cf[1], s=75, marker="D", color=DECOY, edgecolor="white", linewidth=1.2, zorder=7)
    ax_geo.scatter(exact_cf[0], exact_cf[1], s=75, marker="D", color=USEFUL, edgecolor="white", linewidth=1.2, zorder=7)
    ax_geo.annotate("top-5 CF", xy=top_cf, xytext=(10, -18), textcoords="offset points", color=DECOY, fontweight="bold")
    ax_geo.annotate("exhaustive CF", xy=exact_cf, xytext=(8, 10), textcoords="offset points", color=USEFUL, fontweight="bold")

    ax_geo.text(
        10.7,
        -3.4,
        "model target-class region\n(useful component)",
        ha="center",
        va="center",
        color=USEFUL,
        fontsize=9.3,
        fontweight="bold",
    )

    # The five decoy decision boundaries are too small to read at global scale.
    # Show one at its true scale in an inset rather than enlarging it in Panel A.
    zoom_component = 1
    zoom_center = centers[zoom_component]
    zoom = ax_geo.inset_axes([0.72, 0.70, 0.25, 0.25])
    zoom.add_patch(
        Polygon(
            l1_diamond(zoom_center, float(radii[zoom_component])),
            closed=True,
            facecolor=DECOY,
            edgecolor=DECOY,
            lw=1.6,
            alpha=0.28,
        )
    )
    target_support = data["x_support"][data["y_support"] == 1]
    target_near = np.sum(np.abs(target_support - zoom_center[None, :]), axis=1) < 0.08
    anchor_near = np.sum(np.abs(target_anchors - zoom_center[None, :]), axis=1) < 0.08
    zoom.scatter(target_support[target_near, 0], target_support[target_near, 1], s=10, color="#9BABB1", alpha=0.7)
    zoom.scatter(target_anchors[anchor_near, 0], target_anchors[anchor_near, 1], s=38, color=DECOY, edgecolor="white", linewidth=0.8)
    zoom.set_xlim(float(zoom_center[0]) - 0.055, float(zoom_center[0]) + 0.055)
    zoom.set_ylim(float(zoom_center[1]) - 0.055, float(zoom_center[1]) + 0.055)
    zoom.set_aspect("equal", adjustable="box")
    zoom.set_xticks([])
    zoom.set_yticks([])
    zoom.set_title("decoy model boundary (zoom)", fontsize=7.8, color=DECOY, fontweight="bold")
    for spine in zoom.spines.values():
        spine.set_color(DECOY)
        spine.set_linewidth(1.0)

    ax_geo.text(
        -8.7,
        -10.0,
        "Five tiny target islands contain the nearest anchors.\n"
        "The large target component has a slightly farther center,\n"
        "but its certified region reaches much closer to the query.",
        ha="left",
        va="bottom",
        fontsize=9.6,
        bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#CFD8DC", "alpha": 0.95},
    )
    ax_geo.set_xlim(-10.2, 14.8)
    ax_geo.set_ylim(-10.5, 10.5)
    ax_geo.set_aspect("equal", adjustable="box")
    ax_geo.set_xlabel("feature 1")
    ax_geo.set_ylabel("feature 2")
    ax_geo.set_title("A. Anchor ranking sees centers, not certified reach", loc="left", fontweight="bold")
    ax_geo.grid(alpha=0.13)
    ax_geo.legend(
        handles=[
            Patch(facecolor=DECOY, edgecolor=DECOY, alpha=0.28, label="model target islands (decoys)"),
            Patch(facecolor=USEFUL, edgecolor=USEFUL, alpha=0.18, label="model target region (useful)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=DECOY, markeredgecolor="white", markersize=9, label="top-5 anchors"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=USEFUL, markeredgecolor="white", markersize=13, label="exhaustive-optimal anchor"),
            Line2D([0], [0], marker="D", color=DECOY, lw=2, ls=(0, (4, 3)), markerfacecolor=DECOY, markersize=6, label="top-5 projection"),
            Line2D([0], [0], marker="D", color=USEFUL, lw=2, markerfacecolor=USEFUL, markersize=6, label="exhaustive projection"),
        ],
        loc="upper left",
        fontsize=8.3,
        framealpha=0.95,
    )

    draw_distance_row(
        ax_dist,
        y=1.08,
        color=DECOY,
        label="Top-5",
        center_distance=top_center_distance,
        initial_epsilon=float(bounds["eps_initial"][top_index]),
        projection_distance=top_distance,
        annotation=f"LiRPA: 7 shrinks  $\\rightarrow\\;\\epsilon_{{final}}={float(bounds['eps'][top_index]):.3f}$",
    )
    draw_distance_row(
        ax_dist,
        y=0.32,
        color=USEFUL,
        label="Exhaustive",
        center_distance=exact_center_distance,
        initial_epsilon=float(bounds["eps_initial"][exact_index]),
        projection_distance=exact_distance,
        annotation=f"LiRPA: no shrink  $\\rightarrow\\;\\epsilon_{{final}}={float(bounds['eps'][exact_index]):.2f}$",
    )
    ax_dist.axvline(0, color=QUERY, lw=1.5, zorder=0)
    ax_dist.scatter(0, 0.70, marker="*", s=180, color=QUERY, edgecolor="white", linewidth=1.2, zorder=5)
    ax_dist.text(0.06, 0.70, "query", ha="left", va="center", color=QUERY, fontweight="bold")
    ax_dist.annotate(
        "",
        xy=(exact_distance, -0.13),
        xytext=(top_distance, -0.13),
        arrowprops={"arrowstyle": "<->", "color": QUERY, "lw": 1.8},
    )
    ax_dist.text(
        (top_distance + exact_distance) / 2,
        -0.23,
        f"top-5 penalty: +{gap:.2f}  (+{penalty:.1f}%)",
        ha="center",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color=QUERY,
    )
    ax_dist.text(
        0.0,
        1.52,
        "Pale bar: initial Eq. (1) ball     Colored bar: certified query-facing reach",
        ha="left",
        va="center",
        fontsize=9.2,
        color="#52636A",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F5F8F9", "ec": "#D5DEE1"},
    )
    ax_dist.set_xlim(-0.25, 11.25)
    ax_dist.set_ylim(-0.42, 1.75)
    ax_dist.set_yticks([])
    ax_dist.set_xlabel("$L_1$ distance from the query")
    ax_dist.set_title("B. Projection reverses the anchor ordering", loc="left", fontweight="bold")
    ax_dist.grid(axis="x", alpha=0.17)
    for side in ("left", "right", "top"):
        ax_dist.spines[side].set_visible(False)

    fig.suptitle(
        "Why nearest-anchor top-5 misses a better certified counterfactual",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        f"Actual CertCF run (seed {args.seed}, $L_1$, $\\alpha=0.2$). "
        "Both returned counterfactuals are target-valid; top-5 does not trigger fallback.",
        ha="center",
        va="top",
        fontsize=10,
        color="#4E5D63",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
