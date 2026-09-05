"""Plot aggregate diagnostics for the saved top-k counterexample runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from end_to_end import HERE


DECOY = "#D55E00"
USEFUL = "#0072B2"
INK = "#263238"
GRID = "#CFD8DC"


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def jitter(rng: np.random.Generator, center: float, n: int) -> np.ndarray:
    return center + rng.uniform(-0.10, 0.10, size=n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=HERE / "outputs" / "seeds")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "outputs" / "counterexample_diagnostics.png",
    )
    args = parser.parse_args()

    anchor_rows = read_rows(sorted(args.input.glob("seed_*/anchors.csv")))
    query_rows = read_rows(sorted(args.input.glob("seed_*/queries.csv")))
    if not anchor_rows or not query_rows:
        raise RuntimeError(f"Missing anchor or query CSV files under {args.input}")

    decoy = [row for row in anchor_rows if row["role"].startswith("decoy")]
    useful = [row for row in anchor_rows if row["role"] == "useful"]
    groups = [
        np.array([float(row["initial_epsilon"]) for row in decoy]),
        np.array([float(row["final_epsilon"]) for row in decoy]),
        np.array([float(row["initial_epsilon"]) for row in useful]),
        np.array([float(row["final_epsilon"]) for row in useful]),
    ]

    top_distance = np.array([float(row["top_k_distance"]) for row in query_rows])
    exhaustive_distance = np.array([float(row["exhaustive_distance"]) for row in query_rows])
    failures = np.array([row["failure"] == "True" for row in query_rows])
    ranks = np.array([int(row["exhaustive_anchor_rank"]) for row in query_rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)

    # A. Equation (1) radii versus final certified radii.
    ax = axes[0]
    positions = np.arange(1, 5)
    colors = [DECOY, DECOY, USEFUL, USEFUL]
    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "lw": 1.8},
        whiskerprops={"color": INK},
        capprops={"color": INK},
        boxprops={"edgecolor": INK},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    rng = np.random.default_rng(42)
    for position, values, color in zip(positions, groups, colors):
        ax.scatter(
            jitter(rng, float(position), len(values)),
            values,
            s=9,
            color=color,
            alpha=0.22,
            edgecolor="none",
            zorder=1,
        )
    ax.annotate(
        "$\\times 1/128$\n(7 shrinks)",
        xy=(2, float(np.median(groups[1]))),
        xytext=(2.45, 0.075),
        ha="center",
        color=DECOY,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": DECOY, "lw": 1.2},
    )
    ax.annotate(
        "unchanged",
        xy=(4, float(np.median(groups[3]))),
        xytext=(3.45, 0.28),
        ha="center",
        color=USEFUL,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": USEFUL, "lw": 1.2},
    )
    ax.set_yscale("log")
    ax.set_ylabel("$L_1$ radius (log scale)")
    ax.set_xticks(positions, ["Decoy\nEq. (1)", "Decoy\nfinal", "Useful\nEq. (1)", "Useful\nfinal"])
    ax.set_title("A. Certification creates the reach mismatch", loc="left", fontweight="bold")
    ax.grid(axis="y", which="both", color=GRID, alpha=0.55)

    # B. Per-query comparison with the atlas optimum.
    ax = axes[1]
    ax.scatter(
        exhaustive_distance[failures],
        top_distance[failures],
        s=16,
        color=DECOY,
        alpha=0.28,
        edgecolor="none",
        label="top-5 miss",
    )
    ax.scatter(
        exhaustive_distance[~failures],
        top_distance[~failures],
        s=65,
        marker="*",
        color=USEFUL,
        edgecolor="white",
        linewidth=0.7,
        label="top-5 exact",
        zorder=4,
    )
    lower = min(float(exhaustive_distance.min()), float(top_distance.min())) - 0.05
    upper = max(float(exhaustive_distance.max()), float(top_distance.max())) + 0.05
    ax.plot([lower, upper], [lower, upper], color=INK, ls="--", lw=1.1, label="$D_5=D^\\star$")
    ax.text(
        0.97,
        0.05,
        f"{failures.sum()}/{len(failures)} misses\nmean gap = {(top_distance-exhaustive_distance).mean():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "alpha": 0.95},
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Exhaustive-atlas distance $D^\\star$")
    ax.set_ylabel("Top-5 distance $D_5$")
    ax.set_title("B. Top-5 is valid but systematically farther", loc="left", fontweight="bold")
    ax.grid(color=GRID, alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.2)

    # C. Rank at which exhaustive search finds its optimal anchor.
    ax = axes[2]
    possible = np.arange(1, int(ranks.max()) + 1)
    counts = np.array([(ranks == rank).sum() for rank in possible])
    bar_colors = [USEFUL if rank <= 5 else DECOY for rank in possible]
    ax.bar(possible, counts, color=bar_colors, width=0.82, alpha=0.82)
    ax.axvspan(0.5, 5.5, color=USEFUL, alpha=0.09)
    ax.axvline(5.5, color=INK, ls="--", lw=1.2)
    ax.text(3, max(counts) * 0.96, "searched by top-5", ha="center", va="top", color=USEFUL, fontweight="bold")
    ax.text(
        0.97,
        0.90,
        f"{(ranks > 5).sum()}/{len(ranks)} optima\noutside top-5",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GRID, "alpha": 0.95},
    )
    ax.set_xlabel("Rank of exhaustive-optimal anchor")
    ax.set_ylabel("Queries")
    ax.set_title("C. The useful anchor is normally ranked late", loc="left", fontweight="bold")
    ax.set_xticks([1, 5, 10, 15, 20, 24])
    ax.grid(axis="y", color=GRID, alpha=0.45)

    fig.suptitle(
        "Aggregate diagnostics for the controlled top-5 counterexample",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Ten seeds, 1,000 queries. All top-5 and exhaustive outputs are target-valid; fallback is never activated.",
        ha="center",
        va="top",
        color="#52636A",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
