#!/usr/bin/env python3
"""Produce paper-ready statistics and plots for the seven-dataset top-k study."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = (
    "miss_probability",
    "gap_unconditional",
    "gap_conditional",
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _dataset_arrays(frame: pd.DataFrame, k_values: np.ndarray) -> dict[str, np.ndarray]:
    index = "query_position"

    def pivot(column: str) -> np.ndarray:
        table = frame.pivot(index=index, columns="k", values=column).reindex(columns=k_values)
        if table.isna().any().any():
            raise RuntimeError(f"incomplete top-k trajectories for {frame['dataset'].iloc[0]}")
        return table.to_numpy()

    exact = pivot("exact_success").astype(bool)
    strict = pivot("strict_success").astype(bool)
    recovered = pivot("exact_recovery").astype(bool)
    gap = pivot("gap_absolute").astype(float)
    shared = exact & strict
    finite_gap = np.where(shared, gap, 0.0)
    return {
        "exact": exact,
        "shared": shared,
        "miss": exact & ~recovered,
        "miss_shared": shared & ~recovered,
        "gap": finite_gap,
    }


def _ratios(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )


def _observed(arrays: dict[str, np.ndarray]) -> np.ndarray:
    exact = arrays["exact"]
    shared = arrays["shared"]
    miss_shared = arrays["miss_shared"]
    gap = arrays["gap"]
    return np.stack(
        [
            _ratios(arrays["miss"].sum(axis=0), exact.sum(axis=0)),
            _ratios((gap * shared).sum(axis=0), shared.sum(axis=0)),
            _ratios((gap * miss_shared).sum(axis=0), miss_shared.sum(axis=0)),
        ]
    )


def hierarchical_bootstrap(
    frames: list[pd.DataFrame],
    k_values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    chunk_size: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample queries within datasets, then macro-average dataset statistics."""

    rng = np.random.default_rng(seed)
    prepared = [_dataset_arrays(frame, k_values) for frame in frames]
    observed_by_dataset = np.stack([_observed(values) for values in prepared], axis=0)
    observed = np.nanmean(observed_by_dataset, axis=0)
    bootstrap = np.full((repetitions, len(METRICS), len(k_values)), np.nan)

    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        size = stop - start
        dataset_statistics = []
        for values in prepared:
            n_queries = values["exact"].shape[0]
            draws = rng.integers(0, n_queries, size=(size, n_queries))
            exact = values["exact"][draws]
            shared = values["shared"][draws]
            miss = values["miss"][draws]
            miss_shared = values["miss_shared"][draws]
            gap = values["gap"][draws]
            dataset_statistics.append(
                np.stack(
                    [
                        _ratios(miss.sum(axis=1), exact.sum(axis=1)),
                        _ratios((gap * shared).sum(axis=1), shared.sum(axis=1)),
                        _ratios(
                            (gap * miss_shared).sum(axis=1),
                            miss_shared.sum(axis=1),
                        ),
                    ],
                    axis=1,
                )
            )
        stacked = np.stack(dataset_statistics, axis=0)
        valid_counts = np.sum(np.isfinite(stacked), axis=0)
        bootstrap[start:stop] = np.divide(
            np.nansum(stacked, axis=0),
            valid_counts,
            out=np.full((size, len(METRICS), len(k_values)), np.nan),
            where=valid_counts > 0,
        )

    lower = np.nanquantile(bootstrap, 0.025, axis=0)
    upper = np.nanquantile(bootstrap, 0.975, axis=0)
    return observed, lower, upper


def exponential_fit(k: np.ndarray, values: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(values) & (values > 0)
    x = k[valid].astype(float)
    log_y = np.log(values[valid])
    slope, intercept = np.polyfit(x, log_y, 1)
    prediction = intercept + slope * x
    denominator = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((log_y - prediction) ** 2)) / denominator
    return {
        "amplitude": float(np.exp(intercept)),
        "exponent": float(slope),
        "r2_log": r2,
        "multiplicative_factor_per_k": float(np.exp(slope)),
        "relative_reduction_per_k": float(1.0 - np.exp(slope)),
    }


def build_plot_data(
    k_values: np.ndarray,
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {"k": k_values}
    for metric_index, metric in enumerate(METRICS):
        values[metric] = observed[metric_index]
        values[f"{metric}_lower"] = lower[metric_index]
        values[f"{metric}_upper"] = upper[metric_index]
        values[f"{metric}_errminus"] = observed[metric_index] - lower[metric_index]
        values[f"{metric}_errplus"] = upper[metric_index] - observed[metric_index]
    return pd.DataFrame(values)


def plot_results(plot_data: pd.DataFrame, models: dict, output: Path) -> None:
    colors = {
        "miss_probability": "#236B7A",
        "gap_unconditional": "#C76845",
        "gap_conditional": "#6EAA78",
    }
    k = plot_data["k"].to_numpy(float)
    dense_k = np.linspace(k.min(), k.max(), 200)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    metric = "miss_probability"
    axes[0].errorbar(
        k,
        plot_data[metric],
        yerr=np.vstack(
            [plot_data[f"{metric}_errminus"], plot_data[f"{metric}_errplus"]]
        ),
        fmt="o",
        color=colors[metric],
        capsize=2,
        label="Observed (95% CI)",
    )
    fit = models[metric]
    axes[0].plot(
        dense_k,
        fit["amplitude"] * np.exp(fit["exponent"] * dense_k),
        "--",
        color=colors[metric],
        label="Exponential fit",
    )
    axes[0].set(
        title="Probability of missing the atlas optimum",
        xlabel="Number of retrieved regions k",
        ylabel="Miss probability",
        xticks=k,
        yscale="log",
    )
    axes[0].legend(frameon=False)

    for metric, marker, label in (
        ("gap_unconditional", "s", "Unconditional"),
        ("gap_conditional", "^", "Conditional on a miss"),
    ):
        axes[1].errorbar(
            k,
            plot_data[metric],
            yerr=np.vstack(
                [plot_data[f"{metric}_errminus"], plot_data[f"{metric}_errplus"]]
            ),
            fmt=marker,
            color=colors[metric],
            capsize=2,
            label=label,
        )
    unconditional = models["gap_unconditional"]
    axes[1].plot(
        dense_k,
        unconditional["amplitude"]
        * np.exp(unconditional["exponent"] * dense_k),
        "--",
        color=colors["gap_unconditional"],
    )
    axes[1].axhline(
        models["gap_conditional"]["constant"],
        linestyle="--",
        color=colors["gap_conditional"],
    )
    axes[1].set(
        title="L1 optimality gaps",
        xlabel="Number of retrieved regions k",
        ylabel="Mean L1 gap",
        xticks=k,
        yscale="log",
    )
    axes[1].legend(frameon=False, loc="lower left")

    for axis in axes:
        axis.grid(True, which="both", color="#D9D9D9", linewidth=0.6)
        axis.set_facecolor("white")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/appendix_d_tabular/topk/topk_queries.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="results/appendix_d_tabular/topk/paper",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    datasets = sorted(frame["dataset"].unique())
    if len(datasets) != 7:
        raise RuntimeError(f"expected seven datasets, found {len(datasets)}")
    k_values = np.asarray(sorted(frame["k"].unique()), dtype=int)
    frames = [frame.loc[frame["dataset"] == dataset].copy() for dataset in datasets]
    observed, lower, upper = hierarchical_bootstrap(
        frames,
        k_values,
        repetitions=int(args.bootstrap_repetitions),
        seed=int(args.seed),
    )
    plot_data = build_plot_data(k_values, observed, lower, upper)

    miss_model = exponential_fit(k_values, observed[0])
    unconditional_model = exponential_fit(k_values, observed[1])
    conditional_diagnostic = exponential_fit(k_values, observed[2])
    models = {
        "miss_probability": miss_model,
        "gap_unconditional": unconditional_model,
        "gap_conditional": {
            "model": "constant",
            "constant": float(np.nanmean(observed[2])),
            "exponential_diagnostic": conditional_diagnostic,
        },
        "aggregation": "macro-average after within-dataset estimation",
        "bootstrap": {
            "type": "query resampling within each dataset",
            "repetitions": int(args.bootstrap_repetitions),
            "seed": int(args.seed),
            "confidence": 0.95,
        },
        "datasets": datasets,
    }

    table = plot_data[
        ["k", "miss_probability", "gap_unconditional", "gap_conditional"]
    ].copy()
    output = Path(args.output_dir)
    _atomic_csv(table, output / "topk_table.csv")
    _atomic_csv(plot_data, output / "topk_plot_data.csv")
    _atomic_json(models, output / "topk_models.json")
    plot_results(plot_data, models, output / "topk_ablation.png")
    print(json.dumps({"output_dir": str(output), "models": models}, indent=2))


if __name__ == "__main__":
    main()
