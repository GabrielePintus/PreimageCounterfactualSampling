"""Reusable plot builders for benchmark analysis notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from .style import method_color


def plot_validity_bar(
    summary_df: pd.DataFrame,
    *,
    palette: dict[str, tuple[float, float, float]],
    xlim: tuple[float, float] | None = None,
):
    """Plot a horizontal validity bar chart from a summary table."""
    fig, ax = plt.subplots()
    labels = summary_df.index.tolist()
    values = summary_df["validity_pct"].to_numpy(dtype=float)
    bars = ax.barh(
        labels[::-1],
        values[::-1],
        color=[method_color(label, palette) for label in labels[::-1]],
    )
    ax.set_xlabel("Validity (%)")
    ax.set_title("Counterfactual Validity by Method")
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0.0, 105.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
    for bar, value in zip(bars, values[::-1]):
        ax.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    return fig, ax


def plot_metric_boxplots(
    df: pd.DataFrame,
    metrics: list[tuple[str, str]],
    *,
    method_order: list[str],
    palette: dict[str, tuple[float, float, float]],
    method_col: str = "method_label",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Plot one boxplot panel per metric."""
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize or (6 * len(metrics), 4.5),
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, metrics):
        data_by_method = [
            df[df[method_col] == method][metric].dropna().to_numpy()
            for method in method_order
        ]
        bp = ax.boxplot(data_by_method, vert=True, patch_artist=True, tick_labels=method_order)
        for patch, method in zip(bp["boxes"], method_order):
            patch.set_facecolor(method_color(method, palette))
            patch.set_alpha(0.65)
        ax.set_title(label)
        ax.set_ylabel(label.split("\n")[0])
        ax.set_xticklabels(method_order, rotation=30, ha="right")

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_heatmap(
    table: pd.DataFrame,
    *,
    title: str,
    cmap: str = "YlGn",
    fmt: str = ".1f",
    cbar_label: str | None = None,
    annot: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot a dataframe heatmap with shared defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    heatmap_kwargs = {
        "annot": annot,
        "fmt": fmt,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "ax": ax,
    }
    if cbar_label:
        heatmap_kwargs["cbar_kws"] = {"label": cbar_label}
    sns.heatmap(table, **heatmap_kwargs)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig, ax


def plot_runtime_bars(
    summary_df: pd.DataFrame,
    *,
    palette: dict[str, tuple[float, float, float]],
):
    """Plot linear and log runtime bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    labels = summary_df.index.tolist()
    values = summary_df["mean_query_time_s"].to_numpy(dtype=float)
    colors = [method_color(label, palette) for label in labels]

    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("Mean runtime (s)")
    axes[0].set_title("Mean Runtime per Query")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(labels, values, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean runtime (s) — log scale")
    axes[1].set_title("Mean Runtime per Query (log scale)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return fig, axes


def plot_runtime_boxplot(
    df: pd.DataFrame,
    *,
    method_order: list[str],
    palette: dict[str, tuple[float, float, float]],
    method_col: str = "method_label",
):
    """Plot runtime distributions by method."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data_by_method = [
        df[df[method_col] == method]["runtime_s"].dropna().to_numpy()
        for method in method_order
    ]
    bp = ax.boxplot(data_by_method, vert=True, patch_artist=True, tick_labels=method_order)
    for patch, method in zip(bp["boxes"], method_order):
        patch.set_facecolor(method_color(method, palette))
        patch.set_alpha(0.65)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Distribution by Method")
    ax.set_xticklabels(method_order, rotation=30, ha="right")
    fig.tight_layout()
    return fig, ax


def plot_failure_stacked_bar(
    breakdown_df: pd.DataFrame,
    *,
    title: str = "Failure Types by Method",
):
    """Plot stacked failure-category counts."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    breakdown_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig, ax


def plot_validity_distance_curves(
    curve_df: pd.DataFrame,
    *,
    palette: dict[str, tuple[float, float, float]],
    title: str,
    xlabel: str = "Distance threshold",
    ylabel: str = "Validity (%)",
    ylim: tuple[float, float] = (-2.0, 102.0),
    ax=None,
):
    """Plot validity-distance curves for one metric and one dataset slice."""
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for method_label, sub in curve_df.groupby("method_label", sort=False):
        ax.plot(
            sub["epsilon"].to_numpy(dtype=float),
            sub["success_pct"].to_numpy(dtype=float),
            label=method_label,
            color=method_color(method_label, palette),
            linewidth=2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False)
    if created_fig is not None:
        created_fig.tight_layout()
        return created_fig, ax
    return ax.figure, ax


def plot_feature_change_heatmap(
    table: pd.DataFrame,
    *,
    title: str = "Feature Change Heatmap",
    cmap: str = "YlOrRd",
):
    """Plot a heatmap of mean absolute feature changes per method."""
    fig, ax = plt.subplots(
        figsize=(max(10, table.shape[1] * 0.7), max(3.5, table.shape[0] * 0.6))
    )
    sns.heatmap(
        table,
        cmap=cmap,
        linewidths=0.3,
        cbar_kws={"label": "Normalized mean |Δ|"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Method")
    fig.tight_layout()
    return fig, ax
