"""Shared notebook style, labels, and palettes for benchmark analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import pandas as pd
import seaborn as sns

DEFAULT_METHOD_LABELS = {
    "certcf": "CertCF",
    "cpp": "CertCF",
    "CPP": "CertCF",
    "dice": "DiCE",
    "face": "FACE",
    "nearest_neighbor": "Nearest Neighbor",
    "nn": "Nearest Neighbor",
    "growing_spheres": "Growing Spheres",
    "gs": "Growing Spheres",
    "wachter": "Wachter",
}

DEFAULT_METHOD_COLORS = {
    "certcf": "#d62728",
    "CertCF": "#d62728",
    "cpp": "#d62728",
    "CPP": "#d62728",
    "dice": "#1f77b4",
    "DiCE": "#1f77b4",
    "face": "#2ca02c",
    "FACE": "#2ca02c",
    "growing_spheres": "#9467bd",
    "gs": "#9467bd",
    "Growing Spheres": "#9467bd",
    "nearest_neighbor": "#ff7f0e",
    "nn": "#ff7f0e",
    "Nearest Neighbor": "#ff7f0e",
    "wachter": "#8c564b",
    "Wachter": "#8c564b",
}


def _canonical_method_key(method: str) -> str:
    """Normalize common method-name variants for labels and palette lookup."""
    normalized = method.strip().replace("-", "_").replace(" ", "_")
    lowered = normalized.lower()
    if lowered in {"certcf", "cert_cf", "cpp"}:
        return "certcf"
    if lowered in {"dice"}:
        return "dice"
    if lowered in {"face"}:
        return "face"
    if lowered in {"nearest_neighbor", "nearestneighbour", "nn"}:
        return "nearest_neighbor"
    if lowered in {"growing_spheres", "growingspheres", "gs"}:
        return "growing_spheres"
    if lowered in {"wachter"}:
        return "wachter"
    return method


def setup_notebook_style() -> None:
    """Apply the shared plotting style used by benchmark notebooks."""
    sns.set_theme(style="whitegrid", palette="tab10")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.figsize": (10, 4),
        }
    )


def method_palette(methods: list[str]) -> dict[str, tuple[float, float, float]]:
    """Return a stable color mapping for the provided method labels."""
    fallback = iter(sns.color_palette("tab10", len(methods) + 4))
    palette: dict[str, tuple[float, float, float]] = {}
    for method in methods:
        canonical = _canonical_method_key(method)
        display = DEFAULT_METHOD_LABELS.get(canonical, canonical)
        if method in DEFAULT_METHOD_COLORS:
            palette[method] = tuple(to_rgb(DEFAULT_METHOD_COLORS[method]))
        elif canonical in DEFAULT_METHOD_COLORS:
            palette[method] = tuple(to_rgb(DEFAULT_METHOD_COLORS[canonical]))
        elif display in DEFAULT_METHOD_COLORS:
            palette[method] = tuple(to_rgb(DEFAULT_METHOD_COLORS[display]))
        else:
            palette[method] = tuple(next(fallback))
    return palette


def method_color(
    method: str,
    palette: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[float, float, float]:
    """Resolve one method color robustly across display-label variants."""
    if palette is not None and method in palette:
        return palette[method]

    canonical = _canonical_method_key(method)
    display = DEFAULT_METHOD_LABELS.get(canonical, canonical)

    if palette is not None:
        for candidate in (canonical, display):
            if candidate in palette:
                return palette[candidate]

    for candidate in (method, canonical, display):
        if candidate in DEFAULT_METHOD_COLORS:
            return tuple(to_rgb(DEFAULT_METHOD_COLORS[candidate]))

    return tuple(sns.color_palette("tab10", 1)[0])


def bold_best_values(
    styler,
    data: pd.DataFrame,
    *,
    maximize: list[str] | None = None,
    minimize: list[str] | None = None,
):
    """Apply bold styling to the best numeric values in each selected column."""
    maximize = list(maximize or [])
    minimize = list(minimize or [])

    def _styles(_: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)

        for columns, reducer in ((maximize, "max"), (minimize, "min")):
            for column in columns:
                if column not in data.columns:
                    continue
                series = pd.to_numeric(data[column], errors="coerce")
                if not series.notna().any():
                    continue
                best = getattr(series, reducer)()
                styles.loc[series.eq(best), column] = "font-weight: bold"
        return styles

    return styler.apply(_styles, axis=None)


def style_method_table(
    data: pd.DataFrame,
    *,
    fmt=None,
    maximize: list[str] | None = None,
    minimize: list[str] | None = None,
    gradient_cmap: str | None = None,
    gradient_axis: int | None = 0,
):
    """Format a method-comparison table and bold the best values by column."""
    styler = data.style
    if fmt is not None:
        styler = styler.format(fmt)
    if gradient_cmap is not None:
        styler = styler.background_gradient(cmap=gradient_cmap, axis=gradient_axis)
    return bold_best_values(
        styler,
        data,
        maximize=maximize,
        minimize=minimize,
    )
