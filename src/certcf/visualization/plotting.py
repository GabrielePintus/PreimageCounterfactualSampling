"""Plotting utilities for polytope visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import Polygon, MultiPolygon

from ..geometry import make_polygon


def plot_geom(
    ax: Axes,
    geom: Polygon | MultiPolygon,
    color,
    alpha: float = 0.35,
    label: str | None = None,
    linewidth: float = 0.8
) -> None:
    """
    Plot a Polygon or MultiPolygon on matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    geom : Polygon or MultiPolygon
        The geometry to plot.
    color
        Matplotlib color specification.
    alpha : float, optional
        Fill transparency (default: 0.35).
    label : str, optional
        Label for legend (applied to first geometry only).
    linewidth : float, optional
        Edge line width (default: 0.8).
    """
    # Handle both Polygon and MultiPolygon
    geoms = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)

    for i, g in enumerate(geoms):
        if g.is_empty or g.area < 1e-14:
            continue

        xs, ys = g.exterior.xy
        ax.fill(xs, ys, alpha=alpha, color=color,
                label=label if i == 0 else None)
        ax.plot(xs, ys, lw=linewidth, color=color, alpha=0.6)


def plot_polytopes(
    bounds: dict,
    eps,
    n_classes: int,
    norm: int | float,
    figsize: tuple = (7, 7),
    cmap_name: str = 'tab10'
) -> tuple:
    """
    Plot individual certified polytopes for all classes.

    Each polytope is drawn as a filled polygon around its center point.

    Parameters
    ----------
    bounds : dict
        Bounds dictionary from PreimageApproximation.compute_all_bounds().
        If the dict contains an 'eps' key per class, it is used in preference
        to the ``eps`` argument.
    eps : float or np.ndarray
        Perturbation radius used for clipping.  Can be a scalar or a 1-D
        array of shape ``(N,)`` for per-sample radii.  Ignored when
        ``bounds[label]['eps']`` is present.
    n_classes : int
        Number of classes.
    norm : int or float
        Lp norm used to define the exact certified trust region.
    figsize : tuple, optional
        Figure size (default: (7, 7)).
    cmap_name : str, optional
        Colormap name (default: 'tab10').

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    cmap = plt.cm.get_cmap(cmap_name, n_classes)
    fig, ax = plt.subplots(figsize=figsize)

    for label in range(n_classes):
        bd = bounds[label]
        color = cmap(label)

        # Prefer per-sample eps stored in the bounds dict
        eps_class = bd.get('eps', eps)

        # Plot sample points
        ax.scatter(bd['X'][:, 0], bd['X'][:, 1],
                   s=12, alpha=0.4, color=color, edgecolors='none')

        # Plot polytopes
        for i in range(bd['lA'].shape[0]):
            eps_i = float(eps_class[i]) if isinstance(eps_class, np.ndarray) else eps_class
            poly = make_polygon(bd['lA'][i], bd['lbias'][i], bd['X'][i], eps_i, norm=norm)
            if poly is None:
                continue

            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.15, color=color)
            ax.plot(xs, ys, lw=0.7, alpha=0.5, color=color)

    ax.set_aspect('equal')
    ax.set_title('Certified inner polytopes – all classes')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    plt.tight_layout()

    return fig, ax


def plot_class_unions(
    class_unions: dict,
    bounds: dict,
    n_classes: int,
    figsize: tuple = (7, 7),
    cmap_name: str = 'tab10'
) -> tuple:
    """
    Plot certified class unions.

    Parameters
    ----------
    class_unions : dict
        Dictionary of certified class unions.
    bounds : dict
        Bounds dictionary for sample points.
    n_classes : int
        Number of classes.
    figsize : tuple, optional
        Figure size (default: (14, 6)).
    cmap_name : str, optional
        Colormap name (default: 'tab10').

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    cmap = plt.cm.get_cmap(cmap_name, n_classes)
    fig, ax = plt.subplots(figsize=figsize)

    for label in range(n_classes):
        ax.scatter(
            bounds[label]['X'][:, 0],
            bounds[label]['X'][:, 1],
            s=10,
            alpha=0.35,
            color=cmap(label),
            edgecolors='none',
        )
        plot_geom(ax, class_unions[label], color=cmap(label), label=f'class {label}')

    ax.set_aspect('equal')
    ax.set_title('Certified class unions')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()

    return fig, ax


def plot_overlap_heatmap(
    overlap_data: dict,
    figsize: tuple = (6, 5),
    cmap_name: str = 'YlOrRd'
) -> tuple:
    """
    Plot pairwise overlap area heatmap.

    Parameters
    ----------
    overlap_data : dict
        Dictionary with 'overlap' (2D array) and 'labels' (list).
    figsize : tuple, optional
        Figure size (default: (6, 5)).
    cmap_name : str, optional
        Colormap name (default: 'YlOrRd').

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    overlap = np.array(overlap_data['overlap'])
    labels = overlap_data['labels']
    n = len(labels)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(overlap, cmap=cmap_name)
    plt.colorbar(im, ax=ax, label='Intersection area')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    ax.set_title('Pairwise overlap (certified unions)')

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text_color = 'white' if overlap[i, j] > overlap.max() * 0.6 else 'black'
            ax.text(j, i, f'{overlap[i, j]:.4f}',
                    ha='center', va='center', fontsize=7, color=text_color)

    plt.tight_layout()

    return fig, ax
