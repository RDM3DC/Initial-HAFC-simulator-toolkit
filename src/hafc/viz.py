"""Visualisation helpers for HAFC simulations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_network(
    pos: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    title: str = "",
    node_vals: np.ndarray | None = None,
) -> plt.Axes:
    """Draw a 2-D network with edge widths proportional to *weights*.

    Parameters
    ----------
    pos : (N, 2) node positions.
    edges : (E, 2) edge list.
    weights : (E,) optional edge weights (for colour/width).
    node_vals : (N,) optional node scalar field (for colour).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Edges
    segs = pos[edges]  # (E, 2, 2)
    if weights is not None:
        w = np.abs(weights)
        w_norm = w / (w.max() + 1e-12)
        lc = LineCollection(segs, linewidths=1 + 3 * w_norm, cmap=cmap)
        lc.set_array(w_norm)
        ax.add_collection(lc)
    else:
        lc = LineCollection(segs, linewidths=0.8, colors="0.6")
        ax.add_collection(lc)

    # Nodes
    if node_vals is not None:
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=node_vals, s=30,
                        zorder=3, cmap=cmap, edgecolors="k", linewidths=0.5)
        plt.colorbar(sc, ax=ax, shrink=0.6)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], s=20, c="k", zorder=3)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.autoscale_view()
    return ax


def plot_timeseries(
    t: np.ndarray,
    series: dict[str, np.ndarray],
    title: str = "",
    ylabel: str = "",
) -> plt.Figure:
    """Plot one or more time-series on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, y in series.items():
        ax.plot(t, y, label=label)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def grid_positions(rows: int, cols: int) -> np.ndarray:
    """Return (rows*cols, 2) positions for a regular grid layout."""
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
