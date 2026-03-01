"""Maze pruning demo — DC/AC ARP on a grid with random walls.

Usage:
    python examples/maze_dc.py
"""

import numpy as np
import matplotlib.pyplot as plt

from hafc.models.dc_ac import DCACConfig, run
from hafc.viz import plot_network, grid_positions


def main():
    cfg = DCACConfig(
        rows=10, cols=10,
        wall_prob=0.25,
        alpha=1.5, mu=0.02,
        dt=0.05, n_steps=300,
        seed=42,
    )

    out = run(cfg)
    pos = grid_positions(cfg.rows, cfg.cols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Initial conductances
    plot_network(pos, out["edges"], weights=out["G_hist"][0],
                 ax=axes[0], title="t = 0 (initial)")

    # Mid-run
    mid = cfg.n_steps // 2
    plot_network(pos, out["edges"], weights=out["G_hist"][mid],
                 ax=axes[1], title=f"t = {out['t'][mid]:.1f}")

    # Final
    plot_network(pos, out["edges"], weights=out["G_hist"][-1],
                 ax=axes[2], title=f"t = {out['t'][-1]:.1f} (final)")

    fig.suptitle("DC maze pruning — conductance evolution", fontsize=14)
    fig.tight_layout()
    plt.show()

    # Entropy time-series
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(out["t"], out["S_hist"])
    ax2.set(xlabel="time", ylabel="S", title="Flow entropy")
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
