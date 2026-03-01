"""Sparse DC/AC Analog Reconfigurable Processor (ARP) simulator.

Implements Kirchhoff-law network dynamics with adaptive conductances:

  1. Solve  L(G) φ = s          (node potentials)
  2. I_k = G_k (φ_i − φ_j)     (edge currents)
  3. G ← G + dt (α|I| − μ G)   (reinforce / decay)

AC mode adds a time-dependent sinusoidal source term.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..graph import grid_edges, maze_edges
from ..solvers import solve_potentials, edge_currents, euler_step


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DCACConfig:
    rows: int = 8
    cols: int = 8
    source_node: int = 0
    sink_node: int = -1          # -1 → last node
    source_current: float = 1.0
    alpha: float = 1.0           # reinforcement rate
    mu: float = 0.01             # decay rate
    dt: float = 0.05
    n_steps: int = 200
    G_init: float = 1.0
    G_min: float = 1e-6
    G_max: float = 50.0
    ac_freq: float = 0.0        # 0 → DC mode
    ac_amp: float = 1.0
    wall_prob: float = 0.0      # >0 → maze topology
    seed: int | None = None


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def run(cfg: DCACConfig | None = None) -> dict:
    """Run a DC or AC ARP simulation and return history dict.

    Returns
    -------
    dict with keys:
        t        (T,)      time array
        edges    (E, 2)    edge list
        G_hist   (T, E)    conductance history
        I_hist   (T, E)    current history
        phi_hist (T, N)    potential history
        S_hist   (T,)      entropy  S = −Σ p_k ln p_k  (p ∝ |I|)
    """
    if cfg is None:
        cfg = DCACConfig()

    n_nodes = cfg.rows * cfg.cols
    if cfg.sink_node == -1:
        cfg.sink_node = n_nodes - 1

    # Build topology
    if cfg.wall_prob > 0:
        edges = maze_edges(cfg.rows, cfg.cols, cfg.wall_prob, cfg.seed)
    else:
        edges = grid_edges(cfg.rows, cfg.cols)
    n_edges = len(edges)

    # Initialise
    G = np.full(n_edges, cfg.G_init)
    t_arr = np.arange(cfg.n_steps) * cfg.dt

    G_hist = np.zeros((cfg.n_steps, n_edges))
    I_hist = np.zeros((cfg.n_steps, n_edges))
    phi_hist = np.zeros((cfg.n_steps, n_nodes))
    S_hist = np.zeros(cfg.n_steps)

    for step in range(cfg.n_steps):
        # Source vector
        amp = cfg.source_current
        if cfg.ac_freq > 0:
            amp *= cfg.ac_amp * np.sin(2 * np.pi * cfg.ac_freq * t_arr[step])
        sources = np.zeros(n_nodes)
        sources[cfg.source_node] = amp
        sources[cfg.sink_node] = -amp

        # Kirchhoff solve
        phi = solve_potentials(n_nodes, edges, G, sources)
        I = edge_currents(edges, G, phi)

        # Record
        G_hist[step] = G
        I_hist[step] = I
        phi_hist[step] = phi

        # Entropy from normalised |I|
        p = np.abs(I) / (np.abs(I).sum() + 1e-30)
        S_hist[step] = -np.sum(p * np.log(p + 1e-30))

        # Update conductances
        G = euler_step(G, I, cfg.alpha, cfg.mu, cfg.dt)
        G = np.clip(G, cfg.G_min, cfg.G_max)

    return dict(
        t=t_arr,
        edges=edges,
        G_hist=G_hist,
        I_hist=I_hist,
        phi_hist=phi_hist,
        S_hist=S_hist,
        config=cfg,
    )
