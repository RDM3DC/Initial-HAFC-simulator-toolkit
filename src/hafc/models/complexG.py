"""Complex-conductance baseline model.

Each edge carries a full complex conductance  G_k = |G_k| exp(i θ_k).
Currents are complex:  I_k = G_k · (φ_i − φ_j).
The magnitude and phase evolve independently:

    d|G|/dt = α |Re(I)| − μ |G|          (reinforce real current)
    dθ/dt   = −κ Im(I / |I|)             (phase aligns to current direction)

This is a minimal testbed for exploring complex-valued conductance
before the full EGATL entropy gate is bolted on.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..graph import grid_edges
from ..solvers import solve_potentials


@dataclass
class ComplexGConfig:
    rows: int = 6
    cols: int = 6
    source_node: int = 0
    sink_node: int = -1
    source_current: float = 1.0
    alpha: float = 1.0
    mu: float = 0.02
    kappa: float = 0.3
    dt: float = 0.02
    n_steps: int = 250
    G_init: float = 1.0
    G_min: float = 1e-6
    G_max: float = 50.0
    seed: int | None = None


def run(cfg: ComplexGConfig | None = None) -> dict:
    """Run the complex-G simulation.

    Returns
    -------
    dict with keys:
        t, edges,
        Gmag_hist  (T, E)   |G| history
        theta_hist (T, E)   θ history
        I_hist     (T, E)   complex current history
        phi_hist   (T, N)   node potentials (real solver)
        S_hist     (T,)     flow entropy
    """
    if cfg is None:
        cfg = ComplexGConfig()

    n_nodes = cfg.rows * cfg.cols
    if cfg.sink_node == -1:
        cfg.sink_node = n_nodes - 1

    edges = grid_edges(cfg.rows, cfg.cols)
    n_edges = len(edges)

    rng = np.random.default_rng(cfg.seed)
    G_mag = np.full(n_edges, cfg.G_init)
    theta = rng.uniform(-np.pi, np.pi, n_edges)

    t_arr = np.arange(cfg.n_steps) * cfg.dt

    Gmag_hist = np.zeros((cfg.n_steps, n_edges))
    theta_hist = np.zeros((cfg.n_steps, n_edges))
    I_hist = np.zeros((cfg.n_steps, n_edges), dtype=complex)
    phi_hist = np.zeros((cfg.n_steps, n_nodes))
    S_hist = np.zeros(cfg.n_steps)

    for step in range(cfg.n_steps):
        sources = np.zeros(n_nodes)
        sources[cfg.source_node] = cfg.source_current
        sources[cfg.sink_node] = -cfg.source_current

        # Solve on magnitude of G (real Kirchhoff)
        phi = solve_potentials(n_nodes, edges, G_mag, sources)

        # Complex current
        dphi = phi[edges[:, 0]] - phi[edges[:, 1]]
        G_complex = G_mag * np.exp(1j * theta)
        I_complex = G_complex * dphi

        # magnitude update: reinforce on real current
        G_mag += cfg.dt * (cfg.alpha * np.abs(I_complex.real) - cfg.mu * G_mag)
        G_mag = np.clip(G_mag, cfg.G_min, cfg.G_max)

        # Phase update: align to current direction
        I_dir = I_complex / (np.abs(I_complex) + 1e-30)
        theta -= cfg.dt * cfg.kappa * I_dir.imag
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi

        # Entropy
        p = np.abs(I_complex) / (np.abs(I_complex).sum() + 1e-30)
        S = -np.sum(p * np.log(p + 1e-30))

        # Record
        Gmag_hist[step] = G_mag
        theta_hist[step] = theta.copy()
        I_hist[step] = I_complex
        phi_hist[step] = phi
        S_hist[step] = S

    return dict(
        t=t_arr,
        edges=edges,
        Gmag_hist=Gmag_hist,
        theta_hist=theta_hist,
        I_hist=I_hist,
        phi_hist=phi_hist,
        S_hist=S_hist,
        config=cfg,
    )
