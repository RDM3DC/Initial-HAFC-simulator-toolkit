"""Integrators and linear solvers for HAFC conductance dynamics."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from .graph import laplacian


# ---------------------------------------------------------------------------
# Kirchhoff potential solver
# ---------------------------------------------------------------------------

def solve_potentials(
    n_nodes: int,
    edges: np.ndarray,
    G: np.ndarray,
    sources: np.ndarray,
    ground_node: int = 0,
) -> np.ndarray:
    """Solve  L(G) · φ = s  for node potentials with one node grounded.

    Parameters
    ----------
    n_nodes : int
    edges : (E, 2) array
    G : (E,) conductance weights (real, positive).
    sources : (N,) source/sink current vector (should sum to ≈ 0).
    ground_node : int
        Index of the node whose potential is pinned to 0.

    Returns
    -------
    phi : (N,) node potentials.
    """
    L = laplacian(n_nodes, edges, weights=G)
    # Remove grounded row/col to make L non-singular
    mask = np.ones(n_nodes, dtype=bool)
    mask[ground_node] = False
    L_red = L[np.ix_(mask, mask)]
    s_red = sources[mask]
    phi_red = spsolve(L_red, s_red)
    phi = np.zeros(n_nodes)
    phi[mask] = phi_red
    return phi


def edge_currents(
    edges: np.ndarray,
    G: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Compute edge currents  I_k = G_k · (φ_i − φ_j)."""
    dphi = phi[edges[:, 0]] - phi[edges[:, 1]]
    return G * dphi


# ---------------------------------------------------------------------------
# Conductance update rules
# ---------------------------------------------------------------------------

def euler_step(
    G: np.ndarray,
    I: np.ndarray,
    alpha: float,
    mu: float,
    dt: float,
) -> np.ndarray:
    """Forward-Euler conductance update.

    G ← G + dt·(α|I| − μ·G)
    """
    return G + dt * (alpha * np.abs(I) - mu * G)


def semi_implicit_step(
    G: np.ndarray,
    I: np.ndarray,
    alpha: float,
    mu: float,
    dt: float,
) -> np.ndarray:
    """Semi-implicit conductance update (decay treated implicitly).

    G ← (G + dt·α·|I|) / (1 + dt·μ)

    This is unconditionally stable for the decay term and permits
    larger time-steps than forward Euler.
    """
    return (G + dt * alpha * np.abs(I)) / (1.0 + dt * mu)
