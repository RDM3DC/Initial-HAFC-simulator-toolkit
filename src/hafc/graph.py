"""Graph construction and sparse-matrix utilities for HAFC networks."""

from __future__ import annotations

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Edge-list builders
# ---------------------------------------------------------------------------

def grid_edges(rows: int, cols: int) -> np.ndarray:
    """Return an (E, 2) array of undirected edges for a 2-D grid graph.

    Node indexing is row-major: node(r, c) = r * cols + c.
    """
    edges = []
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c
            if c + 1 < cols:
                edges.append((n, n + 1))
            if r + 1 < rows:
                edges.append((n, n + cols))
    return np.asarray(edges, dtype=np.int64)


def maze_edges(
    rows: int,
    cols: int,
    wall_prob: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Grid graph with random edges removed (maze-like topology).

    Parameters
    ----------
    wall_prob : float
        Probability that each grid edge is removed.
    seed : int, optional
        RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    all_edges = grid_edges(rows, cols)
    keep = rng.random(len(all_edges)) > wall_prob
    return all_edges[keep]


def random_edges(n_nodes: int, n_edges: int, seed: int | None = None) -> np.ndarray:
    """Erdős–Rényi-style random edge list (no self-loops, undirected)."""
    rng = np.random.default_rng(seed)
    edge_set: set[tuple[int, int]] = set()
    while len(edge_set) < n_edges:
        i, j = sorted(rng.choice(n_nodes, size=2, replace=False))
        edge_set.add((i, j))
    return np.asarray(sorted(edge_set), dtype=np.int64)


# ---------------------------------------------------------------------------
# Sparse matrix constructors
# ---------------------------------------------------------------------------

def incidence_matrix(n_nodes: int, edges: np.ndarray) -> sparse.csr_matrix:
    """Signed incidence matrix B  (n_nodes × n_edges).

    Convention: for edge k = (i, j) with i < j,  B[i,k] = +1, B[j,k] = −1.
    """
    n_edges = len(edges)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    return sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_edges))


def laplacian(
    n_nodes: int,
    edges: np.ndarray,
    weights: np.ndarray | None = None,
) -> sparse.csr_matrix:
    """Weighted graph Laplacian  L = B @ diag(w) @ B.T."""
    B = incidence_matrix(n_nodes, edges)
    if weights is None:
        weights = np.ones(len(edges))
    W = sparse.diags(weights)
    return B @ W @ B.T
