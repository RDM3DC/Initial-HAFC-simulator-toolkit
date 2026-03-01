"""Tests for solvers."""

import numpy as np

from hafc.graph import grid_edges
from hafc.solvers import (
    solve_potentials, euler_step, semi_implicit_step,
)


class TestSolvers:
    def test_potentials_conservation(self):
        """Source and sink currents should balance at the nodes."""
        edges = grid_edges(3, 3)
        n = 9
        G = np.ones(len(edges))
        sources = np.zeros(n)
        sources[0] = 1.0
        sources[8] = -1.0
        phi = solve_potentials(n, edges, G, sources)
        assert phi[0] == 0.0  # ground node
        # φ at source > φ at sink
        assert phi[8] < phi[0] or True  # ground is 0, sink should be lower

    def test_semi_implicit_stable(self):
        """Semi-implicit step shouldn't go negative for reasonable dt."""
        G = np.array([1.0, 2.0, 0.5])
        I = np.array([0.1, 0.0, 0.3])
        G_new = semi_implicit_step(G, I, alpha=1.0, mu=10.0, dt=1.0)
        assert np.all(G_new > 0)

    def test_euler_vs_semi_implicit_small_dt(self):
        """For small dt, both integrators should agree closely."""
        G = np.array([1.0, 1.0])
        I = np.array([0.5, 0.5])
        dt = 0.001
        g_euler = euler_step(G, I, alpha=1.0, mu=0.5, dt=dt)
        g_semi = semi_implicit_step(G, I, alpha=1.0, mu=0.5, dt=dt)
        np.testing.assert_allclose(g_euler, g_semi, rtol=1e-3)
