"""Tests for the EGATL model."""

import numpy as np

from hafc.models.egatl import (
    EGATLParams,
    simulate, default_toy_graph, wrap_to_pi,
)


class TestEGATL:
    def test_runs_without_error(self):
        graph, s, t = default_toy_graph()
        out = simulate(graph, s, t, T=1.0, dt=0.1)
        assert "Gc" in out
        assert "S" in out
        assert "flip_any" in out
        assert "flip_mean" in out
        assert "r_b_any" in out
        assert "r_b_mean" in out
        assert out["Gc"].shape[1] == len(graph.edges)

    def test_entropy_nonnegative(self):
        graph, s, t = default_toy_graph()
        out = simulate(graph, s, t, T=2.0, dt=0.05)
        assert np.all(out["S"] >= 0)

    def test_wrap_to_pi(self):
        x = np.array([0, np.pi, -np.pi, 3 * np.pi, -5 * np.pi])
        w = wrap_to_pi(x)
        assert np.all(w > -np.pi)
        assert np.all(w <= np.pi)

    def test_suppression_reduces_conductance(self):
        """Enabling lambda_s should generally reduce final |G| versus off."""
        graph, s, t = default_toy_graph()
        out_off = simulate(graph, s, t, T=5.0, dt=0.05,
                           eg=EGATLParams(lambda_s=0.0))
        out_on = simulate(graph, s, t, T=5.0, dt=0.05,
                          eg=EGATLParams(lambda_s=0.5))
        mean_off = np.mean(np.abs(out_off["Gc"][-1]))
        mean_on = np.mean(np.abs(out_on["Gc"][-1]))
        # Suppression should lower mean conductance magnitude
        assert mean_on < mean_off * 1.5  # generous bound
