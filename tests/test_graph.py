"""Tests for graph utilities."""

import numpy as np

from hafc.graph import grid_edges, incidence_matrix, laplacian


class TestGraph:
    def test_grid_edges_count(self):
        # 3x3 grid: 2*3 + 3*2 = 12 edges
        edges = grid_edges(3, 3)
        assert len(edges) == 12

    def test_incidence_rows(self):
        edges = grid_edges(2, 2)
        B = incidence_matrix(4, edges)
        assert B.shape == (4, len(edges))
        # Each column sums to 0 (+1 and -1)
        assert np.allclose(np.array(B.sum(axis=0)).ravel(), 0)

    def test_laplacian_psd(self):
        edges = grid_edges(3, 3)
        L = laplacian(9, edges)
        eigvals = np.linalg.eigvalsh(L.toarray())
        assert np.all(eigvals >= -1e-10), "Laplacian should be PSD"
        # Should have exactly one zero eigenvalue (connected graph)
        assert np.sum(np.abs(eigvals) < 1e-8) == 1
