"""Phase-Lift sanity tests."""

import numpy as np

from hafc.phase_lift import phase_lift_gd, random_measurements


class TestPhaseLift:
    def test_recovery_noiseless(self):
        """Wirtinger flow should recover a random signal (up to global phase)."""
        rng = np.random.default_rng(42)
        N = 4
        x_true = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x_true /= np.linalg.norm(x_true)

        M = 20 * N  # well oversampled
        b, A = random_measurements(x_true, M, noise_std=0.0, seed=42)
        x_est = phase_lift_gd(b, A, n_iter=1000, lr=0.01, seed=0)

        # Align global phase: find θ such that e^{iθ} x_est ≈ x_true
        phase = np.exp(1j * np.angle(np.vdot(x_est, x_true)))
        x_aligned = x_est * phase
        err = np.linalg.norm(x_aligned - x_true) / np.linalg.norm(x_true)
        assert err < 0.15, f"Recovery error {err:.4f} too large"

    def test_measurements_shape(self):
        x = np.array([1.0 + 0j, 0.0 + 1j])
        b, A = random_measurements(x, 20, seed=7)
        assert b.shape == (20,)
        assert A.shape == (20, 2)
        assert np.all(b >= 0)
