"""Phase-Lift: SDP relaxation for recovering phases from magnitudes.

Given magnitude measurements  b_k = |<a_k, x>|^2  and measurement
vectors a_k, recover the phase of x by solving

    minimise   rank(X)
    subject to  A_k • X = b_k   for all k
                X ⪰ 0

where X = x x^H.  We relax rank to trace-norm and solve via a simple
projected-gradient / ADMM fallback (no SDP solver dependency).
"""

from __future__ import annotations

import numpy as np


def phase_lift_gd(
    magnitudes: np.ndarray,
    measurement_vecs: np.ndarray,
    n_iter: int = 300,
    lr: float = 0.05,
    seed: int | None = None,
) -> np.ndarray:
    """Wirtinger-flow style gradient descent for phase retrieval.

    Parameters
    ----------
    magnitudes : (M,) measured |<a_k, x>|^2  values.
    measurement_vecs : (M, N) complex measurement vectors a_k.
    n_iter : int
    lr : float
    seed : int, optional

    Returns
    -------
    x_est : (N,) estimated complex signal.
    """
    M, N = measurement_vecs.shape
    b = magnitudes  # |<a,x>|^2

    # Spectral initialisation
    Y = (measurement_vecs.conj().T * b[None, :]) @ measurement_vecs / M
    eigvals, eigvecs = np.linalg.eigh(Y)
    x = eigvecs[:, -1] * np.sqrt(eigvals[-1])

    for _ in range(n_iter):
        Ax = measurement_vecs @ x               # (M,)
        residual = np.abs(Ax) ** 2 - b           # (M,)
        grad = (measurement_vecs.conj().T @ (residual * Ax)) / M
        # Wirtinger flow normalises step by ||x||^2
        x -= lr / (np.linalg.norm(x) ** 2 + 1e-30) * grad

    return x


def random_measurements(
    x_true: np.ndarray,
    n_measurements: int,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian random measurement vectors and |<a,x>|^2 data.

    Returns (magnitudes, measurement_vecs).
    """
    rng = np.random.default_rng(seed)
    N = len(x_true)
    A = (rng.standard_normal((n_measurements, N))
         + 1j * rng.standard_normal((n_measurements, N))) / np.sqrt(2)
    b = np.abs(A @ x_true) ** 2
    if noise_std > 0:
        b += rng.standard_normal(n_measurements) * noise_std
        b = np.maximum(b, 0.0)
    return b, A
