"""Entropy-Gated Analog Transport Layer (EGATL) simulator.

Aligned with TopEquations leaderboard #1/#15/#16/#17.

Implements (core):
  (#1) Entropy-gated Phase-Lifted Complex Conductance Update:
    d/dt G_ij = alpha_G(S) |I_ij| exp(i theta_R,ij)  -  mu_G(S) G_ij

  Entropy dynamics (from leaderboard #1 card):
    dS/dt = Sum |I_ij|^2 / T_ij * Re(1/G_ij)
            + kappa Sum |dw_ij|
            - gamma (S - S_eq)

  (#16) Phase-Lift update (per edge):
    theta_R,e(k) = theta_R,e(k-1) + clip(wrapToPi(phi_e - theta_R,e), -pi_a, +pi_a)

  (#17) Adaptive ruler:
    dpi_a/dt = alpha_pi S  -  mu_pi (pi_a - pi_0)

Optional:
  (#15) Phase-coupled suppression term:
    dG/dt += -lambda_s G * sin^2( theta_R / (2 pi_a) )

Edge-wise parity tracking:
  b_e = (-1)^{w_e}, flips_e = 1[b_e != b_e_prev]
  Reports flip_any, flip_mean, r_b_any, r_b_mean.

Semi-implicit stabilised conductance update for the decay term:
  G <- (G + dt * drive) / (1 + dt * decay)

Notes
-----
- Uses a complex nodal solve each step (grounded) with sparse GMRES.
- Enforces a passivity clamp: Re(G) >= G_min and bounds Im(G).
- This is a research simulator, not a validated circuit device model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import gmres


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Vectorised wrap to (-pi, pi]."""
    a = (x + math.pi) % (2 * math.pi) - math.pi
    return np.where(a <= -math.pi, a + 2 * math.pi, a)


def _logistic(x: float) -> float:
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Sparse complex-nodal solver
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    n: int
    edges: List[Tuple[int, int]]


def build_nodal_matrix(
    n: int,
    edges: List[Tuple[int, int]],
    Y: np.ndarray,
) -> csr_matrix:
    """Sparse nodal admittance matrix for undirected edges."""
    m = len(edges)
    rows = np.empty(4 * m, dtype=int)
    cols = np.empty(4 * m, dtype=int)
    data = np.empty(4 * m, dtype=Y.dtype)

    for i, (u, v) in enumerate(edges):
        k = 4 * i
        rows[k], cols[k], data[k] = u, u, Y[i]
        rows[k + 1], cols[k + 1], data[k + 1] = v, v, Y[i]
        rows[k + 2], cols[k + 2], data[k + 2] = u, v, -Y[i]
        rows[k + 3], cols[k + 3], data[k + 3] = v, u, -Y[i]

    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def grounded_gmres(
    M: csr_matrix,
    b: np.ndarray,
    ground: int,
    x0: Optional[np.ndarray] = None,
    rtol: float = 1e-10,
    maxiter: int = 2500,
) -> Tuple[np.ndarray, int]:
    """Solve Mx = b with x[ground] = 0 via GMRES on the reduced system."""
    n = b.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[ground] = False

    Mr = M[mask][:, mask]
    br = b[mask]
    x0r = None if x0 is None else x0[mask].copy()

    xr, info = gmres(Mr, br, x0=x0r, rtol=rtol, atol=0.0,
                      maxiter=maxiter, restart=50)
    x = np.zeros(n, dtype=complex)
    x[mask] = xr
    return x, int(info)


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EGATLParams:
    # alpha gate: alpha(S) = alpha0 / [1 + exp((S - Sc)/dS)]
    alpha0: float = 1.0
    S_c: float = 1.0
    dS: float = 0.35

    # mu gate: mu(S) = mu0 * (S / S0)
    mu0: float = 0.55
    S0: float = 1.0

    # complex G clamps
    G_min: float = 1e-3
    G_max: float = 50.0
    G_imag_max: float = 50.0

    # optional budget on sum(Re(G))
    budget_re: Optional[float] = 10.0

    # phase-coupled suppression (#15)
    lambda_s: float = 0.0  # set > 0 to enable

    # symmetry-breaking noise scale on dG
    noise_scale: float = 1e-6


@dataclass
class EntropyParams:
    S_init: float = 0.5
    S_eq: float = 0.5
    gamma: float = 0.25          # relaxation to S_eq
    kappa_slip: float = 0.15     # slip entropy weight
    Tij: float = 1.0             # constant T_ij (Issue #4: make edge-wise)

    # If True, clamp Re(1/G) nonneg to avoid runaway when Re(G) is tiny.
    clamp_Re_invG_nonneg: bool = True


@dataclass
class RulerParams:
    pi0: float = math.pi
    pi_init: float = math.pi
    alpha_pi: float = 0.25
    mu_pi: float = 0.20
    pi_min: float = 0.25
    pi_max: float = 2.75 * math.pi


# ---------------------------------------------------------------------------
# Core gates
# ---------------------------------------------------------------------------

def alpha_G_of_S(S: float, p: EGATLParams) -> float:
    return p.alpha0 * _logistic(-(S - p.S_c) / max(1e-12, p.dS))


def mu_G_of_S(S: float, p: EGATLParams) -> float:
    return p.mu0 * (S / max(1e-12, p.S0))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    graph: Graph,
    source: int,
    sink: int,
    T: float = 40.0,
    dt: float = 0.05,
    seed: int = 0,
    eg: EGATLParams | None = None,
    ent: EntropyParams | None = None,
    ruler: RulerParams | None = None,
    rtol: float = 1e-10,
    maxiter: int = 2500,
) -> Dict[str, np.ndarray]:
    """Run the full EGATL simulation.

    Returns
    -------
    dict with keys
        t, Gc, I, phi, theta_R_e, w_e, S, pi_a,
        flip_any, flip_mean, r_b_any, r_b_mean,
        solve_info, edges
    """
    if eg is None:
        eg = EGATLParams()
    if ent is None:
        ent = EntropyParams()
    if ruler is None:
        ruler = RulerParams()

    rng = np.random.default_rng(seed)
    m = len(graph.edges)
    K = int(np.ceil(T / dt)) + 1
    t_arr = np.linspace(0.0, T, K)

    # State
    Gc = np.ones(m, dtype=complex)
    S = float(ent.S_init)
    pi_a = float(ruler.pi_init)
    theta_R = np.zeros(m)
    theta_prev = np.zeros(m)
    w_prev = np.zeros(m, dtype=int)
    b_prev = np.ones(m, dtype=int)  # per-edge parity state
    flip_any_count = 0
    flip_edge_total = 0
    phi_prev = np.zeros(graph.n, dtype=complex)

    # History
    G_hist = np.zeros((K, m), dtype=complex)
    I_hist = np.zeros((K, m), dtype=complex)
    phi_hist = np.zeros((K, graph.n), dtype=complex)
    thetaR_hist = np.zeros((K, m))
    w_hist = np.zeros((K, m), dtype=int)
    S_hist = np.zeros(K)
    pi_hist = np.zeros(K)
    flip_any_hist = np.zeros(K, dtype=int)
    flip_mean_hist = np.zeros(K)
    rb_any_hist = np.zeros(K)
    rb_mean_hist = np.zeros(K)
    info_hist = np.zeros(K, dtype=int)

    for k in range(K):
        # 1) Complex nodal solve
        bvec = np.zeros(graph.n, dtype=complex)
        bvec[source] = 1.0
        bvec[sink] = -1.0

        M = build_nodal_matrix(graph.n, graph.edges, Gc)
        phi, info = grounded_gmres(M, bvec, ground=sink,
                                   x0=phi_prev, rtol=rtol, maxiter=maxiter)
        phi_prev = phi
        info_hist[k] = info

        # 2) Edge currents and raw phases
        I = np.zeros(m, dtype=complex)
        theta = np.zeros(m)
        for e, (u, v) in enumerate(graph.edges):
            dV = phi[u] - phi[v]
            I[e] = Gc[e] * dV
            theta[e] = float(np.angle(I[e] + 1e-18))

        # 3) Phase-Lift update (#16)
        r = wrap_to_pi(theta - theta_prev)
        r_clip = np.clip(r, -pi_a, pi_a)
        theta_R = theta_R + r_clip
        theta_prev = theta

        w = np.round(theta_R / (2 * math.pi)).astype(int)
        dW = np.abs(w - w_prev).astype(float)
        w_prev = w

        # Edge-wise parity and flips
        b = np.where((w % 2) == 0, 1, -1).astype(int)
        flips_e = (b != b_prev).astype(int)
        b_prev = b

        flip_any = int(np.any(flips_e))
        flip_mean = float(np.mean(flips_e))
        flip_any_count += flip_any
        flip_edge_total += int(np.sum(flips_e))

        # 4) Entropy update (#1 differential)
        Re_invG = np.real(1.0 / (Gc + 1e-18))
        if ent.clamp_Re_invG_nonneg:
            Re_invG = np.maximum(0.0, Re_invG)
        term1 = float(np.sum(
            np.abs(I) ** 2 / max(1e-12, ent.Tij) * Re_invG
        ))
        term2 = float(ent.kappa_slip * np.sum(dW))
        term3 = float(-ent.gamma * (S - ent.S_eq))
        S = float(max(0.0, S + dt * (term1 + term2 + term3)))

        # 5) Adaptive ruler (#17)
        dpi = ruler.alpha_pi * S - ruler.mu_pi * (pi_a - ruler.pi0)
        pi_a = float(np.clip(pi_a + dt * dpi, ruler.pi_min, ruler.pi_max))

        # 6) Conductance update (#1) + optional suppression (#15)
        #    Semi-implicit: G <- (G + dt*drive) / (1 + dt*decay)
        aS = alpha_G_of_S(S, eg)
        mS = mu_G_of_S(S, eg)

        drive = aS * np.abs(I) * np.exp(1j * theta_R)
        decay = mS

        if eg.lambda_s > 0:
            sup = np.sin(theta_R / (2.0 * pi_a + 1e-18)) ** 2
            decay = decay + eg.lambda_s * sup

        Gc = (Gc + dt * drive) / (1.0 + dt * decay)

        # symmetry-breaking noise
        if eg.noise_scale > 0:
            Gc += eg.noise_scale * (
                rng.normal(size=m) + 1j * rng.normal(size=m)
            )

        # 7) Clamps
        Re = np.clip(Gc.real, eg.G_min, eg.G_max)
        Im = np.clip(Gc.imag, -eg.G_imag_max, eg.G_imag_max)
        Gc = Re + 1j * Im

        if eg.budget_re is not None:
            sRe = float(np.sum(Gc.real))
            if sRe > eg.budget_re and sRe > 0:
                scale = eg.budget_re / sRe
                Gc = Gc.real * scale + 1j * Gc.imag * scale

        # Record
        G_hist[k] = Gc
        I_hist[k] = I
        phi_hist[k] = phi
        thetaR_hist[k] = theta_R
        w_hist[k] = w
        S_hist[k] = S
        pi_hist[k] = pi_a
        flip_any_hist[k] = flip_any
        flip_mean_hist[k] = flip_mean
        rb_any_hist[k] = flip_any_count / max(1, k)
        rb_mean_hist[k] = flip_edge_total / max(1, k * m)

    return {
        "t": t_arr,
        "Gc": G_hist,
        "I": I_hist,
        "phi": phi_hist,
        "theta_R_e": thetaR_hist,
        "w_e": w_hist,
        "S": S_hist,
        "pi_a": pi_hist,
        "flip_any": flip_any_hist,
        "flip_mean": flip_mean_hist,
        "r_b_any": rb_any_hist,
        "r_b_mean": rb_mean_hist,
        "solve_info": info_hist,
        "edges": np.array(graph.edges, dtype=int),
    }


# ---------------------------------------------------------------------------
# Convenience: grid-based run (matches skeleton API)
# ---------------------------------------------------------------------------

def default_toy_graph() -> Tuple[Graph, int, int]:
    """6-node Wheatstone-bridge-like test graph."""
    edges = [
        (0, 1), (1, 2), (2, 5),
        (0, 3), (3, 4), (4, 5),
        (1, 3), (2, 4),
    ]
    return Graph(n=6, edges=edges), 0, 5


def run(
    graph: Graph | None = None,
    source: int = 0,
    sink: int = 5,
    T: float = 40.0,
    dt: float = 0.05,
    seed: int = 0,
    eg: EGATLParams | None = None,
    ent: EntropyParams | None = None,
    ruler: RulerParams | None = None,
) -> Dict[str, np.ndarray]:
    """High-level entry point matching the package API convention."""
    if graph is None:
        graph, source, sink = default_toy_graph()
    return simulate(graph, source, sink, T=T, dt=dt, seed=seed,
                    eg=eg, ent=ent, ruler=ruler)
