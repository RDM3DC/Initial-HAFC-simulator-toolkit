"""Toy EGATL demo — runs the full leaderboard-aligned simulator on a small graph.

Usage:
    python examples/toy_egatl.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from hafc.models.egatl import (
    Graph,
    EGATLParams,
    EntropyParams,
    RulerParams,
    simulate,
    default_toy_graph,
)


def quick_dashboard(out: dict, title: str = "EGATL (#1/#15/#16/#17)") -> None:
    t = out["t"]
    Gmag = np.abs(out["Gc"])
    S = out["S"]
    pi_a = out["pi_a"]
    rb = out["r_b_any"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    for i, (u, v) in enumerate(out["edges"]):
        axs[0, 0].plot(t, Gmag[:, i], lw=1.2, label=f"{u}-{v}")
    axs[0, 0].set(title="|G_e|(t)", xlabel="time", ylabel="|G|")
    axs[0, 0].legend(fontsize=9, ncol=2)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t, S, lw=2, label="S(t)")
    axs[0, 1].axhline(float(np.mean(S)), ls="--", alpha=0.5, label="mean")
    axs[0, 1].set(title="Entropy S(t)", xlabel="time")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(t, pi_a, lw=2)
    axs[1, 0].axhline(math.pi, color="gray", ls="--", alpha=0.7, label="pi_0")
    axs[1, 0].set(title="Adaptive ruler pi_a(t)", xlabel="time", ylabel="pi_a")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(t, rb, lw=2, label="r_b_any")
    axs[1, 1].plot(t, out["r_b_mean"], lw=2, label="r_b_mean")
    axs[1, 1].step(t, out["flip_any"], where="post", lw=1.2, alpha=0.7, label="flip_any")
    axs[1, 1].set(title="Parity diagnostics", xlabel="time")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.show()


def main():
    graph, s, t = default_toy_graph()
    out = simulate(
        graph, s, t,
        T=40.0, dt=0.05,
        eg=EGATLParams(alpha0=1.0, S_c=1.0, dS=0.35, mu0=0.55,
                       S0=1.0, budget_re=10.0, lambda_s=0.15),
        ent=EntropyParams(S_init=0.5, S_eq=0.5, gamma=0.25,
                          kappa_slip=0.15, Tij=1.0),
        ruler=RulerParams(pi0=math.pi, pi_init=math.pi,
                          alpha_pi=0.25, mu_pi=0.20),
    )
    quick_dashboard(out)

    print(f"\nFinal r_b_any={out['r_b_any'][-1]:.4f},"
              f" r_b_mean={out['r_b_mean'][-1]:.6f}")

    Gf = out["Gc"][-1]
    edges = out["edges"]
    idx = np.argsort(-np.abs(Gf))
    print("\nTop edges by final |G|:")
    for j in idx[:8]:
        u, v = edges[j]
        print(f"  {u}-{v}:  G={Gf[j].real:+.4f}{Gf[j].imag:+.4f}j"
              f"   |G|={abs(Gf[j]):.4f}")


if __name__ == "__main__":
    main()
