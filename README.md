# HAFC — Hybrid Analog-Flow Computing Simulator

A research toolkit for simulating adaptive-conductance networks with
entropy-gated dynamics, phase-lift, and complex-valued transport.

## Models

| Model | File | Description |
|-------|------|-------------|
| **DC/AC ARP** | `src/hafc/models/dc_ac.py` | Sparse Kirchhoff network with reinforcement/decay conductance laws |
| **EGATL** | `src/hafc/models/egatl.py` | Full leaderboard-aligned (#1/#15/#16/#17) entropy-gated complex-conductance simulator |
| **Complex G** | `src/hafc/models/complexG.py` | Minimal complex-conductance testbed |

## Quick start

```bash
# Install (editable)
pip install -e ".[dev]"

# Run tests
pytest -v

# Run examples
python examples/maze_dc.py
python examples/toy_egatl.py
```

## Package layout

```
src/hafc/
  graph.py        — Graph construction, incidence/Laplacian matrices
  solvers.py      — Kirchhoff solver, Euler & semi-implicit integrators
  phase_lift.py   — Wirtinger-flow phase retrieval
  viz.py          — Network and time-series plotting
  models/
    dc_ac.py      — DC/AC adaptive reconfigurable processor
    egatl.py      — Entropy-Gated Analog Transport Layer
    complexG.py   — Complex conductance baseline

examples/
  maze_dc.py      — Maze pruning demo
  toy_egatl.py    — EGATL dashboard demo
  ml/             — ML integration demos (requires torch/pennylane)

tests/            — pytest suite
```

## TopEquations alignment (EGATL)

The EGATL model implements:

- **#1** — Entropy-gated complex conductance update with differential entropy dynamics
- **#15** — Phase-coupled suppression term (optional `lambda_s`)
- **#16** — Phase-Lift update with adaptive clipping per edge
- **#17** — Adaptive ruler `π_a` dynamics

## Roadmap

See GitHub Issues for planned improvements:

1. Unified `RunConfig` / `RunOutput` across models
2. Semi-implicit integrator everywhere (already in `solvers.py`)
3. Edge-wise parity metrics for EGATL
4. Edge-wise temperature `T_ij`
5. Photonic block module

## License

MIT
